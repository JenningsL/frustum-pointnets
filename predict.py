import tensorflow as tf
import os
import sys
import argparse
import numpy as np
import pickle

from avod.core import box_3d_encoder
from avod.core import evaluator_utils
from avod.core import summary_utils
from avod.core import trainer_utils
from avod.core import constants
from avod.core.models.rpn_model import RpnModel
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from wavedata.tools.core import calib_utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import frustum_pointnets_v2
sys.path.append(os.path.join(ROOT_DIR, 'avod_prop'))

class ProposalObject(object):
    def __init__(self, box_3d, score=0.0, type='Car', roi_features=None):
        # [x, y, z, l, w, h, ry]
        self.t = box_3d[0:3]
        self.l = box_3d[3]
        self.w = box_3d[4]
        self.h = box_3d[5]
        self.ry = box_3d[6]
        self.score = score
        self.type = type
        self.roi_features = roi_features

batch_size = 32 # for PointNet
num_point = 512

def get_proposal_network(model_config, dataset, model_path, GPU_INDEX=0):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            rpn_model = RpnModel(model_config,
                         train_val_test='test',
                         dataset=dataset)
            rpn_pred = rpn_model.build()
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        saver.restore(sess, model_path)
        return rpn_pred, sess, rpn_model

def get_detection_network(model_path, GPU_INDEX=0):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            placeholders = frustum_pointnets_v2.placeholder_inputs(batch_size, num_point)
            pointclouds_pl, feature_vec_pl, cls_label_pl = placeholders[:3]
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = frustum_pointnets_v2.get_model(pointclouds_pl, cls_label_pl, feature_vec_pl,
                is_training_pl)
            end_points['pointclouds_pl'] = pointclouds_pl
            end_points['features_pl'] = feature_vec_pl
            end_points['is_training_pl'] = is_training_pl
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        saver.restore(sess, model_path)

        return end_points, sess

def get_dataset(dataset_config, data_split):
    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)

    dataset_config.data_split = data_split
    dataset_config.data_split_dir = 'training'
    if data_split == 'test':
        dataset_config.data_split_dir = 'testing'

    dataset_config.has_labels = False

    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)
    return dataset

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def compute_box_3d(obj):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    def roty(t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;

    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0];
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1];
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2];
    return corners_3d.T

def get_pointnet_input(sample, proposals_and_scores, roi_features, rpn_score_threshold=0.1):
    proposal_boxes_3d = proposals_and_scores[:, 0:7]
    proposal_scores = proposals_and_scores[:, 7]
    score_mask = proposal_scores > rpn_score_threshold
    # 3D box in the format [x, y, z, l, w, h, ry]
    proposal_boxes_3d = proposal_boxes_3d[score_mask]
    proposal_scores = proposal_scores[score_mask]
    roi_features = roi_features[score_mask]
    # point cloud of this frame
    pc = sample[constants.KEY_POINT_CLOUD].T
    frame_calib = sample[constants.KEY_STEREO_CALIB]
    #pc = calib_utils.lidar_to_cam_frame(pc.T, frame_calib)
    # point cloud in proposals
    point_clouds = []
    features = []
    for box_3d, feat in zip(proposal_boxes_3d, roi_features):
        obj = ProposalObject(box_3d, 1, None, None)
        corners = compute_box_3d(obj)
        #corners = calib_utils.lidar_to_cam_frame(corners, frame_calib)
        _, inds = extract_pc_in_box3d(pc, corners)
        if (np.any(inds) == False):
            # skip proposal with no points
            continue
        # TODO: rotate to center
        point_set = pc[inds]
        choice = np.random.choice(point_set.shape[0], num_point, replace=True)
        point_set = point_set[choice, :]
        point_clouds.append(point_set)
        features.append(feat)
    return point_clouds, features

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(rpn_model_path, detect_model_path, avod_config_path):
    model_config, _, eval_config, dataset_config = \
    config_builder.get_configs_from_pipeline_file(
        avod_config_path, is_training=False)

    # Setup the model
    model_name = model_config.model_name
    # Overwrite repeated field
    model_config = config_builder.proto_to_obj(model_config)
    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]

    dataset = get_dataset(dataset_config, 'val')
    # run avod proposal network
    rpn_endpoints, sess1, rpn_model = get_proposal_network(model_config, dataset, rpn_model_path)

    feed_dict1 = rpn_model.create_feed_dict(sample_index=0)
    kitti_samples = dataset.load_samples([0])
    rpn_predictions = sess1.run(rpn_endpoints, feed_dict=feed_dict1)
    top_anchors = rpn_predictions[RpnModel.PRED_TOP_ANCHORS]
    top_proposals = box_3d_encoder.anchors_to_box_3d(top_anchors)
    softmax_scores = rpn_predictions[RpnModel.PRED_TOP_OBJECTNESS_SOFTMAX]

    proposals_and_scores = np.column_stack((top_proposals,
                                            softmax_scores))
    top_img_roi = rpn_predictions[RpnModel.PRED_TOP_IMG_ROI]
    top_bev_roi = rpn_predictions[RpnModel.PRED_TOP_BEV_ROI]
    print(top_img_roi.shape)
    roi_num = len(top_img_roi)
    top_img_roi = np.reshape(top_img_roi, (roi_num, -1))
    top_bev_roi = np.reshape(top_bev_roi, (roi_num, -1))
    roi_features = np.column_stack((top_img_roi, top_bev_roi))
    '''
    pickle.dump({'proposals_and_scores': proposals_and_scores, 'roi_features': roi_features}, open("rpn_out", "wb"))
    data_dump = pickle.load(open("rpn_out", "rb"))
    proposals_and_scores = data_dump['proposals_and_scores']
    roi_features = data_dump['roi_features']
    kitti_samples = dataset.load_samples([0])
    '''
    # run frustum_pointnets_v2
    end_points, sess2 = get_detection_network(detect_model_path)
    point_clouds, feature_vec = get_pointnet_input(kitti_samples[0], proposals_and_scores, roi_features)
    feed_dict = {\
        end_points['pointclouds_pl']: point_clouds[:batch_size],
        end_points['features_pl']: feature_vec[:batch_size],
        end_points['is_training_pl']: False}

    batch_logits, batch_centers, \
    batch_heading_scores, batch_heading_residuals, \
    batch_size_scores, batch_size_residuals = \
        sess2.run([end_points['cls_logits'], end_points['center'],
            end_points['heading_scores'], end_points['heading_residuals'],
            end_points['size_scores'], end_points['size_residuals']],
            feed_dict=feed_dict)
    print(batch_logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rpn_model_path',
                    type=str,
                    dest='rpn_model_path',
                    required=True,
                    help='rpn_model_path')
    parser.add_argument('--detect_model_path',
                    type=str,
                    dest='detect_model_path',
                    required=True,
                    help='detect_model_path')
    parser.add_argument('--avod_config_path',
                    type=str,
                    dest='avod_config_path',
                    required=True,
                    help='avod_config_path')
    parser.add_argument('--device',
                    type=str,
                    dest='device',
                    default='0',
                    help='CUDA device id')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    inference(args.rpn_model_path, args.detect_model_path, args.avod_config_path)


if __name__ == '__main__':
    main()
