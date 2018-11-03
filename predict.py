import tensorflow as tf
import os
import sys
import argparse
import numpy as np
import pickle
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from avod.core import box_3d_encoder
from avod.core import evaluator_utils
from avod.core import summary_utils
from avod.core import trainer_utils
from avod.core import constants
from avod.core.models.rpn_model import RpnModel
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from wavedata.tools.core import calib_utils
from wavedata.tools.visualization import vis_utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import frustum_pointnets_v2
sys.path.append(os.path.join(ROOT_DIR, 'avod_prop'))
from kitti_object_avod import non_max_suppression
# sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
sys.path.append(os.path.join(ROOT_DIR, 'train'))
import provider

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
NUM_CLASSES = 4
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

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

    dataset_config.has_labels = data_split in ['train', 'val']

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

def nms_on_bev(boxes_3d, iou_threshold=0.1):
    bev_boxes = list(map(lambda p: [p[0] - p[3]/2, p[2] - p[5]/2, p[0] + p[3]/2, p[2] + p[5]/2, p[7]], boxes_3d))
    bev_boxes = np.array(bev_boxes)
    print('fianl output before nms: {0}'.format(len(bev_boxes)))
    nms_idxs = non_max_suppression(bev_boxes, iou_threshold)
    print('fianl output after nms: {0}'.format(len(nms_idxs)))
    # output = [output[i] for i in nms_idxs]
    return nms_idxs

def get_pointnet_input(sample, proposals_and_scores, roi_features, rpn_score_threshold=0.1):
    proposal_boxes_3d = proposals_and_scores[:, 0:7]
    proposal_scores = proposals_and_scores[:, 7]
    score_mask = proposal_scores > rpn_score_threshold
    # 3D box in the format [x, y, z, l, w, h, ry]
    proposal_boxes_3d = proposal_boxes_3d[score_mask]
    proposal_scores = proposal_scores[score_mask]
    roi_features = roi_features[score_mask]

    proposal_objs = list(map(lambda pair: ProposalObject(pair[0], pair[1], None, None), zip(proposal_boxes_3d, proposal_scores)))
    propsasl_corners = list(map(lambda obj: compute_box_3d(obj), proposal_objs))

    # point cloud of this frame
    pc = sample[constants.KEY_POINT_CLOUD].T
    frame_calib = sample[constants.KEY_STEREO_CALIB]
    #pc = calib_utils.lidar_to_cam_frame(pc.T, frame_calib)
    # point cloud in proposals
    point_clouds = []
    features = []
    rot_angle_list = []
    for obj, corners, feat in zip(proposal_objs, propsasl_corners, roi_features):
        #corners = calib_utils.lidar_to_cam_frame(corners, frame_calib)
        _, inds = extract_pc_in_box3d(pc, corners)
        if (np.any(inds) == False):
            # skip proposal with no points
            continue
        # under rect coorination, x->right, y->down, z->front
        center_rect = (np.min(corners, axis=0) + np.max(corners, axis=0)) / 2
        # FIXME: here induces a 90 degrees offset when visualize, should be fix together with prepare_data.py
        frustum_angle = -1 * np.arctan2(center_rect[2], center_rect[0])
        # rotate to center
        pc_rot = rotate_pc_along_y(pc[inds], np.pi/2.0 + frustum_angle)
        rot_angle_list.append(frustum_angle)
        point_set = pc_rot
        choice = np.random.choice(point_set.shape[0], num_point, replace=True)
        point_set = point_set[choice, :]
        point_clouds.append(point_set)
        features.append(feat)
        # import mayavi.mlab as mlab
        # from viz_util import draw_lidar, draw_gt_boxes3d
        # fig = draw_lidar(pc)
        # fig = draw_gt_boxes3d([corners], fig, draw_text=False, color=(1, 1, 1))
        # mlab.plot3d([0, center_rect[0]], [0, center_rect[1]], [0, center_rect[2]], color=(1,1,1), tube_radius=None, figure=fig)
        # input()
    return point_clouds, features, rot_angle_list

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def detect_batch(sess, end_points, point_clouds, feature_vec, rot_angle_list):
    sample_num = len(point_clouds)
    logits = np.zeros((sample_num, NUM_CLASSES))
    centers = np.zeros((sample_num, 3))
    heading_logits = np.zeros((sample_num, NUM_HEADING_BIN))
    heading_residuals = np.zeros((sample_num, NUM_HEADING_BIN))
    size_logits = np.zeros((sample_num, NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((sample_num, NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((sample_num,)) # 3D box score
    for i in range(math.floor(sample_num/batch_size)):
        begin = i * batch_size
        end = min((i + 1) * batch_size, sample_num)

        feed_dict = {\
            end_points['pointclouds_pl']: point_clouds[begin:end],
            end_points['features_pl']: feature_vec[begin:end],
            end_points['is_training_pl']: False}

        batch_logits, batch_seg_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([end_points['cls_logits'], end_points['mask_logits'], end_points['center'],
                end_points['heading_scores'], end_points['heading_residuals'],
                end_points['size_scores'], end_points['size_residuals']],
                feed_dict=feed_dict)

        logits[begin:end,...] = batch_logits
        centers[begin:end,...] = batch_centers
        heading_logits[begin:end,...] = batch_heading_scores
        heading_residuals[begin:end,...] = batch_heading_residuals
        size_logits[begin:end,...] = batch_size_scores
        size_residuals[begin:end,...] = batch_size_residuals

        # Compute scores
        batch_cls_prob = np.max(softmax(batch_logits),1) # B,
        batch_seg_prob = softmax(batch_seg_logits)[:,:,1] # BxN
        batch_seg_mask = np.argmax(batch_seg_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        heading_prob = np.max(softmax(batch_heading_scores),1) # B
        size_prob = np.max(softmax(batch_size_scores),1) # B,
        batch_scores = (batch_cls_prob + mask_mean_prob + heading_prob + size_prob) / 4
        # scores[begin:end] = batch_cls_prob
        scores[begin:end] = batch_scores
        # Finished computing scores

    type_cls = np.argmax(logits, 1)
    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    heading_res = np.array([heading_residuals[i,heading_cls[i]] \
        for i in range(sample_num)])
    size_res = np.vstack([size_residuals[i,size_cls[i],:] \
        for i in range(sample_num)])

    output = []
    for i in range(sample_num):
        if type_cls[i] == 3:
            # background
            continue
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(centers[i],
            heading_cls[i], heading_res[i],
            size_cls[i], size_res[i], rot_angle_list[i])
        obj_type = type_cls[i]
        confidence = scores[i]
        # print(tx,ty,tz,l,w,h,ry,confidence,obj_type)
        output.append([tx,ty,tz,l,w,h,ry,confidence,obj_type])
    # 2d nms on bev
    nms_idxs = nms_on_bev(output, 0.1)
    output = [output[i] for i in nms_idxs]
    return output

def visualize(dataset, sample, prediction):
    fig_size = (10, 6.1)
    sample_name = sample[constants.KEY_SAMPLE_NAME]
    pred_fig, pred_2d_axes, pred_3d_axes = \
        vis_utils.visualization(dataset.rgb_image_dir,
                                int(sample_name),
                                display=False,
                                fig_size=fig_size)
    type_names = ['Car', 'Pedestrian', 'Cyclist', 'Background']
    pc = sample[constants.KEY_POINT_CLOUD].T
    all_corners = []
    for pred in prediction:
        box = np.array(pred[0:7])
        obj = box_3d_encoder.box_3d_to_object_label(box, obj_type=type_names[pred[8]])
        obj.score = pred[7]
        # FIXME: this offset should be fixed in get_pointnet_input
        obj.t = rotate_pc_along_y(np.expand_dims(np.asarray(obj.t), 0), -np.pi/2)[0]
        vis_utils.draw_box_3d(pred_3d_axes, obj, sample[constants.KEY_STEREO_CALIB_P2],
                          show_orientation=False,
                          color_table=['r', 'y', 'r', 'w'],
                          line_width=2,
                          double_line=False)
        corners = compute_box_3d(obj)
        all_corners.append(corners)
    # 3d visualization
    '''
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_gt_boxes3d
    fig = draw_lidar(pc)
    fig = draw_gt_boxes3d(all_corners, fig, draw_text=False, color=(1, 1, 1))
    # mlab.plot3d([0, center_rect[0]], [0, center_rect[1]], [0, center_rect[2]], color=(1,1,1), tube_radius=None, figure=fig)
    input()
    '''

    # 2d visualization
    filename = 'final_out_viz/%s.png' % sample_name
    plt.savefig(filename)
    plt.close(pred_fig)
    # plt.show()
    # input()

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
    end_points, sess2 = get_detection_network(detect_model_path)

    for idx in range(1):
        feed_dict1 = rpn_model.create_feed_dict()
        kitti_samples = dataset.load_samples([0])
        sample = kitti_samples[0]
        print(sample[constants.KEY_SAMPLE_NAME])
        rpn_predictions = sess1.run(rpn_endpoints, feed_dict=feed_dict1)
        top_anchors = rpn_predictions[RpnModel.PRED_TOP_ANCHORS]
        top_proposals = box_3d_encoder.anchors_to_box_3d(top_anchors)
        softmax_scores = rpn_predictions[RpnModel.PRED_TOP_OBJECTNESS_SOFTMAX]

        proposals_and_scores = np.column_stack((top_proposals,
                                                softmax_scores))
        top_img_roi = rpn_predictions[RpnModel.PRED_TOP_IMG_ROI]
        top_bev_roi = rpn_predictions[RpnModel.PRED_TOP_BEV_ROI]
        roi_num = len(top_img_roi)
        top_img_roi = np.reshape(top_img_roi, (roi_num, -1))
        top_bev_roi = np.reshape(top_bev_roi, (roi_num, -1))
        roi_features = np.column_stack((top_img_roi, top_bev_roi))

        '''
        #pickle.dump({'proposals_and_scores': proposals_and_scores, 'roi_features': roi_features}, open("rpn_out", "wb"))
        data_dump = pickle.load(open("rpn_out", "rb"))
        proposals_and_scores = data_dump['proposals_and_scores']
        roi_features = data_dump['roi_features']
        kitti_samples = dataset.load_samples([0])
        '''
        # run frustum_pointnets_v2
        point_clouds, feature_vec, rot_angle_list = get_pointnet_input(sample, proposals_and_scores, roi_features)
        prediction = detect_batch(sess2, end_points, point_clouds, feature_vec, rot_angle_list)
        pickle.dump(prediction, open('final_out/%s' % sample[constants.KEY_SAMPLE_NAME], 'wb'))
        visualize(dataset, sample, prediction)

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


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--avod_config_path',
                    type=str,
                    dest='avod_config_path',
                    required=True,
                    help='avod_config_path')
    args = parser.parse_args()
    _, _, _, dataset_config = \
    config_builder.get_configs_from_pipeline_file(
        args.avod_config_path, is_training=False)
    dataset = get_dataset(dataset_config, 'val')
    kitti_samples = dataset.load_samples([0])
    sample = kitti_samples[0]
    rpn_out = pickle.load(open("rpn_out", "rb"))
    proposals_and_scores = rpn_out['proposals_and_scores']
    ### start visualize rpn output
    proposal_scores = proposals_and_scores[:, 7]
    score_mask = proposal_scores > 0.1
    # 3D box in the format [x, y, z, l, w, h, ry]
    proposals_and_scores = proposals_and_scores[score_mask]
    nms_idxs = nms_on_bev(proposals_and_scores, 0.1)
    proposals_and_scores = proposals_and_scores[nms_idxs]

    proposal_objs = list(map(lambda p: ProposalObject(p[:7], p[7], None, None), proposals_and_scores))
    propsasl_corners = list(map(lambda obj: compute_box_3d(obj), proposal_objs))
    pc = sample[constants.KEY_POINT_CLOUD].T
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_gt_boxes3d
    # fig = draw_lidar(pc)
    # fig = draw_gt_boxes3d(propsasl_corners[:1], fig, draw_text=False, color=(1, 1, 1))
    # input()
    ### end visualize rpn output
    roi_features = rpn_out['roi_features']
    point_clouds, feature_vec, rot_angle_list = get_pointnet_input(kitti_samples[0], proposals_and_scores[:10], roi_features[:10])
    fig = draw_lidar(np.concatenate(point_clouds))
    fig = draw_gt_boxes3d(propsasl_corners[:10], fig, draw_text=False, color=(1, 1, 1))
    input()
    # prediction = pickle.load(open("001101", "rb"))
    # visualize(dataset, sample, prediction)

if __name__ == '__main__':
    main()
