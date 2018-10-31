import tensorflow as tf

from avod.core import box_3d_encoder
from avod.core import evaluator_utils
from avod.core import summary_utils
from avod.core import trainer_utils
from avod.core import constants
from avod.core.models.rpn_model import RpnModel
from avod.builders.dataset_builder import DatasetBuilder
from wavedata.tools.core import calib_utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import frustum_pointnets_v2
sys.path.append(os.path.join(ROOT_DIR, 'avod_prop'))

batch_size = 32 # for PointNet

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
            pointclouds_pl, feature_vec_pl = placeholders[:2]
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = frustum_pointnets_v2.get_model(pointclouds_pl, feature_vec_pl,
                is_training_pl)
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

def get_pointnet_input(sample, proposals_and_scores, roi_features, rpn_score_threshold=0.1):
    proposal_boxes_3d = proposals_and_scores[:, 0:7]
    proposal_scores = proposals_and_scores[:, 7]
    score_mask = proposal_scores > rpn_score_threshold
    # 3D box in the format [x, y, z, l, w, h, ry]
    proposal_boxes_3d = proposal_boxes_3d[score_mask]
    proposal_scores = proposal_scores[score_mask]
    roi_features = roi_features[score_mask]
    # point cloud of this frame
    pc = sample[constants.KEY_POINT_CLOUD]
    frame_calib = sample[constants.KEY_STEREO_CALIB_P2]
    pc = calib_utils.lidar_to_cam_frame(pc, frame_calib)
    # point cloud in proposals
    point_clouds = []
    features = []
    for box_3d, feat in zip(proposal_boxes_3d, roi_features):
        obj = Object()
        obj.t = box_3d[0:3]
        obj.l = box_3d[3]
        obj.w = box_3d[4]
        obj.h = box_3d[5]
        obj.ry = box_3d[6]
        corners = compute_box_3d(box_3d)
        _, inds = extract_pc_in_box3d(pc, corners)
        if (len(inds < 0)):
            # skip proposal with no points
            continue
        point_clouds.append(pc[inds])
        features.append(feat)
    return point_clouds, features


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
    softmax_scores = predictions[RpnModel.PRED_TOP_OBJECTNESS_SOFTMAX]

    proposals_and_scores = np.column_stack((top_proposals,
                                            softmax_scores))
    print(proposals_and_scores)
    # run frustum_pointnets_v2
    detect_endpoints, sess2 = get_detection_network(detect_model_path)
    point_clouds, feature_vec = get_pointnet_input(kitti_samples[0], proposals_and_scores, roi_features)
    feed_dict = {\
        ops['pointclouds_pl']: point_clouds[:batch_size],
        ops['features_pl']: feature_vec[:batch_size],
        ops['is_training_pl']: False}

    batch_logits, batch_centers, \
    batch_heading_scores, batch_heading_residuals, \
    batch_size_scores, batch_size_residuals = \
        sess.run(end_points['cls_logits'], end_points['center'],
            end_points['heading_scores'], end_points['heading_residuals'],
            end_points['size_scores'], end_points['size_residuals']],
            feed_dict=feed_dict)
    print(batch_logits)
    # Compute scores
    heading_prob = np.max(softmax(batch_heading_scores),1) # B
    size_prob = np.max(softmax(batch_size_scores),1) # B,
    batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)


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
