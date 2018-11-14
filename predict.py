import tensorflow as tf
import os
import sys
import argparse
import numpy as np
import pickle
import math
import time
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

from avod.core import box_3d_encoder
from avod.core import evaluator_utils
from avod.core import summary_utils
from avod.core import trainer_utils
from avod.core import constants
from avod.core.models.rpn_model import RpnModel
from avod.core.models.avod_model import AvodModel
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_projector
from wavedata.tools.core import calib_utils
from wavedata.tools.visualization import vis_utils
from shapely.geometry import Polygon

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util import g_type2onehotclass, type_whitelist
import frustum_pointnets_v2
sys.path.append(os.path.join(ROOT_DIR, 'avod_prop'))
from kitti_object_avod import non_max_suppression
# sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
sys.path.append(os.path.join(ROOT_DIR, 'train'))
import provider

BOX_COLOUR_SCHEME = {
    'Car': '#00FF00',           # Green
    'Pedestrian': '#00FFFF',    # Teal
    'Cyclist': '#FFFF00'        # Yellow
}

type_idx_whitelist = [g_type2onehotclass[t] for t in type_whitelist if t != 'NonObject']

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
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJ_CLASSES

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
            rpn_model = AvodModel(model_config,
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
    scores = np.asarray(boxes_3d)[:,7]
    boxes_3d = np.asarray(boxes_3d)[:,0:7]
    corners = list(map(lambda box: compute_box_3d(box_3d_encoder.box_3d_to_object_label(box)), boxes_3d))
    # TODO: use Polygon to do nms
    bev_boxes = list(map(lambda p: [np.amin(p[0],axis=0)[0], np.amin(p[0], axis=0)[2], np.amax(p[0], axis=0)[0], np.amax(p[0], axis=0)[2], p[1]], zip(corners, scores)))
    bev_boxes = np.array(bev_boxes)
    print('final output before nms: {0}'.format(len(bev_boxes)))
    nms_idxs = non_max_suppression(bev_boxes, iou_threshold)
    print('final output after nms: {0}'.format(len(nms_idxs)))
    return nms_idxs

def find_match_label(prop_corners, labels_corners):
    '''
    Find label with largest IOU. Label boxes can be rotated in xy plane
    '''
    # labels = MultiPolygon(labels_corners)
    labels = map(lambda corners: Polygon(corners), labels_corners)
    target = Polygon(prop_corners)
    largest_iou = 0
    largest_idx = -1
    for i, label in enumerate(labels):
        area1 = label.area
        area2 = target.area
        intersection = target.intersection(label).area
        iou = intersection / (area1 + area2 - intersection)
        # print(area1, area2, intersection)
        # print(iou)
        if iou > largest_iou:
            largest_iou = iou
            largest_idx = i
    return largest_idx, largest_iou

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

    # get groundtruth cls label
    label_mask = sample[constants.KEY_LABEL_CLASSES] < g_type2onehotclass['NonObject'] + 1
    gt_cls = sample[constants.KEY_LABEL_CLASSES][label_mask]
    gt_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D][label_mask]
    gt_boxes_bev = gt_boxes_3d[:4, [0,2]]

    # point cloud of this frame
    pc = sample[constants.KEY_POINT_CLOUD].T
    frame_calib = sample[constants.KEY_STEREO_CALIB]
    #pc = calib_utils.lidar_to_cam_frame(pc.T, frame_calib)
    # point cloud in proposals
    point_clouds = []
    features = []
    rot_angle_list = []
    prop_cls_labels = []
    for obj, corners, feat in zip(proposal_objs, propsasl_corners, roi_features):
        _, inds = extract_pc_in_box3d(pc, corners)
        if (np.any(inds) == False):
            # skip proposal with no points
            continue
        # get groundtruth cls label for each proposal
        corners_bev = corners[:4, [0,2]]
        label_idx, iou = find_match_label(corners_bev, gt_boxes_bev)
        if iou >= 0.65:
            prop_cls_labels.append(gt_cls[label_idx] - 1)
        else:
            prop_cls_labels.append(g_type2onehotclass['NonObject'])
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
    return point_clouds, features, rot_angle_list, prop_cls_labels

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def detect_batch(sess, end_points, point_clouds, feature_vec, rot_angle_list, prop_cls_labels):
    sample_num = len(point_clouds)
    logits = np.zeros((sample_num, NUM_OBJ_CLASSES))
    centers = np.zeros((sample_num, 3))
    heading_logits = np.zeros((sample_num, NUM_HEADING_BIN))
    heading_residuals = np.zeros((sample_num, NUM_HEADING_BIN))
    size_logits = np.zeros((sample_num, NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((sample_num, NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((sample_num,)) # 3D box score
    points_num = np.zeros((sample_num,))
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
        points_num[begin:end,...] = np.sum(np.equal(np.argmax(batch_seg_logits, 2), 1), axis=1)

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
        # if type_cls[i] == g_type2onehotclass['NonObject'] or scores[i] < 0.5 or points_num[i] == 0:
        # use ground as cls output
        type_cls[i] = prop_cls_labels[i]
        if type_cls[i] == g_type2onehotclass['NonObject']:
            continue
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(centers[i],
            heading_cls[i], heading_res[i],
            size_cls[i], size_res[i], rot_angle_list[i])
        # FIXME: this offset should be fixed in get_pointnet_input
        tx, ty, tz = rotate_pc_along_y(np.expand_dims(np.asarray([tx,ty,tz]), 0), -np.pi/2)[0]
        ry += np.pi/2

        obj_type = type_cls[i]
        confidence = scores[i]
        output.append([tx,ty,tz,l,w,h,ry,confidence,obj_type])
    if len(output) == 0:
        return output
    # 2d nms on bev
    nms_idxs = nms_on_bev(output, 0.01)
    output = [output[i] for i in nms_idxs]

    return output

def draw_boxes(prediction, sample, plot_axes):
    all_corners = []
    for pred in prediction:
        box = np.array(pred[0:7])
        cls_idx = int(pred[8])
        obj = box_3d_encoder.box_3d_to_object_label(box, obj_type=type_whitelist[cls_idx])
        obj.score = pred[7]

        vis_utils.draw_box_3d(plot_axes, obj, sample[constants.KEY_STEREO_CALIB_P2],
                          show_orientation=False,
                          color_table=['r', 'y', 'r', 'w'],
                          line_width=2,
                          double_line=False)
        corners = compute_box_3d(obj)
        all_corners.append(corners)

        # draw text info
        projected = calib_utils.project_to_image(corners.T, sample[constants.KEY_STEREO_CALIB_P2])
        x1 = np.amin(projected[0])
        y1 = np.amin(projected[1])
        x2 = np.amax(projected[0])
        y2 = np.amax(projected[1])
        text_x = (x1 + x2) / 2
        text_y = y1
        text = "{}\n{:.2f}".format(obj.type, obj.score)
        plot_axes.text(text_x, text_y - 4,
            text,
            verticalalignment='bottom',
            horizontalalignment='center',
            color=BOX_COLOUR_SCHEME[obj.type],
            fontsize=10,
            fontweight='bold',
            path_effects=[
                patheffects.withStroke(linewidth=2,
                                       foreground='black')])
    return all_corners

def visualize(dataset, sample, prediction):
    fig_size = (10, 6.1)
    sample_name = sample[constants.KEY_SAMPLE_NAME]
    pred_fig, pred_2d_axes, pred_3d_axes = \
        vis_utils.visualization(dataset.rgb_image_dir,
                                int(sample_name),
                                display=False,
                                fig_size=fig_size)
    pc = sample[constants.KEY_POINT_CLOUD].T
    # draw prediction on second image
    pred_corners = draw_boxes(prediction, sample, pred_3d_axes)

    # draw groundtruth on first image
    obj_mask = np.zeros((len(sample[constants.KEY_LABEL_CLASSES]),), dtype=bool)
    for i in range(obj_mask.shape[0]):
        if sample[constants.KEY_LABEL_CLASSES][i] - 1 in type_idx_whitelist:
            obj_mask[i] = True
    label_boxes = sample[constants.KEY_LABEL_BOXES_3D]
    label_classes = np.expand_dims(sample[constants.KEY_LABEL_CLASSES], axis=1).astype(int) - 1
    label_scores = np.ones((len(label_classes), 1))
    labels = np.concatenate((label_boxes, label_scores, label_classes), axis=1)
    pred_corners = draw_boxes(labels[obj_mask], sample, pred_2d_axes)

    # 3d visualization
    # import mayavi.mlab as mlab
    # from viz_util import draw_lidar, draw_gt_boxes3d
    # fig = draw_lidar(pc)
    # fig = draw_gt_boxes3d(pred_corners, fig, draw_text=False, color=(1, 1, 1))
    # # mlab.plot3d([0, center_rect[0]], [0, center_rect[1]], [0, center_rect[2]], color=(1,1,1), tube_radius=None, figure=fig)
    # input()

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

    all_prediction = []
    all_id_list = None
    all_2d_boxes = []
    for idx in range(3769):
        feed_dict1 = rpn_model.create_feed_dict()
        kitti_samples = dataset.load_samples([idx])
        sample = kitti_samples[0]
        '''
        if sample[constants.KEY_SAMPLE_NAME] < '001100':
            continue
        if sample[constants.KEY_SAMPLE_NAME] > '001200':
            break
        '''
        start_time = time.time()
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
        # save proposal
        if os.path.exists(os.path.join('/data/ssd/public/jlliu/Kitti/object/training/proposal', '%s.txt'%(sample[constants.KEY_SAMPLE_NAME]))):
            continue
        np.savetxt(os.path.join('./proposals_and_scores/', '%s.txt'%sample[constants.KEY_SAMPLE_NAME]), proposals_and_scores, fmt='%.3f')
        np.savetxt(os.path.join('./roi_features/', '%s_roi.txt'%sample[constants.KEY_SAMPLE_NAME]), roi_features, fmt='%.5f')
        print('save ' + sample[constants.KEY_SAMPLE_NAME])
        '''
        # run frustum_pointnets_v2
        point_clouds, feature_vec, rot_angle_list, prop_cls_labels = get_pointnet_input(sample, proposals_and_scores, roi_features)
        try:
            prediction = detect_batch(sess2, end_points, point_clouds, feature_vec, rot_angle_list, prop_cls_labels)
        except:
            traceback.print_exc()
            continue

        elapsed_time = time.time() - start_time
        print(sample[constants.KEY_SAMPLE_NAME], elapsed_time)
        # concat all predictions for kitti eval
        id_list = np.ones((len(prediction),)) * int(sample[constants.KEY_SAMPLE_NAME])
        if all_id_list is None:
            all_id_list = id_list
        else:
            all_id_list = np.concatenate((all_id_list, id_list), axis=0)
        for pred in prediction:
            obj = box_3d_encoder.box_3d_to_object_label(np.array(pred[0:7]), obj_type=type_whitelist[pred[8]])
            corners = compute_box_3d(obj)
            projected = calib_utils.project_to_image(corners.T, sample[constants.KEY_STEREO_CALIB_P2])
            x1 = np.amin(projected[0])
            y1 = np.amin(projected[1])
            x2 = np.amax(projected[0])
            y2 = np.amax(projected[1])
            all_2d_boxes.append([x1, y1, x2, y2])
        all_prediction += prediction
        # save result
        pickle.dump({'proposals_and_scores': proposals_and_scores, 'roi_features': roi_features}, open("rpn_out/%s"%sample[constants.KEY_SAMPLE_NAME], "wb"))
        pickle.dump(prediction, open('final_out/%s' % sample[constants.KEY_SAMPLE_NAME], 'wb'))
        visualize(dataset, sample, prediction)
    # for kitti eval
    write_detection_results('./detection_results', all_prediction, all_id_list, all_2d_boxes)

def write_detection_results(result_dir, predictions, id_list, boxes_2d):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(predictions)):
        idx = id_list[i]
        tx,ty,tz,l,w,h,ry,score,obj_type = predictions[i]
        output_str = type_whitelist[obj_type] + " -1 -1 -10 "
        box2d = boxes_2d[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close()

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

def visualize_rpn_out(sample, proposals_and_scores, rpn_score_threshold=0.1):
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_gt_boxes3d
    nms_idxs = nms_on_bev(proposals_and_scores, 0.5)
    proposals_and_scores = proposals_and_scores[nms_idxs]

    proposal_boxes_3d = proposals_and_scores[:, 0:7]
    proposal_scores = proposals_and_scores[:, 7]
    score_mask = proposal_scores > rpn_score_threshold
    # 3D box in the format [x, y, z, l, w, h, ry]
    proposal_boxes_3d = proposal_boxes_3d[score_mask]
    proposal_scores = proposal_scores[score_mask]

    proposal_objs = list(map(lambda pair: ProposalObject(pair[0], pair[1], None, None), zip(proposal_boxes_3d, proposal_scores)))
    propsasl_corners = list(map(lambda obj: compute_box_3d(obj), proposal_objs))
    pc = sample[constants.KEY_POINT_CLOUD].T
    fig = draw_lidar(pc)
    fig = draw_gt_boxes3d(propsasl_corners, fig, draw_text=False, color=(1, 1, 1))
    input()

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--avod_config_path',
                    type=str,
                    dest='avod_config_path',
                    required=True,
                    help='avod_config_path')
    parser.add_argument('--sample_idx',
                    type=str,
                    dest='sample_idx',
                    required=True,
                    help='sample id')
    args = parser.parse_args()
    _, _, _, dataset_config = \
    config_builder.get_configs_from_pipeline_file(
        args.avod_config_path, is_training=False)
    dataset = get_dataset(dataset_config, 'val')

    idx = np.argwhere(dataset.sample_names==args.sample_idx).squeeze()
    # print(idx)
    kitti_samples = dataset.load_samples([idx])
    sample = kitti_samples[0]
    # rpn_out = pickle.load(open("rpn_out/%s" % sample[constants.KEY_SAMPLE_NAME], "rb"))
    # visualize_rpn_out(sample, rpn_out['proposals_and_scores'])
    prediction = pickle.load(open("%s"%sample[constants.KEY_SAMPLE_NAME], "rb"))
    visualize(dataset, sample, prediction)

if __name__ == '__main__':
    main()
