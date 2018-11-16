''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import cPickle as pickle
from threading import Thread
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from model_util import NUM_SEG_CLASSES, NUM_OBJ_CLASSES, g_type2onehotclass, type_whitelist
from avod_dataset import AvodDataset, Sample
import provider
from train_util import get_batch
from kitti_object import *
import kitti_util as utils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
# parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)
# NUM_CLASSES = 2
NUM_CHANNEL = 4

TEST_DATASET = AvodDataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', BATCH_SIZE, 'val',
             save_dir='/data/ssd/public/jlliu/frustum-pointnets/train/avod_dataset_0.65/val',
             augmentX=1, random_shift=False, rotate_to_center=True, random_flip=False)
val_loading_thread = Thread(target=TEST_DATASET.load_buffer_repeatedly, args=(0.5, True))
val_loading_thread.start()

kitti_dataset = kitti_object('/data/ssd/public/jlliu/Kitti/object')

def get_session_and_ops(batch_size, num_point):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, features_pl, cls_labels_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, cls_labels_pl, features_pl,
                is_training_pl)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'features_pl': features_pl,
               'cls_label_pl': cls_labels_pl,
               'centers_pl': centers_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'cls_logits': end_points['cls_logits'],
               'center': end_points['center'],
               'end_points': end_points,
               }
        return sess, ops

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(sess, ops, pc, feature_vec, cls_label):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0] == BATCH_SIZE

    ep = ops['end_points']

    feed_dict = {ops['pointclouds_pl']: pc,
                 ops['features_pl']: feature_vec,
                 ops['cls_label_pl']: cls_label,
                 ops['is_training_pl']: False}
    cls_logits, logits, centers, \
    heading_logits, heading_residuals, \
    size_logits, size_residuals = \
        sess.run([ep['cls_logits'], ep['mask_logits'], ep['center'],
            ep['heading_scores'], ep['heading_residuals'],
            ep['size_scores'], ep['size_residuals']],
            feed_dict=feed_dict)

    # Compute scores
    batch_seg_prob = softmax(logits)[:,:,1] # BxN
    batch_seg_mask = np.argmax(logits, 2) # BxN
    mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
    mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
    heading_prob = np.max(softmax(heading_logits),1) # B
    size_prob = np.max(softmax(size_logits),1) # B,
    scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
    # Finished computing scores

    type_cls = np.argmax(cls_logits, 1)
    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    heading_res = np.array([heading_residuals[i,heading_cls[i]] \
        for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i,size_cls[i],:] \
        for i in range(pc.shape[0])])

    return type_cls, centers, heading_cls, heading_res, \
        size_cls, size_res, scores

class DetectObject(object):
    def __init__(h,w,l,tx,ty,tz,ry):
        self.t = [tx,ty,tz]
        self.ry = ry
        self.h = h
        self.w = w
        self.l = l

def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        # box2d = box2d_list[i]
        # output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        # cal 2d box from 3d box
        calib = kitti_dataset.get_calibration(idx)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(DetectObject(h,w,l,tx,ty,tz,ry), calib.P)
        x1 = np.amin(box3d_pts_2d[0])
        y1 = np.amin(box3d_pts_2d[1])
        x2 = np.amax(box3d_pts_2d[0])
        y2 = np.amax(box3d_pts_2d[1])
        output_str += "%f %f %f %f " % (x1, y1, x2, y2)
        score = score_list[i]
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

def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

def test(output_filename, result_dir=None):
    ''' Test frustum pointents with 2D boxes from a RGB detector.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    cls_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    frame_id_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    sess, ops = get_session_and_ops(batch_size=BATCH_SIZE, num_point=NUM_POINT)
    # for batch_idx in range(num_batches):
    batch_idx = 0
    # TODO: return frame_id_list in get_next_batch
    while(True):
        batch_data, batch_cls_label, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_feature_vec, batch_frame_ids, is_last_batch = TEST_DATASET.get_next_batch()

        if is_last_batch and len(batch_data) != BATCH_SIZE:
            # discard last batch with fewer data
            break
        print('batch idx: %d' % (batch_idx))
        batch_idx += 1

        # Run one batch inference
    	batch_cls, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores = \
            inference(sess, ops, batch_data, batch_feature_vec, batch_cls_label)

        tp = np.sum(np.logical_and(batch_cls == batch_cls_label, batch_cls_label < g_type2onehotclass['NonObject']))
        fp = np.sum(np.logical_and(batch_cls != batch_cls_label, batch_cls_label == g_type2onehotclass['NonObject']))
        fn = np.sum(np.logical_and(batch_cls != batch_cls_label, batch_cls_label < g_type2onehotclass['NonObject']))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print('average recall: {}, precision: {}'.format(float(total_tp)/(total_tp+total_fn), float(total_tp)/(total_tp+total_fp)))

        for i in range(BATCH_SIZE):
            ps_list.append(batch_data[i,...])
            cls_list.append(batch_cls[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])
        frame_id_list += map(lambda fid: int(fid), batch_frame_ids)

    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(cls_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)
            pickle.dump(frame_id_list, fp)

    type_list = map(lambda i: type_whitelist[i], cls_list)
    #TODO: box2d_list, project to image
    box2d_list = np.zeros((len(ps_list), 4))
    # Write detection results for KITTI evaluation
    write_detection_results(result_dir, frame_id_list,
        type_list, box2d_list,
        center_list, heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list)


if __name__=='__main__':
    test(FLAGS.output+'.pickle', FLAGS.output)
