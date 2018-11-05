''' Training Frustum PointNets.

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
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
from train_util import get_batch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--pickle', default='', help='pickle name')
FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4 # point feature channel
NUM_SEG_CLASSES = 2 # segmentation has two classes
NUM_OBJ_CLASSES = 4 # classification has 4 classes

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Load Frustum Datasets. Use default data paths.
TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train',
    overwritten_data_path='/data/ssd/public/jlliu/frustum-pointnets/avod_prop/frustum_carpedcyc_train%s.pickle'%FLAGS.pickle,
    rotate_to_center=True, random_flip=True, random_shift=True, extra_feature=True)
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
    overwritten_data_path='/data/ssd/public/jlliu/frustum-pointnets/avod_prop/frustum_carpedcyc_val%s.pickle'%FLAGS.pickle,
    rotate_to_center=True, extra_feature=True)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    ''' Main function for training and simple evaluation. '''
    best_val_loss = float('inf')
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, features_pl, cls_labels_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and losses
            end_points = MODEL.get_model(pointclouds_pl, cls_labels_pl, features_pl,
                is_training_pl, bn_decay=bn_decay)
            loss, loss_endpoints = MODEL.get_loss(cls_labels_pl, labels_pl, centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            tf.summary.scalar('loss', loss)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

            # Write summaries of bounding box IoU and segmentation accuracies
            iou2ds, iou3ds = tf.py_func(provider.compute_box3d_iou, [\
                end_points['center'], \
                end_points['heading_scores'], end_points['heading_residuals'], \
                end_points['size_scores'], end_points['size_residuals'], \
                centers_pl, \
                heading_class_label_pl, heading_residual_label_pl, \
                size_class_label_pl, size_residual_label_pl], \
                [tf.float32, tf.float32])
            end_points['iou2ds'] = iou2ds
            end_points['iou3ds'] = iou3ds
            tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
            tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))

            correct = tf.equal(tf.argmax(end_points['mask_logits'], 2),
                tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
                float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('segmentation accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                    momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # Note: when training, the moving_mean and moving_variance need to be updated.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops = {'pointclouds_pl': pointclouds_pl,
               'features_pl': features_pl,
               'cls_label_pl': cls_labels_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'cls_logits': end_points['cls_logits'],
               'centers_pred': end_points['center'],
               'loss': loss,
               'loss_endpoints': loss_endpoints,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            val_loss = eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: {0}, val_loss: {1}".format(save_path, val_loss))

def train_one_epoch(sess, ops, train_writer):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    '''
    is_training = True
    log_string(str(datetime.now()))

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE

    # To collect statistics
    total_cls_correct = 0
    total_cls_seen = 0
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_obj_sample = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    # Training with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_cls_label, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_feature_vec = \
            get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['features_pl']: batch_feature_vec,
                     ops['cls_label_pl']: batch_cls_label,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training,}

        summary, step, _, loss_val, cls_logits_val, logits_val, centers_pred_val, \
        iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                ops['cls_logits'], ops['logits'], ops['centers_pred'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        # classification acc
        cls_preds_val = np.argmax(cls_logits_val, 1)
        cls_correct = np.sum(cls_preds_val == batch_cls_label)
        total_cls_correct += cls_correct
        total_cls_seen += BATCH_SIZE
        # only calculate seg acc and regression performance with object labels
        obj_mask = batch_cls_label < 3
        obj_sample_num = np.sum(obj_mask)
        total_obj_sample += obj_sample_num
        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val[obj_mask] == batch_label[obj_mask])
        total_correct += correct
        total_seen += (obj_sample_num*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds[obj_mask])
        iou3ds_sum += np.sum(iou3ds[obj_mask])
        iou3d_correct_cnt += np.sum(iou3ds[obj_mask]>=0.7)

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('classification accuracy: %f' % \
                (total_cls_correct / float(total_cls_seen)))
            if total_seen > 0:
                log_string('segmentation accuracy: %f' % \
                    (total_correct / float(total_seen)))
            log_string('box IoU (ground/3D): %f / %f' % \
                (iou2ds_sum / float(total_obj_sample), iou3ds_sum / float(total_obj_sample)))
            log_string('box estimation accuracy (IoU=0.7): %f' % \
                (float(iou3d_correct_cnt)/float(total_obj_sample)))
            total_cls_correct = 0
            total_correct = 0
            total_cls_seen = 0
            total_seen = 0
            total_obj_sample = 0
            loss_sum = 0
            iou2ds_sum = 0
            iou3ds_sum = 0
            iou3d_correct_cnt = 0


def eval_one_epoch(sess, ops, test_writer):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    # To collect statistics
    total_cls_correct = 0
    total_cls_seen = 0
    total_seen_class = [0 for _ in range(NUM_OBJ_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_OBJ_CLASSES)]
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_obj_sample = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    # Simple evaluation with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_cls_label, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_feature_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['features_pl']: batch_feature_vec,
                     ops['cls_label_pl']: batch_cls_label,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training}

        summary, step, loss_val, loss_endpoints, cls_logits_val, logits_val, iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['loss_endpoints'], ops['cls_logits'], ops['logits'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        if np.isnan(loss_val):
            print('nan loss in batch: ', batch_idx)
            print('loss_endpoints: ', loss_endpoints)

        # classification acc
        cls_preds_val = np.argmax(cls_logits_val, 1)
        cls_correct = np.sum(cls_preds_val == batch_cls_label)
        total_cls_correct += cls_correct
        total_cls_seen += BATCH_SIZE
        for l in range(NUM_OBJ_CLASSES):
            total_seen_class[l] += np.sum(batch_cls_label==l)
            total_correct_class[l] += (np.sum((cls_preds_val==l) & (batch_cls_label==l)))

        # only calculate seg acc and regression performance with object labels
        obj_mask = batch_cls_label < 3
        obj_sample_num = np.sum(obj_mask)
        total_obj_sample += obj_sample_num
        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val[obj_mask] == batch_label[obj_mask])
        total_correct += correct
        total_seen += (obj_sample_num*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds[obj_mask])
        iou3ds_sum += np.sum(iou3ds[obj_mask])
        iou3d_correct_cnt += np.sum(iou3ds[obj_mask]>=0.7)

        # for i in range(BATCH_SIZE):
        #     segp = preds_val[i,:]
        #     segl = batch_label[i,:]
        #     part_ious = [0.0 for _ in range(NUM_SEG_CLASSES)]
        #     for l in range(NUM_SEG_CLASSES):
        #         if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0):
        #             part_ious[l] = 1.0 # class not present
        #         else:
        #             part_ious[l] = np.sum((segl==l) & (segp==l)) / \
        #                 float(np.sum((segl==l) | (segp==l)))

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('classification accuracy: %f' % \
        (total_cls_correct / float(total_cls_seen)))
    log_string('eval segmentation accuracy: %f'% \
        (total_correct / float(total_seen)))
    log_string('eval classification avg class acc: %f' % \
        (np.mean(np.array(total_correct_class) / \
            np.array(total_seen_class,dtype=np.float))))
    log_string('eval box IoU (ground/3D): %f / %f' % \
        (iou2ds_sum / float(total_obj_sample), iou3ds_sum / \
            float(total_obj_sample)))
    log_string('eval box estimation accuracy (IoU=0.7): %f' % \
        (float(iou3d_correct_cnt)/float(total_obj_sample)))
    box_estimation_acc = float(iou3d_correct_cnt)/float(total_obj_sample)
    mean_loss = loss_sum / float(num_batches)
    EPOCH_CNT += 1
    return mean_loss


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
