from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pointfly as pf
from pointcnn import PointCNN
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT

class PointCNNBoxNet(PointCNN):
    def __init__(self, points, features, is_training, setting):
        PointCNN.__init__(self, points, features, is_training, setting)
        with tf.variable_scope(setting.network_name):
            batch_size = points.get_shape()[0].value
            fc_flatten = tf.reshape(self.fc_layers[-1], [batch_size, -1])
            fc_flatten = tf.concat([fc_flatten, features], axis=1)
            fc1 = pf.dense(fc_flatten, 512, 'extra_fc_1', is_training)
            # fc1_drop = tf.layers.dropout(fc1, 0.0, training=is_training, name='extra_fc_1_drop')
            # self.fc_layers.append(fc1_drop)
            fc2 = pf.dense(fc1, 256, 'extra_fc_2', is_training)
            self.output = pf.dense(fc2, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, 'output',
                                   is_training, with_bn=False, activation=None)
