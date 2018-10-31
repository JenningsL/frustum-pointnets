import tensorflow as tf
import os
import sys
import argparse

from avod.core import box_3d_encoder
from avod.core import evaluator_utils
from avod.core import summary_utils
from avod.core import trainer_utils
from avod.core.models.rpn_model import RpnModel
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import frustum_pointnets_v2

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

def get_detection_network(GPU_INDEX=0):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = frustum_pointnets_v2.get_model(pointclouds_pl, one_hot_vec_pl,
                is_training_pl)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        saver.restore(sess, MODEL_PATH)

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

    rpn_endpoints, sess1, rpn_model = get_proposal_network(model_config, dataset, rpn_model_path)
    detect_endpoints, sess2 = get_detection_network()

    feed_dict1 = rpn_model.create_feed_dict(sample_index=1)
    rpn_predictions = sess1.run(rpn_endpoints, feed_dict=feed_dict1)
    top_anchors = rpn_predictions[RpnModel.PRED_TOP_ANCHORS]
    top_proposals = box_3d_encoder.anchors_to_box_3d(top_anchors)
    softmax_scores = predictions[RpnModel.PRED_TOP_OBJECTNESS_SOFTMAX]

    proposals_and_scores = np.column_stack((top_proposals,
                                            softmax_scores))
    print(proposals_and_scores)
    return proposals_and_scores


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
