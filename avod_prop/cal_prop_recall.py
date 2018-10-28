from __future__ import print_function

import os
import sys
import numpy as np
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
sys.path.append(os.path.join(BASE_DIR, '../kitti'))
import kitti_util as utils
import cPickle as pickle
from kitti_object_avod import *
import argparse
from shapely.geometry import Polygon, MultiPolygon

def is_recall(label, proposals, cover_threshold=0.7):
    for prop in proposals:
        intersection = label.intersection(prop).area
        if intersection / label.area > cover_threshold:
            return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_path', help='Path to Kitti Object Data')
    parser.add_argument('--split', help='Which split to calculate, train/val')
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if args.split == 'train':
        idx_filename = os.path.join(BASE_DIR, 'image_sets/train.small.txt')
    elif args.split == 'val':
        idx_filename = os.path.join(BASE_DIR, 'image_sets/val.txt')
    else:
        raise Exception('unknown split %s' % args.split)

    type_whitelist = ['Car', 'Pedestrian', 'Cyclist', 'NonObject']
    dataset = kitti_object_avod(args.kitti_path, 'training')
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    recall_num = 0
    total_obj = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        # ground truth
        objects = dataset.get_label_objects(data_idx)
        proposals = dataset.get_proposals(data_idx, rpn_score_threshold=0.1, nms_iou_thres=0.7)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]

        props_bev = []
        for prop in proposals:
            _, prop_corners_3d = utils.compute_box_3d(prop, calib.P)
            props_bev.append(Polygon(prop_corners_3d[:4, [0,2]]))

        objects = filter(lambda obj: obj.type in type_whitelist, objects)
        for obj in objects:
            _, gt_corners_3d = utils.compute_box_3d(obj, calib.P)
            gt_bev = Polygon(gt_corners_3d[:4, [0,2]])

            if is_recall(gt_bev, props_bev):
                recall_num += 1
            total_obj += 1
    print(float(recall_num)/total_obj)

if __name__ == '__main__':
    main()
