from __future__ import print_function

import os
import sys
import numpy as np
import random
from collections import defaultdict
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
        iou = intersection / (prop.area + label.area - intersection)
        # if intersection / label.area > cover_threshold and iou > 0.5:
        if iou > 0.5:
            return True
    return False

def print_statics(recall_num, total_obj, type_whitelist):
    for obj_type in type_whitelist:
        print('------------- %s -------------' % obj_type)
        print('{0} total num: {1}'.format(obj_type, total_obj[obj_type]))
        if total_obj[obj_type] > 0:
            type_recall = float(recall_num[obj_type])/total_obj[obj_type]
            print('Recall for {0} is {1:.4f}'.format(obj_type, type_recall))
    total_recall = float(sum(recall_num.values())) / sum(total_obj.values())
    print('Total recall: {0:.4f}'.format(total_recall))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_path', help='Path to Kitti Object Data')
    parser.add_argument('--split', help='Which split to calculate, train/val')
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if args.split == 'train':
        idx_filename = os.path.join(BASE_DIR, 'image_sets/train.txt')
    elif args.split == 'val':
        idx_filename = os.path.join(BASE_DIR, 'image_sets/val.txt')
    elif args.split == 'trainval':
        idx_filename = os.path.join(BASE_DIR, 'image_sets/trainval.txt')
    else:
        raise Exception('unknown split %s' % args.split)

    type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
    dataset = kitti_object_avod(args.kitti_path, 'training')
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    total_obj = defaultdict(int)
    recall_num = defaultdict(int)
    for i, data_idx in enumerate(data_idx_list):
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
            # 0 - easy, 1 - medium, 2 - hard
            if obj.difficulty not in [0, 1]:
                continue
            _, gt_corners_3d = utils.compute_box_3d(obj, calib.P)
            gt_bev = Polygon(gt_corners_3d[:4, [0,2]])

            if is_recall(gt_bev, props_bev):
                recall_num[obj.type] += 1
            total_obj[obj.type] += 1
        if i % 5 == 0:
            print_statics(recall_num, total_obj, type_whitelist)
    print_statics(recall_num, total_obj, type_whitelist)

if __name__ == '__main__':
    main()
