from __future__ import print_function

import os
import sys
import numpy as np
import copy
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
sys.path.append(os.path.join(ROOT_DIR, 'avod_prop'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from prepare_data import extract_pc_in_box3d
from kitti_object_avod import *
import kitti_util as utils
from model_util import g_type2class, g_class2type, g_type2onehotclass, g_type_mean_size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from provider import *
from shapely.geometry import Polygon, MultiPolygon

type_whitelist = ['Car', 'Pedestrian', 'Cyclist', 'NonObject']

def random_shift_box3d(obj, shift_ratio=0.1):
    '''
    Randomly w, l, h
    '''
    r = shift_ratio
    # 0.9 to 1.1
    obj.t[0] = obj.t[0] + obj.l*r*(np.random.random()*2-1)
    obj.t[1] = obj.t[1] + obj.w*r*(np.random.random()*2-1)
    obj.w = obj.w*(1+np.random.random()*2*r-r)
    obj.l = obj.l*(1+np.random.random()*2*r-r)
    # obj.h = obj.h*(1+np.random.random()*2*r-r)
    return obj

class AvodDataset(object):
    def __init__(self, npoints, kitti_path, batch_size, split,
                 random_flip=False, random_shift=False, rotate_to_center=False):
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.kitti_path = kitti_path
        self.kitti_dataset = kitti_object_avod(kitti_path, 'training')
        self.proposal_per_frame = 1024
        self.num_channel = 4
        rpn_output_path = os.path.join(kitti_path, 'training/proposal')
        def is_prop_file(f):
            return os.path.isfile(os.path.join(rpn_output_path, f)) and not '_roi' in f
        proposal_files = [f for f in os.listdir(rpn_output_path) if is_prop_file(f)]
        self.frame_ids = map(lambda x: x.replace('.txt', ''), proposal_files)
        self.frame_ids = list(set(self.load_split_ids(split)).intersection(self.frame_ids))
        self.cur_batch = -1
        self.load_progress = 0
        self.batch_size = batch_size
        self.augmentX = 1

        self.box3d_list = []
        self.input_list = []
        self.label_list = []
        self.type_list = []
        self.heading_list = []
        self.box3d_size_list = []
        self.frustum_angle_list = []
        self.roi_feature_list = []

    def load_split_ids(self, split):
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            return [line.rstrip('\n') for line in f]

    def is_all_loaded(self):
        return self.load_progress >= len(self.frame_ids)

    def shuffle_samples(self):
        '''shuffle on frames'''
        random.shuffle(self.frame_ids)

    def get_next_batch(self):
        is_last_batch = False
        self.cur_batch += 1
        start = self.cur_batch * self.batch_size
        end = start + self.batch_size
        while end > len(self.input_list) and not self.is_all_loaded():
            self.load_frame_data()
        if end >= len(self.input_list) and self.is_all_loaded():
            # reach end
            end = len(self.input_list)
            self.cur_batch = -1
            self.shuffle_samples()
            is_last_batch = True
        bsize = end - start
        batch_data = np.zeros((bsize, self.npoints, self.num_channel))
        batch_cls_label = np.zeros((bsize,), dtype=np.int32)
        batch_label = np.zeros((bsize, self.npoints), dtype=np.int32)
        batch_center = np.zeros((bsize, 3))
        batch_heading_class = np.zeros((bsize,), dtype=np.int32)
        batch_heading_residual = np.zeros((bsize,))
        batch_size_class = np.zeros((bsize,), dtype=np.int32)
        batch_size_residual = np.zeros((bsize, 3))
        batch_rot_angle = np.zeros((bsize,))
        batch_feature_vec = np.zeros((bsize, self.roi_feature_list[0].shape[0]))
        for i in range(bsize):
            ps,seg,center,hclass,hres,sclass,sres,rotangle,cls_label,feature_vec = \
                self.get_one_sample(i+start)
            batch_data[i,...] = ps[:,0:self.num_channel]
            batch_cls_label[i] = cls_label
            batch_label[i,:] = seg
            batch_center[i,:] = center
            batch_heading_class[i] = hclass
            batch_heading_residual[i] = hres
            batch_size_class[i] = sclass
            batch_size_residual[i] = sres
            batch_rot_angle[i] = rotangle
            batch_feature_vec[i] = feature_vec
        return batch_data, batch_cls_label, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, \
            batch_rot_angle, batch_feature_vec, is_last_batch

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
            self.get_center_view_rot_angle(index))

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), \
            self.get_center_view_rot_angle(index)).squeeze()

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return box3d_center

    def get_one_sample(self, index):
        rot_angle = self.get_center_view_rot_angle(index)

        feature_vec = self.roi_feature_list[index]

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]
        # empty point set
        if point_set.shape[0] == 0:
            point_set = np.array([[0.0, 0.0, 0.0, 0.0]])
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # ------------------------------ LABELS ----------------------------
        # classification
        cls_type = self.type_list[index]
        assert(cls_type in ['Car', 'Pedestrian', 'Cyclist', 'NonObject'])
        cls_label = g_type2onehotclass[cls_type]

        seg = self.label_list[index]
        if seg.shape[0] == 0:
            seg = np.array([0])
        seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.box3d_size_list[index],
            self.type_list[index])

        # Data Augmentation
        # if self.random_flip:
        #     # note: rot_angle won't be correct if we have random_flip
        #     # so do not use it in case of random flipping.
        #     if np.random.random()>0.5: # 50% chance flipping
        #         point_set[:,0] *= -1
        #         box3d_center[0] *= -1
        #         heading_angle = np.pi - heading_angle
        # if self.random_shift:
        #     dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
        #     shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
        #     point_set[:,2] += shift
        #     box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
            NUM_HEADING_BIN)

        return point_set, seg, box3d_center, angle_class, angle_residual,\
            size_class, size_residual, rot_angle, cls_label, feature_vec


    def load_frame_data(self):
        data_idx = int(self.frame_ids[self.load_progress])
        print(data_idx)
        calib = self.kitti_dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = self.kitti_dataset.get_label_objects(data_idx)
        proposals = self.kitti_dataset.get_proposals(data_idx, rpn_score_threshold=0.1, nms_iou_thres=0.8)
        pc_velo = self.kitti_dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        gt_boxes_xy = []
        gt_boxes_3d = []
        objects = filter(lambda obj: obj.type in type_whitelist, objects)
        for obj in objects:
            _, gt_corners_3d = utils.compute_box_3d(obj, calib.P)
            gt_boxes_xy.append(gt_corners_3d[:4, [0,2]])
            gt_boxes_3d.append(gt_corners_3d)
        for prop_ in proposals:
            prop_corners_image_2d, prop_corners_3d = utils.compute_box_3d(prop_, calib.P)
            if prop_corners_image_2d is None:
                # print('skip proposal behind camera')
                continue

            prop_box_xy = prop_corners_3d[:4, [0,2]]
            # find corresponding label object
            obj_idx, iou_with_gt = self.find_match_label(prop_box_xy, gt_boxes_xy)

            if obj_idx == -1:
                # non-object
                obj_type = 'NonObject'
                gt_box_3d = np.zeros((8, 3))
                heading_angle = 0
                box3d_size = np.zeros((1, 3))
                frustum_angle = 0
                # neg_proposals_in_frame.append(prop_corners_3d)

                # get points within proposal box
                # FIXME: sometimes this raise error
                try:
                    _,prop_inds = extract_pc_in_box3d(pc_rect, prop_corners_3d)
                except Exception as e:
                    print(e)
                    continue

                pc_in_prop_box = pc_rect[prop_inds,:]
                # shuffle points order
                np.random.shuffle(pc_in_prop_box)
                label = np.zeros((pc_in_prop_box.shape[0]))

                self.box3d_list.append(gt_box_3d)
                self.input_list.append(pc_in_prop_box)
                self.label_list.append(label)
                self.type_list.append(obj_type)
                self.heading_list.append(heading_angle)
                self.box3d_size_list.append(box3d_size)
                self.frustum_angle_list.append(frustum_angle)
                self.roi_feature_list.append(prop_.roi_features)
            else:
                # only do augmentation on objects
                for _ in range(self.augmentX):
                    prop = copy.deepcopy(prop_)
                    if self.random_shift:
                        prop = random_shift_box3d(prop)
                    prop_corners_image_2d, prop_corners_3d = utils.compute_box_3d(prop, calib.P)
                    if prop_corners_image_2d is None:
                        # print('skip proposal behind camera')
                        continue
                    # get points within proposal box
                    # FIXME: sometimes this raise error
                    try:
                        _,prop_inds = extract_pc_in_box3d(pc_rect, prop_corners_3d)
                    except Exception as e:
                        print(e)
                        continue

                    pc_in_prop_box = pc_rect[prop_inds,:]
                    # shuffle points order
                    np.random.shuffle(pc_in_prop_box)
                    # segmentation label
                    label = np.zeros((pc_in_prop_box.shape[0]))

                    obj = objects[obj_idx]
                    obj_type = obj.type
                    gt_box_3d = gt_boxes_3d[obj_idx]

                    # FIXME: sometimes this raise error
                    try:
                        _,inds = extract_pc_in_box3d(pc_in_prop_box, gt_box_3d)
                    except Exception as e:
                        print(e)
                        continue

                    label[inds] = 1
                    # Reject object without points
                    if np.sum(label)==0:
                        print('Reject object without points')
                        continue
                    # pos_proposals_in_frame.append(prop_corners_3d)
                    # Get 3D BOX heading
                    heading_angle = obj.ry
                    # Get 3D BOX size
                    box3d_size = np.array([obj.l, obj.w, obj.h])
                    # Get frustum angle
                    xmin, ymin = prop_corners_image_2d.min(0)
                    xmax, ymax = prop_corners_image_2d.max(0)
                    box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                    uvdepth = np.zeros((1,3))
                    uvdepth[0,0:2] = box2d_center
                    uvdepth[0,2] = 20 # some random depth
                    box2d_center_rect = calib.project_image_to_rect(uvdepth)
                    frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
                        box2d_center_rect[0,0])

                    self.box3d_list.append(gt_box_3d)
                    self.input_list.append(pc_in_prop_box)
                    self.label_list.append(label)
                    self.type_list.append(obj_type)
                    self.heading_list.append(heading_angle)
                    self.box3d_size_list.append(box3d_size)
                    self.frustum_angle_list.append(frustum_angle)
                    self.roi_feature_list.append(prop_.roi_features)
        self.load_progress += 1

    def find_match_label(self, prop_corners, labels_corners, iou_threshold=0.5):
        '''
        Find label with largest IOU. Label boxes can be rotated in xy plane
        '''
        # labels = MultiPolygon(labels_corners)
        labels = map(lambda corners: Polygon(corners), labels_corners)
        target = Polygon(prop_corners)
        largest_iou = iou_threshold
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
        # print('largest_iou:', '<0.1' if largest_iou == 0.1 else largest_iou)
        return largest_idx, largest_iou

if __name__ == '__main__':
    kitti_path = sys.argv[1]
    dataset = AvodDataset(512, kitti_path, 16, 'train',
                 random_flip=True, random_shift=True, rotate_to_center=True)
    while(True):
        if dataset.get_next_batch()[-1]:
            break
