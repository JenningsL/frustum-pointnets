from __future__ import print_function

import os
import sys
import numpy as np
import copy
import random
import threading
import time
import cPickle as pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
sys.path.append(os.path.join(ROOT_DIR, 'avod_prop'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
from prepare_data import extract_pc_in_box3d
from kitti_object_avod import *
import kitti_util as utils
from model_util import g_type2class, g_class2type, g_type2onehotclass, g_type_mean_size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, REG_IOU
from model_util import type_whitelist
from provider import *
from shapely.geometry import Polygon, MultiPolygon
from Queue import Queue

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

def is_near(prop1, prop2):
    c1 = np.array(prop1.t)
    c2 = np.array(prop2.t)
    r = max(prop1.w, prop1.l, prop1.h, prop2.w, prop2.l, prop2.h)
    return np.linalg.norm(c1-c2) < r / 2.0

class Sample(object):
    def __init__(self, idx, point_set, seg, box3d_center, angle_class, angle_residual,\
        size_class, size_residual, rot_angle, cls_label, proposal, heading_angle, iou):
        self.idx = idx
        self.heading_angle = heading_angle
        self.point_set = point_set
        self.seg_label = seg
        self.box3d_center = box3d_center
        self.angle_class = angle_class
        self.angle_residual = angle_residual
        self.size_class = size_class
        self.size_residual = size_residual
        self.rot_angle = rot_angle
        self.cls_label = cls_label
        self.feature_vec = proposal.roi_features
        # corresponding proposal
        self.proposal = proposal
        self.iou = iou

    def random_flip(self):
        if np.random.random()>0.5: # 50% chance flipping
            self.point_set[:,0] *= -1
            self.box3d_center[0] *= -1
            self.heading_angle = np.pi - self.heading_angle

        self.angle_class, self.angle_residual = angle2class(self.heading_angle,
            NUM_HEADING_BIN)

    def random_shift(self):
        box3d_center = self.box3d_center
        dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
        shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
        self.point_set[:,2] += shift
        self.box3d_center[2] += shift


class AvodDataset(object):
    def __init__(self, npoints, kitti_path, batch_size, split, save_dir,
                 augmentX=1, random_shift=False, rotate_to_center=False, random_flip=False,
                 perturb_prop=False, fill_with_label=False):
        self.npoints = npoints
        self.random_shift = random_shift
        self.random_flip = random_flip
        self.rotate_to_center = rotate_to_center
        self.kitti_path = kitti_path
        self.kitti_dataset = kitti_object_avod(kitti_path, 'training')
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.fill_with_label = fill_with_label
        self.num_channel = 4
        rpn_output_path = os.path.join(kitti_path, 'training/proposal_car_people')
        def is_prop_file(f):
            return os.path.isfile(os.path.join(rpn_output_path, f)) and not '_roi' in f
        proposal_files = [f for f in os.listdir(rpn_output_path) if is_prop_file(f)]
        self.frame_ids = map(lambda x: x.replace('.txt', ''), proposal_files)
        self.frame_ids = list(set(self.load_split_ids(split)).intersection(self.frame_ids))
        # self.frame_ids = self.frame_ids[:5]
        random.shuffle(self.frame_ids)
        self.cur_batch = -1
        self.load_progress = 0
        self.batch_size = batch_size
        self.augmentX = augmentX
        self.perturb_prop = perturb_prop

        self.sample_id_counter = -1 # as id for sample
        self.stop = False # stop loading thread
        self.last_sample_id = None

        self.sample_buffer = Queue(maxsize=2048)

        # roi_features of the first positive proposal, for generating proposal from label
        self.roi_feature_ = None

    def load_split_ids(self, split):
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            return [line.rstrip('\n') for line in f]

    def preprocess(self):
        start = time.time()
        npoints = 0
        obj_points = 0
        pos_count = 0
        neg_count = 0
        recall = 0
        has_obj_count = 0
        avg_iou = []
        type_count = {t: 0 for t in type_whitelist if t != 'NonObject'}
        for frame_id in self.frame_ids:
            frame_data = self.load_frame_data(frame_id)
            for sample in frame_data['samples']:
                if sample.cls_label != g_type2onehotclass['NonObject']:
                    type_count[g_class2type[sample.cls_label]] += 1
            if 'recall' in frame_data:
                has_obj_count += 1
                recall += frame_data['recall']
            if 'avg_iou' in frame_data:
                avg_iou += frame_data['avg_iou']
            with open(os.path.join(self.save_dir, frame_id+'.pkl'), 'wb') as f:
                pickle.dump(frame_data, f)
            print('preprocess progress: {}/{}'.format(self.load_progress, len(self.frame_ids)))
            for i in frame_data['pos_idxs']:
                npoints += len(frame_data['samples'][i].seg_label)
                obj_points += np.sum(frame_data['samples'][i].seg_label)
            pos_count += len(frame_data['pos_idxs'])
            neg_count += len(frame_data['samples']) - len(frame_data['pos_idxs'])
        print('preprocess done, cost time: {}'.format(time.time() - start))
        print('pos: {}, neg: {}'.format(pos_count, neg_count))
        print('sample of each class: ', type_count)
        print('recall: {}'.format(recall/has_obj_count))
        print('Avg iou: {}'.format(np.mean(avg_iou)))
        print('Avg points: {}, pos_ratio: {}'.format(npoints/pos_count, obj_points/npoints))

    def do_sampling(self, frame_data, pos_ratio=0.5, is_eval=False):
        samples = frame_data['samples']
        pos_idxs = frame_data['pos_idxs']
        neg_idxs = [i for i in range(0, len(samples)) if i not in pos_idxs]
        random.shuffle(neg_idxs)
        if is_eval:
            #need_neg = int(len(neg_idxs) * 0.5)
            need_neg = len(neg_idxs)
            keep_idxs = pos_idxs + neg_idxs[:need_neg]
        elif pos_ratio == 0.0:
            keep_idxs = neg_idxs
        elif pos_ratio == 1.0:
            keep_idxs = pos_idxs
        else:
            need_neg = int(len(pos_idxs) * ((1-pos_ratio)/pos_ratio)) + 10
            keep_idxs = pos_idxs + neg_idxs[:need_neg]
        random.shuffle(keep_idxs)
        p = 0
        n = 0
        for i in keep_idxs:
            if samples[i].cls_label == 0:
                p += 1
            else:
                n += 1
        kept_samples = [samples[i] for i in keep_idxs]

        # data augmentation
        for sample in kept_samples:
            if self.random_flip:
                sample.random_flip()
            if self.random_shift:
                sample.random_shift()

        print('Sampling result: pos {}, neg {}'.format(p, n))
        return kept_samples

    def stop_loading(self):
        self.stop = True

    def load_buffer_repeatedly(self, pos_ratio=0.5, is_eval=False):
        i = -1
        last_sample_id = None
        while not self.stop:
            frame_id = self.frame_ids[i]
            with open(os.path.join(self.save_dir, frame_id+'.pkl'), 'rb') as f:
                frame_data = pickle.load(f)
            samples = self.do_sampling(frame_data, pos_ratio=pos_ratio, is_eval=is_eval)
            for s in samples:
                s.frame_id = frame_id
                self.sample_buffer.put(s)
            # update last_sample_id
            if len(samples) > 0:
                last_sample_id = samples[-1].idx
            # reach end
            if i == len(self.frame_ids) - 1:
                self.last_sample_id = last_sample_id
                random.shuffle(self.frame_ids)
            i = (i + 1) % len(self.frame_ids)

    def get_next_batch(self):
        is_last_batch = False
        samples = []
        for _ in range(self.batch_size):
            sample = self.sample_buffer.get()
            samples.append(sample)
            if sample.idx == self.last_sample_id:
                is_last_batch = True
                self.last_sample_id = None
                break

        bsize = len(samples) # note that bsize can be smaller than self.batch_size
        batch_data = np.zeros((bsize, self.npoints, self.num_channel))
        batch_cls_label = np.zeros((bsize,), dtype=np.int32)
        batch_ious = np.zeros((bsize,), dtype=np.float32)
        batch_label = np.zeros((bsize, self.npoints), dtype=np.int32)
        batch_center = np.zeros((bsize, 3))
        batch_heading_class = np.zeros((bsize,), dtype=np.int32)
        batch_heading_residual = np.zeros((bsize,))
        batch_size_class = np.zeros((bsize,), dtype=np.int32)
        batch_size_residual = np.zeros((bsize, 3))
        batch_rot_angle = np.zeros((bsize,))
        batch_feature_vec = np.zeros((bsize, samples[0].feature_vec.shape[0]))
        frame_ids = []
        for i in range(bsize):
            sample = samples[i]
            point_set = copy.deepcopy(sample.point_set[:,0:self.num_channel])
            box3d_center = copy.deepcopy(sample.box3d_center)
            # Data Augmentation
            if self.random_shift:
                dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
                shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
                point_set[:,2] += shift
                box3d_center[2] += shift
            batch_data[i,...] = point_set
            batch_center[i,:] = box3d_center
            batch_cls_label[i] = sample.cls_label
            batch_ious[i] = sample.iou
            batch_label[i,:] = sample.seg_label
            batch_heading_class[i] = sample.angle_class
            batch_heading_residual[i] = sample.angle_residual
            batch_size_class[i] = sample.size_class
            batch_size_residual[i] = sample.size_residual
            batch_rot_angle[i] = sample.rot_angle
            batch_feature_vec[i] = sample.feature_vec
            frame_ids.append(sample.frame_id)
        return batch_data, batch_cls_label, batch_ious, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, \
            batch_rot_angle, batch_feature_vec, frame_ids, is_last_batch

    def get_center_view_rot_angle(self, frustum_angle):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + frustum_angle

    def get_center_view_point_set(self, points, rot_angle):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(points)
        return rotate_pc_along_y(point_set, rot_angle)

    def get_center_view_box3d_center(self, box3d, rot_angle):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (box3d[0,:] + box3d[6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), rot_angle).squeeze()

    def get_box3d_center(self, box3d):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (box3d[0,:] + box3d[6,:])/2.0
        return box3d_center

    def get_one_sample(self, proposal, pc_rect, calib, iou, gt_box_3d, gt_object):
        '''convert to frustum sample format'''
        prop_corners_image_2d, prop_corners_3d = utils.compute_box_3d(proposal, calib.P)
        if prop_corners_image_2d is None:
            print('skip proposal behind camera')
            return False
        # get points within proposal box
        # FIXME: sometimes this raise error
        try:
            _,prop_inds = extract_pc_in_box3d(pc_rect, prop_corners_3d)
        except Exception as e:
            print('extract_pc_in_box3d fail')
            return False

        pc_in_prop_box = pc_rect[prop_inds,:]
        # shuffle points order
        np.random.shuffle(pc_in_prop_box)
        # segmentation label
        seg_mask = np.zeros((pc_in_prop_box.shape[0]))

        if gt_object is not None:
            obj_type = gt_object.type

            # FIXME: sometimes this raise error
            try:
                _,inds = extract_pc_in_box3d(pc_in_prop_box, gt_box_3d)
            except Exception as e:
                print('extract_pc_in_box3d fail')
                return False

            seg_mask[inds] = 1
            # Reject object without points
            if np.sum(seg_mask)==0:
                print('Reject object without points')
                return False

            # Get 3D BOX heading
            heading_angle = gt_object.ry
            # Get 3D BOX size
            box3d_size = np.array([gt_object.l, gt_object.w, gt_object.h])
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
        else:
            obj_type = 'NonObject'
            gt_box_3d = np.zeros((8, 3))
            heading_angle = 0
            box3d_size = np.zeros((1, 3))
            frustum_angle = 0

        #############
        rot_angle = self.get_center_view_rot_angle(frustum_angle)

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(pc_in_prop_box, rot_angle)
        else:
            point_set = pc_in_prop_box
        # empty point set
        if point_set.shape[0] == 0:
            point_set = np.array([[0.0, 0.0, 0.0, 0.0]])
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # ------------------------------ LABELS ----------------------------
        # classification
        # assert(obj_type in ['Car', 'Pedestrian', 'Cyclist', 'NonObject'])
        assert(obj_type in type_whitelist)
        cls_label = g_type2onehotclass[obj_type]

        if seg_mask.shape[0] == 0:
            seg_mask = np.array([0])
        seg_mask = seg_mask[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(gt_box_3d, rot_angle)
        else:
            box3d_center = self.get_box3d_center(gt_box_3d)

        # Heading
        if self.rotate_to_center:
            heading_angle = heading_angle - rot_angle

        # Size
        size_class, size_residual = size2class(box3d_size, obj_type)

        angle_class, angle_residual = angle2class(heading_angle,
            NUM_HEADING_BIN)

        self.sample_id_counter += 1
        return Sample(self.sample_id_counter, point_set, seg_mask, box3d_center, angle_class, angle_residual,\
            size_class, size_residual, rot_angle, cls_label, proposal, heading_angle, iou)

    def visualize_one_sample(self, pc_rect, pc_in_prop_box, gt_box_3d, prop_box_3d):
        import mayavi.mlab as mlab
        from viz_util import draw_lidar, draw_gt_boxes3d
        # fig = draw_lidar(pc_rect)
        fig = draw_lidar(pc_in_prop_box, pts_color=(1,1,1))
        # fig = draw_gt_boxes3d([gt_box_3d], fig, color=(1, 0, 0))
        # fig = draw_gt_boxes3d([prop_box_3d], fig, draw_text=False, color=(1, 1, 1))
        # roi_feature_map
        # roi_features_size = 7 * 7 * 32
        # img_roi_features = prop.roi_features[0:roi_features_size].reshape((7, 7, -1))
        # bev_roi_features = prop.roi_features[roi_features_size:].reshape((7, 7, -1))
        # img_roi_features = np.amax(img_roi_features, axis=-1)
        # bev_roi_features = np.amax(bev_roi_features, axis=-1)
        # fig1 = mlab.figure(figure=None, bgcolor=(0,0,0),
        #     fgcolor=None, engine=None, size=(500, 500))
        # fig2 = mlab.figure(figure=None, bgcolor=(0,0,0),
        #     fgcolor=None, engine=None, size=(500, 500))
        # mlab.imshow(img_roi_features, colormap='gist_earth', name='img_roi_features', figure=fig1)
        # mlab.imshow(bev_roi_features, colormap='gist_earth', name='bev_roi_features', figure=fig2)
        # mlab.plot3d([0, box2d_center_rect[0][0]], [0, box2d_center_rect[0][1]], [0, box2d_center_rect[0][2]], color=(1,1,1), tube_radius=None, figure=fig)
        raw_input()

    def get_proposal_from_label(self, label, calib, roi_features):
        '''construct proposal from label'''
        _, corners_3d = utils.compute_box_3d(label, calib.P)
        # wrap ground truth with box parallel to axis
        bev_box = corners_3d[:4, [0,2]]
        xmax = bev_box[:, 0].max(axis=0)
        ymax = bev_box[:, 1].max(axis=0)
        xmin = bev_box[:, 0].min(axis=0)
        ymin = bev_box[:, 1].min(axis=0)
        l = xmax - xmin
        w = ymax - ymin
        h = label.h
        prop_obj = ProposalObject(list(label.t) + [l, w, h, 0.0], 1, label.type, roi_features)
        _, corners_prop = utils.compute_box_3d(prop_obj, calib.P)
        bev_box_prop = corners_prop[:4, [0,2]]

        prop_poly = Polygon(bev_box_prop)
        gt_poly = Polygon(bev_box)
        intersection = prop_poly.intersection(gt_poly)
        iou = intersection.area / (prop_poly.area + gt_poly.area - intersection.area)
        # this iou maybe lower, force to use this for regression
        if iou < REG_IOU:
            iou = REG_IOU
        return prop_obj, iou

    def visualize_proposals(self, pc_rect, prop_boxes, neg_boxes, gt_boxes):
        import mayavi.mlab as mlab
        from viz_util import draw_lidar, draw_gt_boxes3d
        fig = draw_lidar(pc_rect)
        fig = draw_gt_boxes3d(prop_boxes, fig, draw_text=False, color=(1, 0, 0))
        fig = draw_gt_boxes3d(neg_boxes, fig, draw_text=False, color=(0, 1, 0))
        fig = draw_gt_boxes3d(gt_boxes, fig, draw_text=False, color=(1, 1, 1))
        raw_input()

    def load_frame_data(self, data_idx_str):
        '''load data for the first time'''
        # if os.path.exists(os.path.join(self.save_dir, frame_id+'.pkl')):
        #     with open(os.path.join(self.save_dir, frame_id+'.pkl'), 'rb') as f:
        #         return pickle.load(f)
        start = time.time()
        data_idx = int(data_idx_str)
        # print(data_idx_str)
        calib = self.kitti_dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = self.kitti_dataset.get_label_objects(data_idx)
        proposals = self.kitti_dataset.get_proposals(data_idx, rpn_score_threshold=0.1)
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
        recall = np.zeros((len(objects),))

        samples = []
        pos_idxs = []
        pos_box = []
        neg_box = []
        avg_iou = []
        for prop_ in proposals:
            prop_corners_image_2d, prop_corners_3d = utils.compute_box_3d(prop_, calib.P)
            if prop_corners_image_2d is None:
                # print('skip proposal behind camera')
                continue

            prop_box_xy = prop_corners_3d[:4, [0,2]]
            # find corresponding label object
            obj_idx, iou_with_gt = self.find_match_label(prop_box_xy, gt_boxes_xy)

            # iou < 0.3 is no object, iou >= 0.5 is object, iou > 0.65 will be used to
            # train regression
            if iou_with_gt < 0.3:
                # non-object
                sample = self.get_one_sample(prop_, pc_rect, calib, iou_with_gt, None, None)
                if sample:
                    samples.append(sample)
                    # neg_box.append(prop_corners_3d)
            elif iou_with_gt >= 0.5:
                if self.roi_feature_ is None:
                    self.roi_feature_ = prop_.roi_features
                avg_iou.append(iou_with_gt)

                for _ in range(self.augmentX):
                    prop = copy.deepcopy(prop_)
                    if self.perturb_prop:
                        prop = random_shift_box3d(prop)
                    sample = self.get_one_sample(prop, pc_rect, calib, iou_with_gt, gt_boxes_3d[obj_idx], objects[obj_idx])
                    if sample:
                        pos_idxs.append(len(samples))
                        samples.append(sample)
                        recall[obj_idx] = 1
                        # pos_box.append(prop_corners_3d)
                    # only do augmentation for those iou >= REG_IOU
                    if iou_with_gt < REG_IOU:
                        break
            else:
                continue

        # use groundtruth to generate proposal
        if self.fill_with_label and self.roi_feature_ is not None:
            for i in range(len(objects)):
                if recall[i]:
                    continue
                # FIXME: use roi feature of the first found positive proposal now
                gt_prop, iou_with_gt = self.get_proposal_from_label(objects[i], calib, self.roi_feature_)
                for _ in range(self.augmentX):
                    prop = copy.deepcopy(prop_)
                    if self.perturb_prop:
                        prop = random_shift_box3d(prop)
                    sample = self.get_one_sample(prop, pc_rect, calib, iou_with_gt, gt_boxes_3d[i], objects[i])
                    if sample:
                        pos_idxs.append(len(samples))
                        samples.append(sample)
                        recall[i] = 1

        # self.visualize_proposals(pc_rect, pos_box, neg_box, gt_boxes_3d)
        self.load_progress += 1
        print('load {} samples, pos {}'.format(len(samples), len(pos_idxs)))
        ret = {'samples': samples, 'pos_idxs': pos_idxs}
        if len(objects) > 0:
            ret['recall'] = np.sum(recall)/len(objects)
        if len(pos_idxs) > 0:
            ret['avg_iou'] = avg_iou
        return ret

    def find_match_label(self, prop_corners, labels_corners):
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
            # if a proposal cover enough ground truth, take it as positive
            #if intersection / area1 >= 0.8:
            #    iou = 0.66
            # print(area1, area2, intersection)
            # print(iou)
            if iou > largest_iou:
                largest_iou = iou
                largest_idx = i
        return largest_idx, largest_iou

if __name__ == '__main__':
    kitti_path = sys.argv[1]
    split = sys.argv[2]
    if split == 'train':
        augmentX = 2
        perturb_prop = True
        fill_with_label = True
    else:
        augmentX = 1
        perturb_prop = False
        fill_with_label = True
    dataset = AvodDataset(512, kitti_path, 16, split, save_dir='./avod_dataset_car_people/'+split,
                 augmentX=augmentX, random_shift=True, rotate_to_center=True, random_flip=True,
                 perturb_prop=perturb_prop, fill_with_label=fill_with_label)
    dataset.preprocess()

    # produce_thread = threading.Thread(target=dataset.load_buffer_repeatedly, args=(1.0,))
    # produce_thread.start()
    #
    # while(True):
    #     batch = dataset.get_next_batch()
    #     is_last_batch = batch[-1]
    #     print(batch[1])
    #     if is_last_batch:
    #         break
    # dataset.stop_loading()
    #
    # produce_thread.join()
