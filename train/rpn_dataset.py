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
    obj.t[0] = obj.t[0] + min(obj.w, obj.l)*r*(np.random.random()*2-1)
    obj.t[2] = obj.t[2] + min(obj.w, obj.l)*r*(np.random.random()*2-1)
    obj.w = obj.w*(1+np.random.random()*2*r-r)
    obj.l = obj.l*(1+np.random.random()*2*r-r)
    return obj

def is_near(prop1, prop2):
    c1 = np.array(prop1.t)
    c2 = np.array(prop2.t)
    r = max(prop1.w, prop1.l, prop1.h, prop2.w, prop2.l, prop2.h)
    return np.linalg.norm(c1-c2) < r / 2.0

def get_iou(bev_box1, bev_box2):
    p1 = Polygon(bev_box1)
    p2 = Polygon(bev_box2)
    intersection = p1.intersection(p2).area
    return intersection / (p1.area + p2.area - intersection)

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
        # self.feature_vec = proposal.roi_features
        # corresponding proposal without roi features
        prop_box = [proposal.t[0], proposal.t[1], proposal.t[2], proposal.l, proposal.w, proposal.h, proposal.ry]
        self.proposal = ProposalObject(prop_box, proposal.score, proposal.type)
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


class RPNDataset(object):
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
        self.split = split
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.fill_with_label = fill_with_label
        self.num_channel = 7
        # rpn_output_path = os.path.join(kitti_path, 'training/proposal_car_people')
        # def is_prop_file(f):
        #     return os.path.isfile(os.path.join(rpn_output_path, f)) and not '_roi' in f
        # proposal_files = [f for f in os.listdir(rpn_output_path) if is_prop_file(f)]
        # self.frame_ids = map(lambda x: x.replace('.txt', ''), proposal_files)
        # self.frame_ids = list(set(self.load_split_ids(split)).intersection(self.frame_ids))
        self.frame_ids = self.load_split_ids(split)
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

        self.sample_buffer = Queue(maxsize=1024)

        # roi_features of the first positive proposal, for generating proposal from label
        self.roi_feature_ = {}

    def get_proposals(self, data_idx):
        if data_idx in self.proposals:
            return self.proposals[data_idx]
        else:
            return []

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
        self._load_proposals('rpn_out_{0}.pkl'.format(self.split))
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

    def group_overlaps(self, objs, calib, iou_thres=0.01):
        bev_boxes = map(lambda obj: utils.compute_box_3d(obj, calib.P)[1][:4, [0,2]], objs)
        groups = []
        candidates = range(len(objs))
        while len(candidates) > 0:
            idx = candidates[0]
            group = [idx]
            for i in candidates[1:]:
                if get_iou(bev_boxes[idx], bev_boxes[i]) >= iou_thres:
                    group.append(i)
            for j in group:
                candidates.remove(j)
            groups.append(map(lambda i: objs[i], group))
            # groups.append(group)
        return groups

    def do_sampling(self, frame_data, pos_ratio=0.5, is_eval=False):
        samples = frame_data['samples']
        pos_idxs = frame_data['pos_idxs']
        neg_idxs = [i for i in range(0, len(samples)) if i not in pos_idxs]
        random.shuffle(neg_idxs)
        if is_eval:
            #need_neg = int(len(neg_idxs) * 0.5)
            #need_neg = len(neg_idxs)
            need_neg = 1
            #keep_idxs = pos_idxs + neg_idxs[:need_neg]
            keep_idxs = pos_idxs
        elif pos_ratio == 0.0:
            keep_idxs = neg_idxs
        elif pos_ratio == 1.0:
            keep_idxs = pos_idxs
        else:
            cyclist_idxs = [i for i in pos_idxs if samples[i].cls_label == g_type2onehotclass['Cyclist']]
            pedestrian_idxs = [i for i in pos_idxs if samples[i].cls_label == g_type2onehotclass['Pedestrian']]
            car_idxs = [i for i in pos_idxs if samples[i].cls_label == g_type2onehotclass['Car']]
            '''
            # downsample
            car_idxs = random.sample(car_idxs, int(len(car_idxs) * 0.5))
            # oversample
            cyclist_idxs = cyclist_idxs * 10
            pedestrian_idxs = pedestrian_idxs * 5
            pos_idxs = car_idxs + cyclist_idxs + pedestrian_idxs
            '''
            need_neg = int(len(pos_idxs) * ((1-pos_ratio)/pos_ratio))
            keep_idxs = pos_idxs + neg_idxs[:need_neg]
            #keep_idxs = pos_idxs
        random.shuffle(keep_idxs)
        p = 0
        n = 0
        for i in keep_idxs:
            if samples[i].cls_label != g_type2onehotclass['NonObject']:
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
        batch_proposal_score = np.zeros((bsize,), dtype=np.float32)
        for i in range(bsize):
            sample = samples[i]
            assert(sample.point_set.shape[0] == sample.seg_label.shape[0])
            if sample.point_set.shape[0] == 0:
                point_set = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                seg_label = np.array([0])
            else:
                # Resample
                choice = np.random.choice(sample.point_set.shape[0], self.npoints, replace=True)
                point_set = sample.point_set[choice, 0:self.num_channel]
                seg_label = sample.seg_label[choice]
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
            batch_label[i,:] = seg_label
            batch_heading_class[i] = sample.angle_class
            batch_heading_residual[i] = sample.angle_residual
            batch_size_class[i] = sample.size_class
            batch_size_residual[i] = sample.size_residual
            batch_rot_angle[i] = sample.rot_angle
            # batch_feature_vec[i] = sample.feature_vec
            frame_ids.append(sample.frame_id)
            batch_proposal_score[i] = sample.proposal.score
        return batch_data, batch_cls_label, batch_ious, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, \
            batch_rot_angle, batch_feature_vec, frame_ids, batch_proposal_score, is_last_batch

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

    def get_one_sample(self, proposal, pc_rect, image, calib, iou, gt_box_3d, gt_object):
        '''convert to frustum sample format'''
        prop_corners_image_2d, prop_corners_3d = utils.compute_box_3d(proposal, calib.P)
        if prop_corners_image_2d is None:
            print('skip proposal behind camera')
            return False
        # get points within proposal box
        _,prop_inds = extract_pc_in_box3d(pc_rect, prop_corners_3d)

        if gt_object is not None:
            obj_type = gt_object.type
            # expand points to cover the whole object
            # TODO: use dbscan instead of ground truth
            _,gt_inds = extract_pc_in_box3d(pc_rect, gt_box_3d)
            prop_inds = np.logical_or(prop_inds, gt_inds)
            pc_in_prop_box = pc_rect[prop_inds,:]
            seg_mask = np.zeros((pc_in_prop_box.shape[0]))

            _,inds = extract_pc_in_box3d(pc_in_prop_box, gt_box_3d)
            seg_mask[inds] = 1
            # Reject object without points
            if np.sum(seg_mask)==0 or len(pc_in_prop_box) < 5:
                print('Reject object with too few point')
                return False

            # Get 3D BOX heading
            heading_angle = gt_object.ry
            # Get 3D BOX size
            box3d_size = np.array([gt_object.l, gt_object.w, gt_object.h])
            # Get frustum angle
            image_points = calib.project_rect_to_image(pc_in_prop_box[:,:3])
            expand_image_points = np.concatenate((prop_corners_image_2d, image_points), axis=0)
            xmin, ymin = expand_image_points.min(0)
            xmax, ymax = expand_image_points.max(0)
            # TODO: frustum angle is important, make use of image
            # xmin,ymin,xmax,ymax = gt_object.box2d
            # xmin, ymin = prop_corners_image_2d.min(0)
            # xmax, ymax = prop_corners_image_2d.max(0)
            box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
            uvdepth = np.zeros((1,3))
            uvdepth[0,0:2] = box2d_center
            uvdepth[0,2] = 20 # some random depth
            box2d_center_rect = calib.project_image_to_rect(uvdepth)
            frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
                box2d_center_rect[0,0])
        else:
            pc_in_prop_box = pc_rect[prop_inds,:]
            seg_mask = np.zeros((pc_in_prop_box.shape[0]))
            obj_type = 'NonObject'
            gt_box_3d = np.zeros((8, 3))
            heading_angle = 0
            box3d_size = np.zeros((1, 3))
            frustum_angle = 0

        #############
        rot_angle = self.get_center_view_rot_angle(frustum_angle)

        ### RGB
        point_set_rgb = np.zeros((pc_in_prop_box.shape[0], pc_in_prop_box.shape[1]+3))
        image_points = calib.project_rect_to_image(pc_in_prop_box[:,:3])
        for i, pt in enumerate(image_points):
            x, y = pt
            if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                rgb = np.array([0.0,0.0,0.0])
            else:
                rgb = image[int(y)][int(x)].astype(np.float32) / 255
            point_set_rgb[i] = np.concatenate((pc_in_prop_box[i], rgb), axis=0)
        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(point_set_rgb, rot_angle)
        else:
            point_set = point_set_rgb

        # if obj_type == 'Pedestrian':
        #     img_show = np.zeros(image.shape)
        #     image_points = calib.project_rect_to_image(point_set[:,:3])
        #     for i, pt in enumerate(image_points):
        #         x, y = pt
        #         if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
        #             continue
        #         img_show[int(y)][int(x)] = point_set[i, 4:]
        #     cv2.imshow('image',img_show)
        #     cv2.waitKey(0)

        # ------------------------------ LABELS ----------------------------
        # classification
        # assert(obj_type in ['Car', 'Pedestrian', 'Cyclist', 'NonObject'])
        assert(obj_type in type_whitelist)
        cls_label = g_type2onehotclass[obj_type]

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
        from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
        # fig = draw_lidar(pc_rect)
        # fig = draw_lidar(pc_in_prop_box, pts_color=(1,1,1))
        fig = draw_lidar_simple(pc_in_prop_box[:, :3], pc_in_prop_box[:, 4:])
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
        bev_box = corners_3d[:4, [0,2]]
        box_l = label.l
        box_w = label.w
        box_h = label.h
        # Rotate to nearest multiple of 90 degrees
        box_ry = label.ry
        half_pi = np.pi / 2
        box_ry = np.abs(np.round(box_ry / half_pi) * half_pi)
        cos_ry = np.abs(np.cos(box_ry))
        sin_ry = np.abs(np.sin(box_ry))
        w = box_l * cos_ry + box_w * sin_ry
        l = box_w * cos_ry + box_l * sin_ry
        h = box_h

        prop_obj = ProposalObject(list(label.t) + [l, w, h, box_ry], 1, label.type, roi_features)
        _, corners_prop = utils.compute_box_3d(prop_obj, calib.P)
        bev_box_prop = corners_prop[:4, [0,2]]

        prop_poly = Polygon(bev_box_prop)
        gt_poly = Polygon(bev_box)
        intersection = prop_poly.intersection(gt_poly)
        iou = intersection.area / (prop_poly.area + gt_poly.area - intersection.area)
        return prop_obj, iou

    def visualize_proposals(self, pc_rect, prop_boxes, neg_boxes, gt_boxes):
        import mayavi.mlab as mlab
        from viz_util import draw_lidar, draw_gt_boxes3d
        fig = draw_lidar(pc_rect)
        fig = draw_gt_boxes3d(prop_boxes, fig, draw_text=False, color=(1, 0, 0))
        fig = draw_gt_boxes3d(neg_boxes, fig, draw_text=False, color=(0, 1, 0))
        fig = draw_gt_boxes3d(gt_boxes, fig, draw_text=False, color=(1, 1, 1))
        raw_input()

    def _load_proposals(self, proposal_path):
        with open(proposal_path, 'rb') as f:
            frame_ids = pickle.load(f)
            segmentation = pickle.load(f)
            centers = pickle.load(f)
            angles = pickle.load(f)
            sizes = pickle.load(f)
            proposal_boxes = pickle.load(f)
            nms_indices = pickle.load(f)
            scores = pickle.load(f)
            pc_choices = pickle.load(f)
        self.proposals = {}
        self.pc_choices = {}
        self.pc_seg = {}
        print('Loading proposals for {0} frames...'.format(len(frame_ids)))
        for i in range(len(frame_ids)):
            frame_id = frame_ids[i]
            self.pc_choices[frame_id] = pc_choices[i]
            self.pc_seg[frame_id] = segmentation[i]
            for j in range(len(nms_indices[i])):
                if nms_indices[i][j] == -1:
                    continue
                ind = nms_indices[i][j]
                # to ProposalObject
                x,y,z = centers[i][ind]
                l, h, w = sizes[i][ind]
                ry = angles[i][ind]
                proposal = ProposalObject(np.array([x,y,z,l, h, w, ry]))
                frame_id = int(frame_ids[i])
                if frame_id not in self.proposals:
                    self.proposals[frame_id] = []
                self.proposals[frame_id].append(proposal)
        print('Loading proposals done.')

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
        image = self.kitti_dataset.get_image(data_idx)
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
        groups = self.group_overlaps(proposals, calib, 0.7)
        proposals_reduced = []
        KEEP_OVERLAP = self.augmentX
        for g in groups:
            random.shuffle(g)
            proposals_reduced += g[:KEEP_OVERLAP]
        proposals = proposals_reduced
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
                sample = self.get_one_sample(prop_, pc_rect, image, calib, iou_with_gt, None, None)
                if sample:
                    samples.append(sample)
                    # neg_box.append(prop_corners_3d)
            elif iou_with_gt >= 0.5 \
                or (iou_with_gt >= 0.45 and objects[obj_idx].type in ['Pedestrian', 'Cyclist']):
                obj_type = objects[obj_idx].type
                if self.roi_feature_.get(obj_type) is None:
                    self.roi_feature_[obj_type] = prop_.roi_features
                # adjust proposal box with ground truth
                '''
                gt_prop, iou_with_gt = self.get_proposal_from_label(objects[obj_idx], calib, prop_.roi_features)
                prop_.t = gt_prop.t
                prop_.w = gt_prop.w
                prop_.h = gt_prop.h
                prop_.l = gt_prop.l
                prop_.ry = gt_prop.ry
                '''
                #####
                avg_iou.append(iou_with_gt)

                #prop = copy.deepcopy(prop_)
                prop = prop_
                sample = self.get_one_sample(prop, pc_rect, image, calib, iou_with_gt, gt_boxes_3d[obj_idx], objects[obj_idx])
                if sample:
                    pos_idxs.append(len(samples))
                    samples.append(sample)
                    recall[obj_idx] = 1
                    #_, prop_corners_3d = utils.compute_box_3d(prop, calib.P)
                    #pos_box.append(prop_corners_3d)
            else:
                continue

        # use groundtruth to generate proposal
        if self.fill_with_label:
            for i in range(len(objects)):
                if recall[i]:
                    continue
                # FIXME: use roi feature of the first found positive proposal now
                fake_roi_feature = self.roi_feature_.get(objects[i].type)
                if fake_roi_feature is None:
                    continue
                gt_prop, iou_with_gt = self.get_proposal_from_label(objects[i], calib, fake_roi_feature)

                for _ in range(self.augmentX):
                    prop = copy.deepcopy(gt_prop)
                    # if self.perturb_prop:
                    prop = random_shift_box3d(prop)
                    sample = self.get_one_sample(prop, pc_rect, image, calib, iou_with_gt, gt_boxes_3d[i], objects[i])
                    if sample:
                        pos_idxs.append(len(samples))
                        samples.append(sample)
                        recall[i] = 1
                        #_, prop_corners_3d = utils.compute_box_3d(prop, calib.P)
                        #pos_box.append(prop_corners_3d)

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
        augmentX = 5
        perturb_prop = False
        fill_with_label = True
    else:
        augmentX = 1
        perturb_prop = False
        fill_with_label = True
    dataset = RPNDataset(512, kitti_path, 16, split, save_dir='./avod_dataset_car_people/'+split,
                 augmentX=augmentX, random_shift=False, rotate_to_center=True, random_flip=False,
                 perturb_prop=perturb_prop, fill_with_label=fill_with_label)
    dataset.preprocess()
    '''
    produce_thread = threading.Thread(target=dataset.load_buffer_repeatedly, args=(1.0,))
    produce_thread.start()

    while(True):
        batch = dataset.get_next_batch()
        is_last_batch = batch[-1]
        print(batch[0][0][0])
        break
        if is_last_batch:
            break
    dataset.stop_loading()

    produce_thread.join()
    '''
