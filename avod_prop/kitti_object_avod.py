import os
import sys
import numpy as np
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
sys.path.append(os.path.join(BASE_DIR, '../kitti'))
import kitti_util as utils
import cPickle as pickle
from kitti_object import *

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(boxes[:,4])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        # WARNING: (x1, y1) must be the relatively small point
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick

class ProposalObject(object):
    def __init__(self, box_3d, score=0.0, type='Car'):
        # [x, y, z, l, w, h, ry]
        self.t = box_3d[0:3]
        self.l = box_3d[3]
        self.w = box_3d[4]
        self.h = box_3d[5]
        self.ry = box_3d[6]
        self.score = score
        self.type = type

class kitti_object_avod(kitti_object):
    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)
        # if split not in ['training', 'testing']:
        #     print('Unknown split: %s' % (split))
        #     exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.proposal_dir = os.path.join(self.split_dir, 'proposal_120000')

        # self.num_samples = len(os.listdir(self.image_dir))
        # print(self.num_samples)

    def get_proposals(self, idx, rpn_score_threshold=0.5, nms_iou_thres=0.3):
        assert(idx<self.num_samples)
        proposals_file_path = os.path.join(self.proposal_dir, '%06d.txt'%(idx))
        proposals_and_scores = np.loadtxt(proposals_file_path)
        proposal_boxes_3d = proposals_and_scores[:, 0:7]
        proposal_scores = proposals_and_scores[:, 7]

        # Apply score mask to proposals
        score_mask = proposal_scores > rpn_score_threshold
        # 3D box in the format [x, y, z, l, w, h, ry]
        proposal_boxes_3d = proposal_boxes_3d[score_mask]
        proposal_scores = proposal_scores[score_mask]
        proposal_objs = \
            [ProposalObject(proposal) for proposal in proposal_boxes_3d]
        for obj, score in zip(proposal_objs, proposal_scores):
            obj.score = score

        boxes = []
        box_scores = []
        calib = self.get_calibration(idx)
        for obj in proposal_objs:
            _, corners = utils.compute_box_3d(obj, calib.P)
            corners_velo = calib.project_rect_to_velo(corners)
            boxes.append(corners_velo)
            box_scores.append(obj.score)

        bev_boxes = list(map(lambda bs: [bs[0][1][0], bs[0][1][1], bs[0][3][0], bs[0][3][1], bs[1]], zip(boxes, box_scores)))
        bev_boxes = np.array(bev_boxes)
        print('before nms: {0}'.format(len(bev_boxes)))
        nms_idxs = non_max_suppression(bev_boxes, nms_iou_thres)
        print('after nms: {0}'.format(len(nms_idxs)))
        # boxes = [boxes[i] for i in nms_idxs]
        return [proposal_objs[i] for i in nms_idxs]
