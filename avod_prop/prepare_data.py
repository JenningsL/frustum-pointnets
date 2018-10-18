''' Prepare Avod proposal output data

Author: Jennings L
Date: Oct 2018
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
from viz_util import draw_lidar, draw_gt_boxes3d
sys.path.append(os.path.join(BASE_DIR, '../kitti'))
import kitti_util as utils
import cPickle as pickle
from kitti_object_avod import *
import argparse
from shapely.geometry import Polygon, MultiPolygon

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]]
    box2d_corners[1,:] = [box2d[2],box2d[1]]
    box2d_corners[2,:] = [box2d[2],box2d[3]]
    box2d_corners[3,:] = [box2d[0],box2d[3]]
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds

def demo():
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))
    data_idx = 0

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    pc_velo = dataset.get_lidar(data_idx)[:,0:3]
    calib = dataset.get_calibration(data_idx)

    ## Draw lidar in rect camera coord
    #print(' -------- LiDAR points in rect camera coordination --------')
    #pc_rect = calib.project_velo_to_rect(pc_velo)
    #fig = draw_lidar_simple(pc_rect)
    #raw_input()

    # Draw 2d and 3d boxes on image
    print(' -------- 2D/3D bounding boxes in images --------')
    show_image_with_boxes(img, objects, calib)
    raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    #show_lidar_with_boxes(pc_velo, objects, calib)
    #raw_input()
    show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    raw_input()

    # Visualize LiDAR points on images
    print(' -------- LiDAR points projected to image plane --------')
    show_lidar_on_image(pc_velo, img, calib, img_width, img_height)
    raw_input()

    # Show LiDAR points that are in the 3d box
    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()

    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:,0:2] = imgfov_pts_2d
    cameraUVDepth[:,2] = imgfov_pc_rect[:,2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    xmin,ymin,xmax,ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])

def iou_2d(box1, box2):
    '''
    box: [start(x, y), end(x, y)]
    '''
    xx1 = np.maximum(box1[0][0], box2[0][0])
    yy1 = np.maximum(box1[0][1], box2[0][1])
    xx2 = np.minimum(box1[1][0], box2[1][0])
    yy2 = np.minimum(box1[1][1], box2[1][1])

    # compute the width and height of the bounding box
    # WARNING: (x1, y1) must be the relatively small point
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    overlap = (w * h)

    area1 = (box1[1][0] - box1[0][0]) * (box1[1][1] - box1[0][1])
    area2 = (box2[1][0] - box2[0][0]) * (box2[1][1] - box2[0][1])
    # compute the ratio of overlap
    return overlap / (area1 + area2 - overlap)

def find_match_label(prop_corners, labels_corners):
    '''
    Find label with largest IOU. Label boxes can be rotated in xy plane
    '''
    # labels = MultiPolygon(labels_corners)
    labels = map(lambda corners: Polygon(corners), labels_corners)
    target = Polygon(prop_corners)
    largest_iou = 0.0
    largest_idx = -1
    for i, label in enumerate(labels):
        area1 = label.area
        area2 = target.area
        intersection = target.intersection(label).area
        iou = intersection / (area1 + area2 - intersection)
        # print(area1, area2, intersection)
        print(iou)
        if iou > largest_iou:
            largest_iou = iou
            largest_idx = i
    print('largest_idx:', largest_idx)
    return largest_idx

def extract_proposal_data(idx_filename, split, output_filename, viz=False,
                       perturb_box2d=False, augmentX=1, type_whitelist=['Car'],
                       kitti_path=os.path.join(ROOT_DIR,'dataset/KITTI/object')):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    import mayavi.mlab as mlab
    dataset = kitti_object_avod(kitti_path, split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in rect camera coord
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis
    type_count = {key: 0 for key in type_whitelist}

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        # ground truth
        objects = dataset.get_label_objects(data_idx)
        # proposal boxes
        try:
            proposals = dataset.get_proposals(data_idx)
        except:
            print('proposal not found for: ', data_idx)
            continue
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]

        # TODO: use avod feature map output
        gt_boxes_xy = []
        gt_boxes_3d = []
        objects = filter(lambda obj: obj.type in type_whitelist, objects)
        for obj in objects:
            _, gt_corners_3d = utils.compute_box_3d(obj, calib.P)
            gt_boxes_xy.append(gt_corners_3d[:4, [0,2]])
            gt_boxes_3d.append(gt_corners_3d)

        for prop in proposals:
            prop_corners_image_2d, prop_corners_3d = utils.compute_box_3d(prop, calib.P)
            if prop_corners_image_2d is None:
                print('skip proposal behind camera')
                continue
            prop_box_xy = prop_corners_3d[:4, [0,2]]
            # get points within proposal box
            _,prop_inds = extract_pc_in_box3d(pc_rect, prop_corners_3d)
            pc_in_prop_box = pc_rect[prop_inds,:]
            # segmentation label
            label = np.zeros((pc_in_prop_box.shape[0]))
            # find corresponding label object
            obj_idx = find_match_label(prop_box_xy, gt_boxes_xy)
            if obj_idx == -1:
                # non-object
                obj_type = 'NonObject'
                gt_box_3d = np.zeros((8, 3))
                heading_angle = 0
                box3d_size = np.zeros((1, 3))
                frustum_angle = 0
            else:
                obj = objects[obj_idx]
                obj_type = obj.type
                gt_box_3d = gt_boxes_3d[obj_idx]

                _,inds = extract_pc_in_box3d(pc_in_prop_box, gt_box_3d)
                label[inds] = 1
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

                # visualize
                # if obj_type == 'NonObject':
                #     print('NonObject')
                #     continue
                # fig = draw_lidar(pc_rect)
                # # fig = draw_gt_boxes3d([gt_box_3d], fig, color=(1, 0, 0))
                # fig = draw_gt_boxes3d([prop_corners_3d], fig, draw_text=False, color=(1, 0, 0))
                # mlab.plot3d([0, box2d_center_rect[0][0]], [0, box2d_center_rect[0][1]], [0, box2d_center_rect[0][2]], color=(1,1,1), tube_radius=None, figure=fig)
                # raw_input()

            # Reject object without points
            # if np.sum(label)==0:
            #     continue


            id_list.append(data_idx)
            # box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
            box3d_list.append(gt_box_3d)
            input_list.append(pc_in_prop_box)
            label_list.append(label)
            type_list.append(obj_type)
            heading_list.append(heading_angle)
            box3d_size_list.append(box3d_size)
            frustum_angle_list.append(frustum_angle)

            # collect statistics
            pos_cnt += np.sum(label)
            all_cnt += pc_in_prop_box.shape[0]
            type_count[obj_type] += 1

    print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt)/len(id_list)))
    print('Sample numbers: %d' % len(input_list))
    print('Type count:', type_count)

    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(box3d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)

    if viz:
        import mayavi.mlab as mlab
        for i in range(len(id_list)):
            if type_list[i] == 'NonObject':
                continue
            p1 = input_list[i]
            seg = label_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def get_box3d_dim_statistics(idx_filename):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type=='DontCare':continue
            dimension_list.append(np.array([obj.l,obj.w,obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle','wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--gen_train', action='store_true', help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true', help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, peds and cycs')
    parser.add_argument('--kitti_path', help='Path to Kitti Object Data')
    args = parser.parse_args()

    if args.demo:
        demo()
        exit()

    if args.car_only:
        type_whitelist = ['Car', 'NonObject']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist', 'NonObject']
        output_prefix = 'frustum_carpedcyc_'

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASE_DIR)
    if args.gen_train:
        extract_proposal_data(\
            os.path.join(BASE_DIR, 'image_sets/train.small.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'train.pickle'),
            viz=False, perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist,
            kitti_path=args.kitti_path)

    if args.gen_val:
        extract_proposal_data(\
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'val.pickle'),
            viz=False, perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist,
            kitti_path=args.kitti_path)
