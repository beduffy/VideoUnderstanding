#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
# from __future__ import absolute_import
import argparse
import os

import os.path as osp
import sys
import json
import decimal

os.getcwd()
print 'DIRECTORY AT THE START OF EXECUTION IS: ', osp.dirname(__file__)

print 'changing directory: ', os.chdir('/home/ben/Documents/py-faster-rcnn/tools/')

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)
        print 'adding path: ', path
        # sys.path.insert(0, path)

class SysModule(object):
    pass


# def import_non_local(name, local_module=None, path=None, full_name=None, accessor=SysModule()):
#     import imp, sys, os
#
#     path = path or sys.path[1:]
#     if isinstance(path, basestring):
#         path = [path]
#
#     if '.' in name:
#         package_name = name.split('.')[0]
#         f, pathname, desc = imp.find_module(package_name, path)
#         if pathname not in __path__:
#             __path__.insert(0, pathname)
#         imp.load_module(package_name, f, pathname, desc)
#         v = import_non_local('.'.join(name.split('.')[1:]), None, pathname, name, SysModule())
#         setattr(accessor, package_name, v)
#         if local_module:
#             for key in accessor.__dict__.keys():
#                 setattr(local_module, key, getattr(accessor, key))
#         return accessor
#     try:
#         f, pathname, desc = imp.find_module(name, path)
#         if pathname not in __path__:
#             __path__.insert(0, pathname)
#         module = imp.load_module(name, f, pathname, desc)
#         setattr(accessor, name, module)
#         if local_module:
#             for key in accessor.__dict__.keys():
#                 setattr(local_module, key, getattr(accessor, key))
#             return module
#         return accessor
#     finally:
#         try:
#             if f:
#                 f.close()
#         except:
#             pass

def import_non_local(name, custom_name=None):
    import imp, sys

    custom_name = custom_name or name

    print 'syspath: ', sys.path

    f, pathname, desc = imp.find_module(name, [sys.path[-1]])
    print f, pathname, desc
    module = imp.load_module(custom_name, f, pathname, desc)
    f.close()

    return module

# this_dir = osp.dirname(__file__)
this_dir = os.getcwd()

# Add caffe to PYTHONPATH
this_dir_one_up = os.path.dirname(this_dir)
caffe_path = osp.join(this_dir_one_up, 'caffe-fast-rcnn', 'python')
# add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir_one_up, 'lib')
add_path(lib_path)

# add_path(this_dir_one_up)

#utils lib to path
# utils_path = osp.join(this_dir_one_up, 'lib', 'utils')

print 'os.cwd', os.getcwd()
print 'THIS DIRECTORY: ', this_dir
print 'CAFFE path: ', caffe_path
print 'Library path: ', lib_path

print 'syspath: ', sys.path

import caffe
import cv2
import matplotlib.pyplot as plt
import numpy as np

from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms

# timer = import_non_local('utils.timer', 'bla')
# utils = import_non_local('utils.timer')
# utils = import_non_local('timer')
# print utils
from utils.timer import Timer
# print 'timer', timer
# from utils.timer import Timer
# from lib.utils.timer import Timer
# from utils.timer import timer
# import utils.timer.Timer
# from fast_rcnn/.test import im_detect

from fast_rcnn.test import im_detect

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image, json_struct, idx):
    """Detect object classes in an image using pre-computed object proposals."""

    #todo rename function
    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, image)
    timer.toc()
    print ('Detection took {:.3f}s for '
		'{:d} object proposals').format(timer.total_time, boxes.shape[0])


    # why not work here? unexpected indent?
	#im = im[:, :, (2, 1, 0)]
    # Visualize detections for each class
    number_of_detections = 0
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    faster_rcnn_20 = []
    objects_found = False
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue

        objects_found = True



        #score = ''
        for i in inds:
            number_of_detections += 1
            bbox = dets[i, :4]
            score = dets[i, -1]
            s = '{:s} {:.3f}'.format(cls, score)
            print cls, score
            faster_rcnn_20.append({'class': cls, 'score': str(round(score, 3))})
            # print bbox
            #cv2.rectangle(im, (bbox[0], bbox[1])), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
            cv2.putText(image, s, (5, 5 + number_of_detections * 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            #score += ' ' + str(score)

        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #im = im[:, :, (2, 1, 0)]

    if not objects_found:
        print 'NOTHING FOUND'

    cv2.imshow('image', image)
    cv2.waitKey(200)
    # json_struct['images'][idx]['object_lists'] = []
    json_struct['images'][idx]['object_lists'] = {}
    json_struct['images'][idx]['object_lists']['faster_rcnn_20'] = faster_rcnn_20

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def main_object_detect(json_struct, video_path):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    directory = os.path.dirname(video_path)
    image_directory_path = os.path.join(directory, 'images', 'full')
    json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')

    args = parse_args()
    #
    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                         'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    # caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                           NETS[args.demo_net][1])

    prototxt = '/home/ben/Documents/py-faster-rcnn/models/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt'
    caffemodel = '/home/ben/Documents/py-faster-rcnn/data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']

    for idx, image_info in enumerate(json_struct['images']):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(image_info['image_name'])

        image_path = ('{0}/{1}').format(image_directory_path, image_info['image_name'])
        print image_path
        image = cv2.imread(image_path)

        # cv2.imshow('image', image)
        # cv2.waitKey(1)
        demo(net, image, json_struct, idx)

    plt.show()

    print 'DIRECTORY AT THE END OF EXECUTION IS: ', osp.dirname(__file__)
    print 'DIRECTORY AT THE END OF EXECUTION IS: ', os.getcwd()
    json.dump(json_struct, open(json_struct_path, 'w'), indent=4)

# main_object_detect('bla', 'bla')