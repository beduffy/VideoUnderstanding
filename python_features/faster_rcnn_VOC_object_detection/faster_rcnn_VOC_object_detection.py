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
import argparse
import os

import os.path as osp
import sys
import json
# from ..utilities.globals import log, HEADER_SIZE

# print 'DIRECTORY AT THE START OF EXECUTION IS: ', os.getcwd()
os.chdir('/home/ben/Documents/py-faster-rcnn/tools/')
# print 'changing directory: ', os.getcwd()

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)
        print 'adding path: ', path

this_dir = os.getcwd()

# Add caffe to PYTHONPATH
this_dir_one_up = os.path.dirname(this_dir)
caffe_path = osp.join(this_dir_one_up, 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir_one_up, 'lib')
add_path(lib_path)

# print 'os.cwd', os.getcwd()
# print 'THIS DIRECTORY: ', this_dir
# print 'CAFFE path: ', caffe_path
# print 'Library path: ', lib_path
# print 'syspath: ', sys.path

import caffe
import cv2
import matplotlib.pyplot as plt
import numpy as np

from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
# print 'before imdetect cwd', os.getcwd()
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

def demo(net, image, json_struct, idx, num_images):
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
    print ('Detection took {:.3f}s for {:d} object proposals. Image no. {}/{}').format(timer.total_time, boxes.shape[0], idx, num_images)


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
            # log(cls, score) #TODO
            print s
            faster_rcnn_20.append({'class': cls, 'score': str(round(score, 3))})
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
            cv2.putText(image, s, (5, 5 + number_of_detections * 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #im = im[:, :, (2, 1, 0)]

    if not objects_found:
        print 'NOTHING FOUND'

    cv2.imshow('image', image)
    #TODO TODO CHANGE TO 100-200 SO EASIER TO SEE FOR DEMO

    cv2.waitKey(1)
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

def main_object_detect(json_struct_path, video_path):
    os.chdir('/home/ben/Documents/py-faster-rcnn/tools/')
    this_dir = os.getcwd()

    # Add caffe to PYTHONPATH
    this_dir_one_up = os.path.dirname(this_dir)
    caffe_path = osp.join(this_dir_one_up, 'caffe-fast-rcnn', 'python')
    add_path(caffe_path)

    # Add lib to PYTHONPATH
    lib_path = osp.join(this_dir_one_up, 'lib')
    add_path(lib_path)


    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    directory = os.path.dirname(video_path)
    image_directory_path = os.path.join(directory, 'images', 'full')
    # json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')
    # args = parse_args()
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

    caffe.set_mode_gpu()

    # if args.cpu_mode:
    #     caffe.set_mode_cpu()
    #     print 'cpu mode'
    # else:
    #     print 'gpu mode'
    #     caffe.set_mode_gpu()
    #     caffe.set_device(args.gpu_id)
    #     cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    with open(json_struct_path) as data_file:
        json_struct = json.load(data_file)

    num_images = len(json_struct['images'])
    for idx, image_info in enumerate(json_struct['images']):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        # print 'Detecting objects for {}'.format(image_info['image_name'])
        # print 'D'

        image_path = ('{0}/{1}').format(image_directory_path, image_info['image_name'])
        # print image_info['image_name']
        image = cv2.imread(image_path)
        demo(net, image, json_struct, idx, num_images)

    json.dump(json_struct, open(json_struct_path, 'w'), indent=4)

    # print 'DIRECTORY AT THE END OF EXECUTION IS: ', os.getcwd()
    os.chdir('/home/ben/VideoUnderstanding')
    # print 'curdir', os.getcwd()

print sys.argv
json_struct_path = sys.argv[1]
video_path = sys.argv[2]
main_object_detect(json_struct_path, video_path)

# json_struct_path = '/home/ben/VideoUnderstanding/example_images/Montage_-_The_Best_of_YouTubes_Mishaps_Involving_Ice_Snow_Cars_and_People/metadata/result_struct.json'
# with open(json_struct_path) as data_file:
#     json_struct = json.load(data_file)
# main_object_detect(json_struct, '/home/ben/VideoUnderstanding/example_images/Montage_-_The_Best_of_YouTubes_Mishaps_Involving_Ice_Snow_Cars_and_People/Montage_-_The_Best_of_YouTubes_Mishaps_Involving_Ice_Snow_Cars_and_People.mp4')