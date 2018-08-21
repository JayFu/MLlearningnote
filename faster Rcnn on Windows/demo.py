#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# Written by Jian Hu, based on code above
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.

Edit by Jian Hu：
it was a demo file, but i rewrite it as an predict file
acturally, the most predict part didnot change,
if you still want demo file, find the folder and demo_bk.py 
you will get it and use it as demo, only you have to do is change that file name
or dont, its up to you
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, os, cv2, shutil
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

# set the data url
data_url = ''

# set the classes
# 设置种类
CLASSES = ('__background__',
           'text')

# set the net and database
# 设置网络和数据集
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_20000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',)}


def vis_detections(im, class_name, dets, thresh=0.5):
    """locate the text location"""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    # used store location. 
    # this location include 2 points
    # 用于储存选框的位置，包含了两个点
    text_location = []
    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')

    # get the location
    # 获得位置
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        text_location.append(bbox)

    # center point location
    # 中心点位置
    location = []
    # count location
    # 计算坐标
    for i in range(len(text_location)):
        mmm = (text_location[i][0] + text_location[i][2]) // 2
        nnn = (text_location[i][1] + text_location[i][3]) // 2
        location.append(mmm)
        location.append(nnn)
    return location


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the image
    # 加载照片
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'lib', image_name)
    print(im_file)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    # 定位到所有的物体框
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    # 把定位结果显示出来
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args

def main_predict(data_url):
    args = parse_args()

    # model path
    # 模型路径
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('default', DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    # 设置
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    # 初始化会话
    sess = tf.Session(config=tfconfig)
    # load network
    # 加载网络
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 2,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # set imgs path
    # 设置图片读取路径
    im_names = os.listdir(data_url)
    # set file type in case of other file type
    # 设置文件类型以防其他类型文件
    file_type_list = ['jpg', 'png', 'JPG', 'PNG']
    # return value, list of points location
    # 用于返回一个保存中心点的列表
    list_of_location = []


    for im_name in im_names:
        # get file type used for filitering
        # 获取文件类型用于过滤
        file_type = im_name[-3:]
        if file_type not in file_type_list:
            continue
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('data for data/data/{}'.format(im_name))
        # demo() return the location of center point
        # demo()函数返回中心点坐标
        list_of_location.append(demo(sess, net, im_name))
    return list_of_location

if __name__ == '__main__':
    pass
