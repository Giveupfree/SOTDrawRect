# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import time
from multiprocessing import Pool
from tqdm import tqdm

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.bbox import get_axis_aligned_bbox
from siamban.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

'''
该文件将模型并行推理的最小线程单位设置为Video，即在一个数据集上的测试可以以多视频序列并行的方式来减少模型在数据集上总的测试的时间。
取消了跟踪结果可视化的功能。
该文件放至tools目录下使用，使用方法同原始的test.py文件。

说明：
请留意模型在正常测试时（使用test.py）的GPU占用率（在命令行使用nvidia-smi命令查看），
尽量使利用率的数值与并行线程数的乘机靠近100%。
如某一个模型在GOT上推理的GPU占用率为23%，则理论最优并行线程数为100/23=4.3，即可将下列参数中的‘--thread’参数设置为4或5,来减少总的测试时间。

注意：
1.尽量不要将线程数设置过大（超过CPU线程数或总GPU利用率的数值大于100%）
2.python设置并行线程本身会有一定开销，如单线程测试时间为370s，双线程测试时间为200s，4线程测试时间为120s（理论GPU占用率均未超过100%）
3.设置线程数的时候也须注意总显存占用不要爆掉
4.本文件只为尽量缩小模型在数据集上的一次推理时间，若有多个模型/多个chekpoint/多个数据集测试的需求，建议写个shell脚本使用原始的test.py脚本进行
  并行推理，原因见2.
5.若出现‘AttributeError: Can't pickle local object 'ResNet.__init__.<locals>.<lambda>’报错，请将模型中用到‘lambda’函数的地方替换为
  以其他方法定义的函数，原因是multiprocessing库不支持‘lambda’函数


若发现bug欢迎反馈
'''


parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--dataset', type=str,default='VOT2018',
        help='datasets')
parser.add_argument('--config', default='config1.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='snapshot/checkpoint_e16.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--thread',default=2,type=int,
        help='threads for parallel test')
parser.add_argument('--gpu_id', default='not_set', type=str, 
        help="gpu id")

args = parser.parse_args()

if args.gpu_id != 'not_set':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.set_num_threads(1)

def main():
    # load config
    torch.multiprocessing.set_start_method('spawn')
    cfg.merge_from_file(args.config)


    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    trackers = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    trackers = [trackers for _ in range(len(dataset))]

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    ind = list(range(len(dataset)))
    paras = [i for i in dataset]
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking

        with Pool(processes=args.thread) as p:
            for loss_num in tqdm(p.imap_unordered(vot_test, zip(ind, paras, trackers)), desc='testing', total=len(dataset),
                          ncols=100):
                total_lost=total_lost+loss_num

        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        with Pool(processes=args.thread) as p:
            for _ in tqdm(p.imap_unordered(ope_test, zip(ind, paras, trackers)), desc='testing',
                                 total=len(dataset),
                                 ncols=100):
                ...

def vot_test(parameters):
    cfg.merge_from_file(args.config)
    v_idx, video, tracker=parameters

    if args.video != '':
        # test one special video
        if not args.video in video.name:
            return
    frame_counter = 0
    lost_number = 0
    toc = 0
    pred_bboxes = []
    for idx, (img, gt_bbox) in enumerate(video):
        if len(gt_bbox) == 4:
            gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                       gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                       gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
        tic = cv2.getTickCount()
        if idx == frame_counter:
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]  ##x1y1wh
            tracker.init(img, gt_bbox_)
            pred_bbox = gt_bbox_
            pred_bboxes.append(1)
        elif idx > frame_counter:
            outputs = tracker.track(img)
            pred_bbox = outputs['bbox']
            overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
            if overlap > 0:
                # not lost
                pred_bboxes.append(pred_bbox)
            else:
                # lost object
                pred_bboxes.append(2)
                frame_counter = idx + 5  # skip 5 frames
                lost_number += 1
        else:
            pred_bboxes.append(0)
        toc += cv2.getTickCount() - tic
        if idx == 0:
            cv2.destroyAllWindows()

    toc /= cv2.getTickFrequency()
    # save results
    model_name = args.snapshot.split('/')[-1].split('.')[0]
    video_path = os.path.join('results', args.dataset, model_name,
                              'baseline', video.name)
    if not os.path.isdir(video_path):
        os.makedirs(video_path)
    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
    with open(result_path, 'w') as f:
        for x in pred_bboxes:
            if isinstance(x, int):
                f.write("{:d}\n".format(x))
            else:
                f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
    print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
        v_idx + 1, video.name, toc, idx / toc, lost_number))

    return lost_number

def ope_test(parameters):
    cfg.merge_from_file(args.config)
    # OPE tracking
    v_idx, video, tracker=parameters
    if args.video != '':
        # test one special video
        if not args.video in video.name:
            return
    toc = 0
    pred_bboxes = []
    scores = []
    track_times = []
    for idx, (img, gt_bbox) in enumerate(video):
        tic = cv2.getTickCount()
        if idx == 0:
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            tracker.init(img, gt_bbox_)
            pred_bbox = gt_bbox_
            scores.append(None)
            if 'VOT2018-LT' == args.dataset:
                pred_bboxes.append([1])
            else:
                pred_bboxes.append(pred_bbox)
        else:
            outputs = tracker.track(img)
            pred_bbox = outputs['bbox']
            pred_bboxes.append(pred_bbox)
            scores.append(outputs['best_score'])
        toc += cv2.getTickCount() - tic
        track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
        if idx == 0:
            cv2.destroyAllWindows()

    toc /= cv2.getTickFrequency()
    model_name = args.snapshot.split('/')[-1].split('.')[0]
    # save results
    if 'VOT2018-LT' == args.dataset:
        video_path = os.path.join('results', args.dataset, model_name,
                                  'longterm', video.name)
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path,
                                   '{}_001.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        result_path = os.path.join(video_path,
                                   '{}_001_confidence.value'.format(video.name))
        with open(result_path, 'w') as f:
            for x in scores:
                f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
        result_path = os.path.join(video_path,
                                   '{}_time.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in track_times:
                f.write("{:.6f}\n".format(x))
    elif 'GOT-10k' in args.dataset:
        video_path = os.path.join('results', args.dataset, model_name, video.name)
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        result_path = os.path.join(video_path,
                                   '{}_time.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in track_times:
                f.write("{:.6f}\n".format(x))
    else:
        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
    print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
        v_idx + 1, video.name, toc, idx / toc))
    return


if __name__ == '__main__':
    start = time.time()
    main()
    duration=time.time()-start
    print('total seconds={:.0f}  total minutes={:.0f}  total hours={:.1f}'.format(duration,duration/60,duration/60/60))
