import os
import json
import argparse


## 使用之前需要从作者打包的results集合中，把该数据集内所有的txt文件单独复制到archive_path里去
parser = argparse.ArgumentParser(description='transfer the format of raw results from pytracking to pysot')
parser.add_argument('--dataset', type=str,default='OTB100',
        help='datasets')
parser.add_argument('--archive_path', type=str,default='/home/xiao/pythoncode/4.pytracking/pytracking/pytracking/util_scripts/unpack/atom/default_003',
        help='the path of raw results in pytracking style')
parser.add_argument('--saved_path', type=str,default='/home/xiao/pythoncode/4.pytracking/pytracking/pytracking/util_scripts/unpack/atom/tmp',
        help='the path to save raw results in pysot style')

args = parser.parse_args()

tmp=['Jogging-2.txt', 'Human4-2.txt', 'Skating2-1.txt', 'Jogging-1.txt', 'Skating2-2.txt']

videos=os.listdir(args.archive_path)
videos=[os.path.join(args.archive_path,i) for i in videos]
trajs=[]
for v in videos:
    with open(v,'r') as f:
        traj=[list(map(float,x.strip().split('\t'))) for x in f.readlines()]
        trajs.append(traj)
video_files=os.listdir(args.archive_path)
saved_path=os.path.join(args.saved_path,args.dataset)
if not os.path.isdir(saved_path):
    os.makedirs(saved_path)
for n,i in enumerate(video_files):
    with open(os.path.join(saved_path,i), 'w') as f:
        for x in trajs[n]:
            f.write(','.join([str(i) for i in x]) + '\n')
