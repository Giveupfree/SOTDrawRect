import json
import os
import argparse

parser = argparse.ArgumentParser(description='creat json for got test tuning')
parser.add_argument('--result_path', type=str,default='./',
                    help='the path where raw results are saved')
parser.add_argument('--tracker_name', type=str,default='OSTrack')
parser.add_argument('--save_path', type=str,default='./',help='default is the current workpath')
args = parser.parse_args()

result_path=os.path.join(args.result_path,args.tracker_name)

traj_list=[]
meta={}

videos=os.listdir(result_path)
videos.sort()
video_abs=[os.path.join(result_path,i) for i in videos]

for i,j in zip(video_abs,videos):
    with open(os.path.join(i,j+'_001.txt'), 'r') as f:
        traj = [list(map(float, x.strip().split(',')))
                for x in f.readlines()]
        traj_list.append(traj)

for v,t in zip(videos,traj_list):
    meta[v]={}
    meta[v]['gt_rect']=t
    meta[v]['img_names']=[v+'/'+'{:0>8d}.jpg'.format(i) for i in range(1,len(t)+1)]
    meta[v]['init_rect']=t[0]
    meta[v]['video_dir']=v


json.dump(meta, open(os.path.join(args.save_path,'GOT-10k.json'), 'w'), indent=4, sort_keys=True)
