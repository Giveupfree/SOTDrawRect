import json
import os
import cv2
import numpy as np
import argparse
testdir = {}

def LaSOT_Json(path, annos, name):
    assert name in ["LaSOT", "LaSOText"]
    attr = ["Illumination Variation", "Partial Occlusion", "Deformation", "Motion Blur", "Camera Motion", "Rotation",
            "Background Clutter", "Viewpoint Change", "Scale Variation", "Full Occlusion", "Fast Motion", "Out-of-View",
            "Low Resolution", "Aspect Ratio Change"]
    att = os.path.join(annos, 'att')
    absent = os.path.join(annos, 'absent')
    name_lists = os.listdir(path)
    name_lists.sort()
    for name_list in name_lists:
        video_names = os.path.join(path, name_list)
        video_names_list = os.listdir(video_names)
        video_names_list.sort(key=lambda x: int(x.split('-')[-1]))
        for video_name in video_names_list:
            print(video_name)
            testdir[video_name] = {}
            testdir[video_name]["video_dir"] = video_name
            groundtruth_path = os.path.join(annos, video_name) + ".txt"
            images_path = os.path.join(video_names, video_name, 'img')
            absents_dir = os.path.join(absent, video_name) + '.txt'
            atts_dir = os.path.join(att, video_name) + '.txt'
            with open(groundtruth_path, 'r') as f:
                groundtruth = f.readlines()
            for idx, gt_line in enumerate(groundtruth):
                gt_image = gt_line.strip().split(',')
                bbox = [int(float(gt_image[0])), int(float(gt_image[1])), int(float(gt_image[2])),
                        int(float(gt_image[3]))]
                if idx == 0:
                    testdir[video_name]["init_rect"] = bbox
                    testdir[video_name]["img_names"] = []
                    testdir[video_name]["gt_rect"] = []
                testdir[video_name]["gt_rect"].append(bbox)
                im = cv2.imread(os.path.join(images_path, str(idx + 1).zfill(8) + '.jpg'))
                if im is None:
                    print(images_path, idx)
                    exit()
                img_name = video_name + "/img/" + str(idx + 1).zfill(8) + '.jpg'
                testdir[video_name]["img_names"].append(img_name)
            gt_ls = idx + 1

            with open(atts_dir, 'r') as f:
                atts = f.readlines()
            att_datas = atts[0]
            att_list = list(att_datas.split(','))
            testdir[video_name]["attr"] = []
            for idx, att_data in enumerate(att_list):
                if int(att_data) == 1:
                    testdir[video_name]["attr"].append(attr[idx])

            testdir[video_name]["absent"] = []
            with open(absents_dir, 'r') as f:
                absents = f.readlines()
            if len(absents) == 1:
                absents = absents[0]
                absents = list(absents.split(","))
            for absent_data in absents[:gt_ls]:
                testdir[video_name]["absent"].append(1 if int(absent_data) == 0 else 0)
    json.dump(testdir, open(name + '.json', 'w'))


if __name__ == '__main__' :
	parser = argparse.ArgumentParser(description='creat json for got test tuning')
	parser.add_argument('--dataset_path', type=str,default='./',
						help='The path is the LaSOT or LaSOT dataset path')
	parser.add_argument('--annos', type=str,default='./',help='The annos path in the toolkit officially provided by LaSOT(LaSOT_Evaluation_Toolkit)/LaSOText(LaSOT_Evaluation_Toolkit_V2)')
	# ²Î¼û£ºhttp://vision.cs.stonybrook.edu/~lasot/results.html
	# LaSOT_Evaluation_Toolkit : https://github.com/HengLan/LaSOT_Evaluation_Toolkit
	# LaSOT_Evaluation_Toolkit_V2: http://vision.cs.stonybrook.edu/~lasot/toolkit/LaSOT_Evaluation_Toolkit_V2.zip
	args = parser.parse_args()
    LaSOT_Json(path, args.annos, args.dataset_path)