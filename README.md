# SOTDrawRect
The purpose of this repo is to provide evaluation API of Current Single Object Tracking Dataset, including
- [x] [VOT2016](http://www.votchallenge.net/vot2016/dataset.html)
- [x] [VOT2018](http://www.votchallenge.net/vot2018/dataset.html)
- [x] [VOT2018-LT](http://www.votchallenge.net/vot2018/dataset.html)
- [x] [VOT2019](http://www.votchallenge.net/vot2019/dataset.html)
- [x] [OTB100(OTB2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)
- [x] [UAV123](https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx)
- [x] [NFS](http://ci2cv.net/nfs/index.html)
- [x] [LaSOT](http://vision.cs.stonybrook.edu/~lasot/index.html)
- [x] [LaSOText](http://vision.cs.stonybrook.edu/~lasot/index.html)
- [ ] [TrackingNet (Evaluation on Server)](https://tracking-net.org)
- [ ] [GOT-10k (Evaluation on Server)](http://got-10k.aitestunion.com)

## Install 

You need to download versions of all json files at the same time
```bash
git clone https://github.com/Giveupfree/SOTDrawRect.git
pip install -r requirements.txt
cd toolkit/utils/
python setup.py build_ext --inplace
```

If you already have a json file, you only need to download the version of the code.
```bash
git clone -b nojson https://github.com/Giveupfree/SOTDrawRect.git
pip install -r requirements.txt
cd toolkit/utils/
python setup.py build_ext --inplace
```
## Update toolkit(optional)
The entire toolkit folder is from [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit), but here's a fix for some of its problems


## Download Dataset

Download json files used in our toolkit [baidu pan](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA) or [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI)

1. Put CVRP13.json, OTB100.json, OTB50.json in OTB100 dataset directory (you need to copy Jogging to Jogging-1 and Jogging-2, and copy Skating2 to Skating2-1 and Skating2-2 or using softlink)

   The directory should have the below format

   | -- OTB100/

   ​	| -- Basketball

   ​	| 	......

   ​	| -- Woman

   ​	| -- OTB100.json

   ​	| -- OTB50.json

   ​	| -- CVPR13.json

2. Put all other jsons in the dataset directory like in step 1

## Usage

### Draw rectangular boxes
```bash
cd /path/to/SOTDrawRect
export PYTHONPATH=./:$PYTHONPATH
python draw_rect.py \                     
	--dataset_dir /path/to/dataset/root \		# dataset path
	--dataset VOT2018 \				# dataset name(VOT2019, VOT2018, VOT2016, OTB100, GOT10k, LaSOT, UAV123)
	--tracker_result_dir /path/to/tracker/dir \	# tracker dir
    	--format pdf \                                  # save fomat (pdf,png,jpg)
	--trackers ours ECO UPDT SiamRPNpp \ 		# tracker names 
    	--save_dir /path/to/save                        # save dir
```

### Draw bounding boxes for a video sequence
```bash
cd /path/to/SOTDrawRect
export PYTHONPATH=./:$PYTHONPATH
python draw_rect.py \    
    	-- video videoname \                 
	--dataset_dir /path/to/dataset/root \		# dataset path
	--dataset VOT2018 \				# dataset name(VOT2019, VOT2018, VOT2016, OTB100, GOT10k, LaSOT, UAV123)
	--tracker_result_dir /path/to/tracker/dir \	# tracker dir
    	--format pdf \                                  # save fomat (pdf,png,jpg)
	--trackers ours ECO UPDT SiamRPNpp \ 	        # tracker names 
    	--save_dir /path/to/save                        # save dir
```

### Evaluation on VOT2018(VOT2016, VOT2019)
```bash
cd /path/to/pysot-toolkit
export PYTHONPATH=./:$PYTHONPATH
python bin/eval.py \
	--dataset_dir /path/to/dataset/root \		# dataset path
	--dataset VOT2018 \				# dataset name(VOT2018, VOT2016, VOT2019)
	--tracker_result_dir /path/to/tracker/dir \	# tracker dir
	--trackers ECO UPDT SiamRPNpp 			# tracker names 
	--vis                                           # draw graph

# you will see
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
| SiamRPNpp  |  0.600   |   0.234    |    50.0     | 0.415 |
|    UPDT    |  0.536   |   0.184    |    39.2     | 0.378 |
|    ECO     |  0.484   |   0.276    |    59.0     | 0.280 |
------------------------------------------------------------
```

### Evaluation on OTB100(UAV123, NFS, LaSOT, LaSOText)

converted *.txt tracking results will be released soon

```bash
cd /path/to/pysot-toolkit
export PYTHONPATH=./:$PYTHONPATH
python bin/eval.py \
	--dataset_dir /path/to/dataset/root \		# dataset path
	--dataset OTB100 \				# dataset name(OTB100, UAV123, NFS, LaSOT, LaSOText)
	--tracker_result_dir /path/to/tracker/dir \	# tracker dir
	--trackers SiamRPN++ C-COT DaSiamRPN ECO  \	# tracker names 
	--num 4 \				  	# evaluation thread
	--show_video_level \ 	  			# wether to show video results
	--vis 					  	# draw graph

# you will see (Normalized Precision not used in OTB evaluation)
-----------------------------------------------------
|Tracker name| Success | Norm Precision | Precision |
-----------------------------------------------------
| SiamRPN++  |  0.696  |     0.000      |   0.914   |
|    ECO     |  0.691  |     0.000      |   0.910   |
|   C-COT    |  0.671  |     0.000      |   0.898   |
| DaSiamRPN  |  0.658  |     0.000      |   0.880   |
-----------------------------------------------------

-----------------------------------------------------------------------------------------
|    Tracker name     |      SiamRPN++      |      DaSiamRPN      |         ECO         |
-----------------------------------------------------------------------------------------
|     Video name      | success | precision | success | precision | success | precision |
-----------------------------------------------------------------------------------------
|     Basketball      |  0.423  |   0.555   |  0.677  |   0.865   |  0.653  |   0.800   |
|        Biker        |  0.728  |   0.932   |  0.319  |   0.448   |  0.506  |   0.832   |
|        Bird1        |  0.207  |   0.360   |  0.274  |   0.508   |  0.192  |   0.302   |
|        Bird2        |  0.629  |   0.742   |  0.604  |   0.697   |  0.775  |   0.882   |
|      BlurBody       |  0.823  |   0.879   |  0.759  |   0.767   |  0.713  |   0.894   |
|      BlurCar1       |  0.803  |   0.917   |  0.837  |   0.895   |  0.851  |   0.934   |
|      BlurCar2       |  0.864  |   0.926   |  0.794  |   0.872   |  0.883  |   0.931   |
......
|        Vase         |  0.564  |   0.698   |  0.554  |   0.742   |  0.544  |   0.752   |
|       Walking       |  0.761  |   0.956   |  0.745  |   0.932   |  0.709  |   0.955   |
|      Walking2       |  0.362  |   0.476   |  0.263  |   0.371   |  0.793  |   0.941   |
|        Woman        |  0.615  |   0.908   |  0.648  |   0.887   |  0.771  |   0.936   |
-----------------------------------------------------------------------------------------
```

### Evaluation on VOT2018-LT
```bash
cd /path/to/pysot-toolkit
export PYTHONPATH=./:$PYTHONPATH
python bin/eval.py \
	--dataset_dir /path/to/dataset/root \		# dataset path
	--dataset VOT2018-LT \				# dataset name
	--tracker_result_dir /path/to/tracker/dir \	# tracker dir
	--trackers SiamRPN++ MBMD DaSiam-LT \		# tracker names 
	--num 4 \				  	# evaluation thread
	--vis  					  	# wether to draw graph

# you will see
-------------------------------------------
|Tracker Name| Precision | Recall |  F1   |
-------------------------------------------
| SiamRPN++  |   0.649   | 0.610  | 0.629 |
|    MBMD    |   0.634   | 0.588  | 0.610 |
| DaSiam-LT  |   0.627   | 0.588  | 0.607 |
|    MMLT    |   0.574   | 0.521  | 0.546 |
|  FuCoLoT   |   0.538   | 0.432  | 0.479 |
|  SiamVGG   |   0.552   | 0.393  | 0.459 |
|   SiamFC   |   0.600   | 0.334  | 0.429 |
-------------------------------------------
```

### Update 2022.1.31
1.Add JSON file required for LaSOText evaluation.

2.Update the VOT2019.json tags list and corresponding values, and modify its the picture path list.

### Update 2022.2.21
1.Fix the "RuntimeWarning: Mean of empty slice   acc = np.nanmean(overlaps)" warning when [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) runs eval.py to verify the VOT2019 Benchmark

### warning 2022.3.16
1.LaSOText.json中视频序列和注释长度不对应问题：实验发现，在paddle序列中出现视频长度和注释数目(官方提供的数据集和注释)和LaSOT官方提供的工具箱中，与其工具箱所提供的其它跟踪器的结果所对应的视频长度不一致，其中包括数据集的视频序列长度和注释少于跟踪器结果的长度，也出现多于踪器结果的情况。已知这是LaSOT官方的问题，因此慎用LaSOText.json文件做测试，等待LaSOT官方修复，这里也将及时对LaSOText.json文件做出更新。

### Update 2022.3.17
1.暂时修正paddle-1视频序列长度（同LaSOT工具箱中的一致）

2.使官方提供的其他跟踪器在paddle视频序列下的有效长度与LaSOText所公开的视频序列长度和注释保持一致。
### warning 2022.3.17
经上述修改后若使用修正后的工具箱测评，成功率结果依然是正常的，但是准确率和归一化准确率结果由于与LaSOText官方的有一定的差别，因此仅提供参考。

### Update 2022.3.23
1.新增VOT系列各个属性的雷达图绘代码

2.修复原有VOT系列各个属性的雷达图绘制中不能绘制下滑线的问题

### Update 2022.4.02
1.新增LaSOText2.json，其中LaSOText2.json中paddle-1中的数据长度和官方数据集所提供的长度保持一致。如果想利用官方提供的跟踪器结果文件绘制曲线，建议使用LaSOText.json。LaSOText2.json则是为了方便用户和近期的相关论文中提供的结果进行比较。

### Update 2022.06.09
1.根据OSTrack Tracker作者所提供的GOT10k结果所生成的GOT10k测试集的json文件“GOT-10k-test.json”，方便用户进行搜参操作。(搜参或测试时，记得将GOT-10k-test.json重命名为GOT-10k.json)。

2.新增“creat_got_json.py”文件，用于生成GOT-10k.json，其中raw results可从上got官网下载(http://got-10k.aitestunion.com/leaderboard) 。

3.新增"LaSOT.json/LaSOText.json" 文件生成方式

感谢某位不知名的大佬的支持。

### Update 2022.08.02
新增test_parallel.py脚本，该脚本将模型并行推理的最小线程单位设置为Video，即在一个数据集上的测试可以以多视频序列并行的方式来减少模型在数据集上总的测试的时间。
具体使用方法见文件内注释。

### Update 2022.09.24
1.更新GOT-10k-test.json，结果来自MixViT

### Update 2022.11.20
1.更新GOT-10k-test.json，结果来自GOT-10k榜首

2.更新creat_got_json.py文件，主要针对Linux下的搜参问题，更新后的文件生成的json文件可同时适用于windows和Linux

### Update 2022.11.21
1.修复2022.11.20号的一个代码bug

### Update 2022.11.26
1.修复bin/eval.py中的一个bug

### Update 2023.01.16
1.更新GOT-10k-test.json，结果来自GOT-10k榜首

### Update 2023.07.16
1.更新GOT-10k-test.json，结果来自GOT-10k榜首
