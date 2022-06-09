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


```bash
git clone https://github.com/Giveupfree/SOTDrawRect.git
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
python draw_rect.py \                     
	--dataset_dir /path/to/dataset/root \		# dataset path
	--dataset VOT2018 \				# dataset name(VOT2019, VOT2018, VOT2016, OTB100, GOT10k, LaSOT, UAV123)
	--tracker_result_dir /path/to/tracker/dir \	# tracker dir
    	--format pdf \                                  # save fomat (pdf,png,jpg)
	--trackers ours ECO UPDT SiamRPNpp \ 		# tracker names 
    	--save_dir /path/to/save\                       # save dir
```

### Draw bounding boxes for a video sequence
```bash
cd /path/to/SOTDrawRect
python draw_rect.py \    
    	-- video videoname \                 
	--dataset_dir /path/to/dataset/root \		# dataset path
	--dataset VOT2018 \				# dataset name(VOT2019, VOT2018, VOT2016, OTB100, GOT10k, LaSOT, UAV123)
	--tracker_result_dir /path/to/tracker/dir \	# tracker dir
    	--format pdf \                                  # save fomat (pdf,png,jpg)
	--trackers ours ECO UPDT SiamRPNpp \ 	        # tracker names 
    	--save_dir /path/to/save\                       # save dir
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
1.根据Mixformer Tracker作者所提供的GOT10k结果所生成的GOT10k测试集的json文件“GOT-10k-test.json”，方便用户进行搜参操作。(搜参或测试时，记得将GOT-10k-test.json重命名为GOT-10k.json)。

2.新增“creat_got_json.py”文件，用于生成GOT-10k.json，其中raw results可从上got官网下载(http://got-10k.aitestunion.com/leaderboard)。

感谢某位不知名的大佬的支持。
