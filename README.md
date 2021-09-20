# SOTDrawRect
The purpose of this repo is to provide evaluation API of Current Single Object Tracking Dataset, including

- [x] [VOT2016](http://www.votchallenge.net/vot2016/dataset.html)
- [x] [VOT2018](http://www.votchallenge.net/vot2018/dataset.html)
- [x] [VOT2018-LT](http://www.votchallenge.net/vot2018/dataset.html)
- [x] [OTB100(OTB2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)
- [x] [UAV123](https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx)
- [x] [NFS](http://ci2cv.net/nfs/index.html)
- [x] [LaSOT](https://cis.temple.edu/lasot/)
- [ ] [TrackingNet (Evaluation on Server)](https://tracking-net.org)
- [ ] [GOT-10k (Evaluation on Server)](http://got-10k.aitestunion.com)

## Install 


```bash
git clone https://github.com/Giveupfree/SOTDrawRect.git
pip install -r requirements.txt
cd toolkit/utils/
python setup.py build_ext --inplace
# if you need to draw graph, you need latex installed on your system
```
## Update toolkit(optional)
The contents of the entire toolkit folder can be replaced directly from [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit)


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
python bin/draw_rect.py \                     
	--dataset_dir /path/to/dataset/root \		# dataset path
	--dataset VOT2018 \				# dataset name(VOT2018, VOT2016, OTB100, GOT10k)
	--tracker_result_dir /path/to/tracker/dir \	# tracker dir
    --format pdf \                              # save fomat (pdf,png,jpg)
	--trackers ours ECO UPDT SiamRPNpp \ 			# tracker names 
    --save_dir \                                  # save dir
```

### Draw a bounding box for a video sequence
```bash
cd /path/to/SOTDrawRect
python bin/draw_rect.py \    
    -- video videoname \                 
	--dataset_dir /path/to/dataset/root \		# dataset path
	--dataset VOT2018 \				# dataset name(VOT2018, VOT2016, OTB100, GOT10k)
	--tracker_result_dir /path/to/tracker/dir \	# tracker dir
    --format pdf \                              # save fomat (pdf,png,jpg)
	--trackers ours ECO UPDT SiamRPNpp \ 			# tracker names 
    --save_dir \                                  # save dir
```
