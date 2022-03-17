import os
import sys
import time
import argparse
import functools
sys.path.append("./")

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from pysot.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset
from pysot.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from pysot.visualization import draw_success_precision, draw_eao, draw_f1
from matplotlib import rc
# rc("text", usetex=False)
import matplotlib.pyplot as plt
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=10)
# --dataset_dir F:/datasets/VOT2019 --tracker_result_dir D:/资料/result/Banchmark/VOT2019 --dataset VOT2019 --trackers Ours SiamRPN++ ATOM SiamMask SA_SIAM_R SPM ARTCS SiamDW_ST DCFST
# -p ./results -d VOT2016 -r E:/datasets -n 1 --tracker_prefix CARAlle12 -s --vis
# --dataset_dir E:/datasets/OTB100 --dataset OTB100 --tracker_result_dir D:/资料/Project/SiamPD(train)/tune_results/OTB100/checkpoint_e17 --trackers 0.670checkpoint_e17_wi-0.588_pk-0.414_lr-0.527 0.670checkpoint_e17_wi-0.578_pk-0.456_lr-0.504 0.670checkpoint_e17_wi-0.472_pk-0.561_lr-0.519 --show_video_level
# --dataset_dir ../../../dataset/OTB100 --dataset OTB100 --tracker_result_dir D:/资料/Project/SiamPD(train)/results/OTB100/ --trackers SiamCAR Allocean0.35_0.2_0.45e11-0.686 Allocean0.35_0.2_0.45e12-0.682 Allocean0.35_0.2_0.45e13-0.681 Allocean0.35_0.2_0.45e14-0.691-P2 Allocean0.35_0.2_0.45e15-0.691-P3 Allocean0.35_0.2_0.45e16-0.691-P1 Allocean0.35_0.2_0.45e17 --show_video_level
# --dataset_dir E:/datasets/VOT2018 --dataset VOT2018 --tracker_result_dir D:/资料/Project/SiamPD(train)/results/VOT2018/ --trackers Allocean0.44_0.04_0.33e14-0.335 Allocean0.44_0.04_0.33e18 --show_video_level
# --dataset_dir E:/datasets/VOT2018 --dataset VOT2018 --tracker_result_dir D:/资料/Project/SiamPD(train)/results/VOT2018/ --trackers Allocean0.44_0.04_0.33e14-0.372 Allocean0.44_0.04_0.33 --show_video_level
# OTB100 ours SiamRPN++ SiamCAR DiMP50 Ocean-online Ocean-offline DaSiamRPN SiamBAN GCT ECO-HC
# --dataset_dir F:/datasets/OTB100 --tracker_result_dir D:/Tracker/TrFTB2022/results/OTB100 --dataset OTB100 --trackers GATAll2e11 GATAll2e12 GATAll2e13 GATAll2e14 GATAll2e15 GATAll2e16 GATAll2e17 GATAll2e18 GATAll2e19 GATAll2e20 GATAll2e21 GATAll2e22 GATAll2e23 GATAll2e24 GATAll2e25 GATAll2e26 GATAll2e27 GATAll2e28
# --dataset_dir F:/datasets/LaSOT --tracker_result_dir D:/资料/result/Banchmark/LaSOT --dataset LaSOT --trackers Ours(Transformer) Ours(DW-Corr) SiamCAR SiamBAN Ocean-offline GlobalTrack SiamRPN++ CLNet DaSiamRPN C-RPN --vis
# --dataset_dir F:/datasets/LaSOT --tracker_result_dir D:/Tracker/TrFTB2022/results/LaSOT --dataset LaSOT --trackers LaSOTmodel1e11 LaSOTmodel1e12 LaSOTmodel1e13 LaSOTmodel1e14 LaSOTmodel1e15
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--dataset_dir', type=str, help='dataset root directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--tracker_result_dir', type=str, help='tracker result root')
    parser.add_argument('--trackers', nargs='+')
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.add_argument('--show_video_level', dest='show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=1)
    args = parser.parse_args()

    tracker_dir = args.tracker_result_dir
    trackers = args.trackers
    root = args.dataset_dir

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'OTB' in args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                           name=dataset.name,
                                           videos=videos,
                                           attr=attr,
                                           precision_ret=precision_ret)
    elif 'LaSOT' in args.dataset:
        dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        # success_ret = benchmark.eval_success(trackers)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                           name=dataset.name,
                                           videos=videos,
                                           attr=attr,
                                           precision_ret=precision_ret,
                                           norm_precision_ret=norm_precision_ret)
        # if args.vis:
        #     draw_success_precision(success_ret,
        #             name=dataset.name,
        #             videos=dataset.attr['ALL'],
        #             attr='ALL',
        #             precision_ret=precision_ret,
        #             norm_precision_ret=norm_precision_ret)
    elif 'UAV' in args.dataset or 'UAV123' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                        name=dataset.name,
                        videos=videos,
                        attr=attr,
                        precision_ret=precision_ret)
    elif 'NFS' in args.dataset:
        dataset = NFSDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                            name=dataset.name,
                            video=videos,
                            attr=attr,
                            precision_ret=precision_ret)
    elif args.dataset in ['VOT2016', 'VOT2018', "VOT2019"]:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)
        # benchmark.show_result(ar_result)

        benchmark = EAOBenchmark(dataset)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        benchmark.show_result(eao_result)
        ar_benchmark.show_result(ar_result, eao_result,
                show_video_level=args.show_video_level)
        # draw_eao(eao_result)
    elif 'VOT2018-LT' == args.dataset:
        dataset = VOTLTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                show_video_level=args.show_video_level)
        if args.vis:
            draw_f1(f1_result)
