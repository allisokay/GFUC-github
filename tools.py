# -*- coding: utf-8
"""
@author：67543
@date：  2022/3/20
@contact：675435108@qq.com
"""
import time
import os
import argparse
import warnings
from modules.PoissonFactorModel import PoissonFactorModel
from modules.MultiGaussianModel import MultiGaussianModel
from modules.TimeAwareMF import TimeAwareMF
from modules.GraphModel import GCN,Graph,TrainLoader
from modules.metrics import *
from modules.LocationFriendshipBookmarkColoringAlgorithm import LocationFriendshipBookmarkColoringAlgorithm
import numpy as np
import multiprocessing as mp
warnings.filterwarnings("ignore")
import json
import torch
from torch import nn,optim
from torch.nn import Module,Embedding
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

top_k = 100


def format_print(content, _type="", length=0):
        print(_type*length+content)


def get_config(args,origin='FUC'):
    run_time = time.strftime("%Y%m%d-%H:%M", time.localtime(time.time()))
    print("程序开始时间：" + run_time+"-"+origin)
    format_print("初始化参数，生成目录...")
    temp_path = "run/tmp_{}_{}_{}-{}/".format(args.dataset, args.beta, run_time,origin)
    result_path = "run/result_{}_{}_{}-{}/".format(args.dataset, args.beta, run_time,origin)
    try:
        os.makedirs(temp_path)
        os.makedirs(result_path)
    except OSError as e:
        print(e)
    return temp_path, result_path


def parse_args():
    parse = argparse.ArgumentParser(description="GFUC")
    parse.add_argument('--dataset', default='Gowalla', type=str)
    parse.add_argument('--beta', default=0.7, type=float)
    parse.add_argument('--gnn_w', default=0.01, type=float) # 图神经网络在模块中的权重，gowalla(0.01)有效，
    parse.add_argument('--batch_size',default=4096,type=int)
    parse.add_argument('--embed_size', default=64, type=int)
    parse.add_argument('--lr', default=0.001, type=float)
    parse.add_argument('--decay', default=1e-4, type=float)
    parse.add_argument('--layers', default=3, type=int)
    parse.add_argument('--cores', default=4, type=int)
    parse.add_argument('--cuda', default="cuda:0", type=str)
    parse.add_argument('--num_epoch', default=1000, type=int)
    parse.add_argument('--pfm_iters', default=30, type=int)
    parse.add_argument('--find_pfm_iters', default=False, type=bool)
    parse.add_argument('--wk_dmax', default=15, type=int)
    parse.add_argument('--ed_dmax', default=15, type=int)
    parse.add_argument('--tamf_iters', default=90, type=int)
    parse.add_argument('--find_tamf_iters', default=False, type=bool)
    return parse.parse_args()


def get_result(scores,actual):
    predict_result = list(reversed(scores.argsort()))
    predict_max_topk = predict_result[:top_k]
    predict_metrics = get_metrics(actual, predict_max_topk)
    return predict_metrics


def json_file_write(json_var,path):
    with open(path, 'w') as f:
        json.dump(json_var,f)
    print("文件写入完毕")


def out_write(cnt,file_path,metrics_arr):
    str_reslut = '\t'.join(str(i) for i in metrics_arr.tolist())
    file_path.write(str_reslut + "\n")
    print('\t'.join([str(cnt), str_reslut]))

