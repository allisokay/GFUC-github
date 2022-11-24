# -*- coding: utf-8
"""
@author：67543
@date：  2022/4/8
@contact：675435108@qq.com
"""
import math
import numpy as np
import multiprocessing as mp
import torch
import pickle
import random

def prac((cnt,uid)):
    print cnt,uid
    with open("prac.txt", 'a') as f:
        f.write(str([cnt,uid])+ '\n')
    return [cnt,uid]

if __name__ == '__main__':
     ids = 100
     uids = [i for i in range(ids)]
     np.random.shuffle(uids)
     test_paras = [(i,uid) for i,uid in enumerate(uids)]
     pool = mp.Pool()
     all_perf = pool.map(prac, test_paras)
     pool.close()
     pool.join()
     all_perf = np.array(all_perf)
     perf_info = np.mean(all_perf, axis=0)
     print(np.mean(uids))
     print perf_info
