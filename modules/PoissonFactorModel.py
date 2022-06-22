# -*- coding: utf-8
import time
import math
import numpy as np


class PoissonFactorModel(object):
    def __init__(self, K=30, alpha=20.0, beta=0.2):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.U, self.L = None, None

    def save_model(self, path):
        ctime = time.time()
        np.save(path + "PFM_U", self.U)
        np.save(path + "PFM_L", self.L)
        print("泊松因子模块隐矩阵：U和L保存完毕,用时{}s".format(time.time() - ctime))

    def load_model(self, path):
        ctime = time.time()
        self.U = np.load(path + "PFM_U.npy")
        self.L = np.load(path + "PFM_L.npy")
        print("泊松因子模块重新加载完毕,用时{}s".format(time.time() - ctime))

    def train(self, sparse_check_in_matrix, max_iters=50, learning_rate=1e-4,find_max_iters=False):
        ctime = time.time()
        print("开始训练泊松因子模块...")
        alpha, beta, K = self.alpha, self.beta, self.K
        F = sparse_check_in_matrix
        M, N = sparse_check_in_matrix.shape
        U = 0.5 * np.sqrt(np.random.gamma(alpha, beta, (M, K))) / K  # alpha、beta，K如何确定的？
        L = 0.5 * np.sqrt(np.random.gamma(alpha, beta, (N, K))) / K

        F = F.tocoo()
        entry_index = list(zip(F.row, F.col))

        F = F.tocsr()
        F_dok = F.todok()  # 转了半天又转回来了F_dok=sparse_check_in_matrix

        tau = 10
        last_loss = float('Inf')
        iters = 1
        while iters > 0:
            F_Y = F_dok.copy()
            for i, j in entry_index:
                F_Y[i, j] = 1.0 * F_dok[i, j] / U[i].dot(L[j]) - 1
            F_Y = F_Y.tocsr()

            learning_rate_k = learning_rate * tau / (tau + iters)
            U += learning_rate_k * (F_Y.dot(L) + (alpha - 1) / U - 1 / beta)
            L += learning_rate_k * ((F_Y.T).dot(U) + (alpha - 1) / L - 1 / beta)

            loss = 0.0
            for i, j in entry_index:
                loss += (F_dok[i, j] - U[i].dot(L[j])) ** 2

            print('Iteration:', iters, 'loss:', loss)

            if find_max_iters and loss > last_loss and iters > 30:
                print("loss不再下降，此时iter={}".format(iters))
                break
            if find_max_iters is False and iters > max_iters:
                print("迭代次数完毕，此时iter={}".format(iters))
                break
            last_loss = loss
            iters += 1

        print("泊松因子模型训练完毕,用时{}s".format(time.time() - ctime))
        self.U, self.L = U, L

    def predict(self, uid, lid, sigmoid=False):
        if sigmoid:
            return 1.0 / (1 + math.exp(-self.U[uid].dot(self.L[lid])))
        return self.U[uid].dot(self.L[lid]) # 点乘相加后返回单值
