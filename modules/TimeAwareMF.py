# -*- coding: utf-8
import time
import numpy as np
import scipy.sparse as sparse


class TimeAwareMF(object):
    def __init__(self, K, Lambda, alpha, beta, T):
        self.K = K
        self.T = T
        self.Lambda = Lambda
        self.alpha = alpha
        self.beta = beta
        self.U = None
        self.L = None
        self.LT = None

    def save_model(self, path):
        ctime = time.time()
        for i in range(self.T):
            np.save(path + "U" + str(i), self.U[i])
        np.save(path + "L", self.L)
        print("保存时间感知模块-隐矩阵U和L完毕，用时{}s:".format(time.time() - ctime))

    def load_model(self, path):
        ctime = time.time()
        self.U = [np.load(path + "U%d.npy" % i) for i in range(self.T)]
        self.L = np.load(path + "L.npy")
        self.LT = self.L.T
        print("重新加载时间感知模块-隐矩阵：U和L完毕，用时{}s:".format(time.time() - ctime))

    def get_phi(self, C, i, t):
        t_1 = (t - 1) if not t == 0 else (self.T - 1)
        norm_t = np.linalg.norm(C[t][i, :].toarray(), 'fro') # 求向量二范数=平方和开根
        norm_t_1 = np.linalg.norm(C[t_1][i, :].toarray(), 'fro')
        if norm_t == 0 or norm_t_1 == 0:
            return 0.0
        return C[t][i, :].dot(C[t_1][i, :].T)[0, 0] / norm_t / norm_t_1  # 不知道含义， 向量（用户）相似性？

    def train(self, sparse_check_in_matrices, temp_dir, max_iters=100, load_sigma=False, find_max_iters=False):
        Lambda = self.Lambda  # 初始Lambda=1.0
        alpha = self.alpha    # alpha=2.0
        beta = self.beta      # beta=2.0
        T = self.T            # T=24
        K = self.K            # K=100

        C = sparse_check_in_matrices
        M, N = sparse_check_in_matrices[0].shape # 用户数和POIs数目
        ctime = time.time()
        if load_sigma: # 初始sigma等于False
            sigma = np.load(temp_dir + "sigma.npy")
            print("加载时间感知模块 sigma完毕，用时{}".format(time.time() - ctime))
        else:
            print("初始化时间感知模型sigma...")
            sigma = [np.zeros(M) for _ in range(T)]  # 24*1（1=用户数）
            for t in range(T): # 遍历24小时 0-23
                C[t] = C[t].tocsr()
                for i in range(M): # 遍历用户 u0-um
                    sigma[t][i] = self.get_phi(C, i, t) # 计算0时和23时，1时和0时，2时和1时，...,23和22时用户习惯相似性
            sigma = [sparse.dia_matrix(sigma_t) for sigma_t in sigma] # dia_matrix具有对角存储的稀疏矩阵
            print("时间感知模型sigma初始化完毕，用时{}s".format(time.time() - ctime))
            np.save(temp_dir + "/sigma", sigma)

        U = [np.random.rand(M, K) for _ in range(T)]
        L = np.random.rand(N, K)

        C = [Ct.tocoo() for Ct in C]
        entry_index = [zip(C[t].row, C[t].col) for t in range(T)]

        C_est = [Ct for Ct in C]
        C = [Ct.tocsr() for Ct in C]
        iters, last_loss = 1, float('Inf')
        while iters > 0:
            for t in range(T):
                C_est[t] = C_est[t].todok()
                for i, j in entry_index[t]:
                    C_est[t][i, j] = U[t][i].dot(L[j])
                C_est[t] = C_est[t].tocsr()

            for t in range(T):  # 不断更新U
                t_1 = (t - 1) if not t == 0 else (self.T - 1)
                numerator = C[t] * L + Lambda * sigma[t] * U[t_1]
                denominator = np.maximum(1e-6, C_est[t] * L + Lambda * sigma[t] * U[t_1] + alpha * U[t_1])
                U[t] *= np.sqrt(1.0 * numerator / denominator)

            numerator = np.sum([C[t].T * U[t] for t in range(T)], axis=0)
            denominator = np.maximum(1e-6, np.sum([C_est[t].T * U[t]], axis=0) + beta * L)
            L *= np.sqrt(1.0 * numerator / denominator)

            error = 0.0
            for t in range(T):
                C_est_dok = C_est[t].todok()
                C_dok = C[t].todok()
                for i, j in entry_index[t]:
                    error += (C_est_dok[i, j] - C_dok[i, j]) * (C_est_dok[i, j] - C_dok[i, j])
            print('Iteration:', iters, error)
            if find_max_iters and last_loss < error and iters > 100:
                print('done:,iters = {}'.format(iters))
                break
            if find_max_iters is False and iters > max_iters:
                print('迭代次数已完毕:,iters = {}'.format(iters))
                break
            last_loss = error
            iters += 1
        self.U, self.L = U, L
        self.LT = L.T

    def predict(self, i, j):
        return np.sum([self.U[t][i].dot(self.L[j]) for t in range(self.T)])
