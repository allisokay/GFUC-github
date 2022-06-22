# -*- coding: utf-8
"""
@author：67543
@date：  2022/4/18
@contact：675435108@qq.com
"""
import pickle
import time

from tools import *


def train():
    PFM.train(sparse_training_matrix, max_iters=args.pfm_iters, learning_rate=1e-4,find_max_iters=args.find_pfm_iters)
    PFM.save_model(temp_path)
    PFM.load_model(temp_path)
    ctime = time.time()
    # MGMWT.multi_center_discovering(sparse_training_matrix_WT, poi_coos)  # 工作日的多活动中心
    # MGMLT.multi_center_discovering(sparse_training_matrix_LT, poi_coos)  # 周末的多活动中心
    for key in MGMWK.keys():
        MGMWK[key].multi_center_discovering(training_data_workday[key], poi_coos)  # 工作日的多活动中心
        MGMED[key].multi_center_discovering(training_data_weekend[key], poi_coos)  # 周末的多活动中心
    print("多活动中心加载完毕，用时" + str(time.time() - ctime) + "s")
    TAMF.train(sparse_training_matrices, temp_path, max_iters=args.tamf_iters, load_sigma=False,find_max_iters=args.find_tamf_iters)
    TAMF.save_model(temp_path)
    TAMF.load_model(temp_path)
    LFBCA.precompute_rec_scores(training_matrix2, social_matrix)
    LFBCA.save_result(temp_path)
    LFBCA.load_model(temp_path)


def cal_metrics((cnt, uid)):
    metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
              "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}"
    overall_scores = [PFM.predict(uid, lid) *
                      ((MGMWK['day'].predict(uid, lid) + MGMWK['night'].predict(uid, lid)) +
                       (MGMED['day'].predict(uid, lid) + MGMED['night'].predict(uid, lid))) *
                      TAMF.predict(uid, lid) * LFBCA.predict(uid, lid) if (uid, lid) not in training_tuples else -1 for
                      lid in all_lids]
    # overall_scores = [PFM.predict(uid, lid) * (MGMWT.predict(uid, lid) + MGMLT.predict(uid, lid))
    #                   * TAMF.predict(uid, lid) * LFBCA.predict(uid, lid)
    #                   if (uid, lid) not in training_tuples else -1
    #                   for lid in all_lids]
    overall_scores = np.array(overall_scores)
    predict_topk = list(reversed(overall_scores.argsort()))[:top_k]
    actual = ground_truth[uid]
    result = get_metrics(actual, predict_topk)
    metrics = metrics.format(*result)
    print(cnt, uid, metrics)
    with open(result_path + "/FUC_result.txt", 'a') as f:
        str_reslut = '\t'.join(str(i) for i in result.tolist())
        f.write(str_reslut + '\n')
    return result.tolist()


def predict():
    test_paras = [(i, uid) for i, uid in enumerate(test_uids)]
    pool = mp.Pool()
    results = pool.map(cal_metrics, test_paras)
    results = np.array(results)
    final_gnn_resuluts = np.mean(results, axis=0)
    metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
              "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}".format(*final_gnn_resuluts)
    with open(result_path + "/FUC_result.txt", 'a') as f:
        f.write(metrics)
    print(metrics)


if __name__ == '__main__':
    """
       1.基本配置

    """
    args = parse_args()
    temp_path, result_path = get_config(args, "FUC")
    format_print("完成", " ", 6)
    """
    2.加载数据
    """
    start_time = time.time()
    format_print("加载数据...")
    data_name = args.dataset
    with open("processed/" + data_name + "/fuc.pkl", 'r') as f:
        sparse_training_matrix, training_tuples = pickle.load(f), pickle.load(f)
        training_matrix, training_matrix2 = pickle.load(f), pickle.load(f)
        sparse_training_matrices = pickle.load(f)
        # sparse_training_matrix_WT = pickle.load(f)
        # sparse_training_matrix_LT = pickle.load(f)
        training_data_workday = pickle.load(f)
        training_data_weekend = pickle.load(f)
        ground_truth, poi_coos = pickle.load(f), pickle.load(f),
        social_matrix = pickle.load(f)
        user_num, poi_num = pickle.load(f)
    all_uids, all_lids, test_uids = list(range(user_num)), list(range(poi_num)), ground_truth.keys()
    np.random.shuffle(test_uids)
    format_print("完成，用时{}s".format(time.time() - start_time), " ", 6)
    """
    3.初始化模块
    """
    PFM = PoissonFactorModel(K=30, alpha=20.0, beta=0.2)
    # MGMWT = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=15)
    # MGMLT = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=15)
    MGMWK = {'day': None, 'night': None}
    MGMED = {'day': None, 'night': None}
    for key in MGMWK.keys():
        MGMWK[key] = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=args.wk_dmax)
        MGMED[key] = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=args.ed_dmax)
    TAMF = TimeAwareMF(K=100, Lambda=1.0, beta=2.0, alpha=2.0, T=24)
    LFBCA = LocationFriendshipBookmarkColoringAlgorithm(alpha=0.85, beta=float(args.beta), epsilon=0.001)
    """
    4.模块训练
    """
    start_time = time.time()  # 记录图模型训练时间
    format_print("FUC训练开始...")
    train()
    format_print("FUC训练结束，共用时{}s".format(time.time() - start_time))
    """
    5.模块预测
    """
    predict()