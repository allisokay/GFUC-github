# -*- coding: utf-8
"""
@author：67543
@date：  2022/4/18
@contact：675435108@qq.com
"""
import pickle

from util import *
from tools import *


def cal_metrics((cnt, uid)):
    metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
              "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}"
    FUC_score = [PFM.predict(uid, lid) *
                 ((MGMWK['day'].predict(uid, lid) + MGMWK['night'].predict(uid, lid)) +
                  (MGMED['day'].predict(uid, lid) + MGMED['night'].predict(uid, lid))) *
                 TAMF.predict(uid, lid) * LFBCA.predict(uid, lid) if (uid, lid) not in training_tuples else -1 for
                 lid in all_lids]
    FUC_score, GCN_score = np.array(FUC_score), scores[uid]
    unite_score = (1 - args.gnn_w) * FUC_score + args.gnn_w * GCN_score
    predict_topk = list(reversed(unite_score.argsort()))[:top_k]
    actual = ground_truth[uid]
    result = get_metrics(actual, predict_topk)
    metrics = metrics.format(*result)
    print(cnt, uid, metrics)
    with open(result_path + "/final_result.txt", 'a') as f:
        str_reslut = '\t'.join(str(i) for i in result.tolist())
        f.write(str_reslut + '\n')
    return result.tolist()


def predict():
    test_paras = [(i, uid) for i, uid in enumerate(test_uids)]
    pool = mp.Pool()
    results = pool.map(cal_metrics, test_paras)
    pool.close()
    pool.join()
    results = np.array(results)
    final_gnn_resuluts = np.mean(results, axis=0)
    metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
              "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}".format(*final_gnn_resuluts)
    with open(result_path + "/final_result.txt", 'a') as f:
        f.write(metrics)
    print(metrics)


if __name__ == '__main__':
    """
      1.基本配置
   """
    args = parse_args()
    temp_path, result_path = get_config(args, "uniteModel")
    data_name = args.dataset
    FUC_load_path = "./uniteModels/" + data_name + "-FUC/"
    GCN_load_path = "./uniteModels/" + data_name + "-GCN/"
    """
   2.加载测试数据
   """
    with open("processed/" + data_name + "/fuc.pkl", 'r') as f:
        _, training_tuples, _, _, _ = pickle.load(f), pickle.load(f), pickle.load(f), pickle.load(f), pickle.load(f)
        training_data_workday, training_data_weekend = pickle.load(f), pickle.load(f)
        ground_truth, poi_coos = pickle.load(f), pickle.load(f)
        social_matrix = pickle.load(f)
        user_num, poi_num = pickle.load(f)
    all_uids, all_lids, test_uids = list(range(user_num)), list(range(poi_num)), ground_truth.keys()
    # all_uids, all_lids, test_uids = list(range(user_num)), list(range(poi_num)), test_U2I.keys()
    np.random.shuffle(test_uids)
    """
   3.初始化模块
   """
    PFM = PoissonFactorModel(K=30, alpha=20.0, beta=0.2)
    MGMWK = {'day': None, 'night': None}
    MGMED = {'day': None, 'night': None}
    for key in MGMWK.keys():
        MGMWK[key] = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=args.wk_dmax)
        MGMED[key] = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=args.ed_dmax)
    TAMF = TimeAwareMF(K=100, Lambda=1.0, beta=2.0, alpha=2.0, T=24)
    LFBCA = LocationFriendshipBookmarkColoringAlgorithm(alpha=0.85, beta=float(args.beta), epsilon=0.001)
    """
   4.模块参数装配
   """
    user_embed, item_embed = torch.load(GCN_load_path + data_name + "-graph_model.pth")
    scores = np.matmul(user_embed, item_embed.T)
    PFM.load_model(FUC_load_path)
    for key in MGMWK.keys():
        MGMWK[key].multi_center_discovering(training_data_workday[key], poi_coos)  # 工作日的多活动中心
        MGMED[key].multi_center_discovering(training_data_weekend[key], poi_coos)  # 周末的多活动中心
    TAMF.load_model(FUC_load_path)
    LFBCA.load_model(FUC_load_path)
    """
      5.模块预测
    """
    predict()
