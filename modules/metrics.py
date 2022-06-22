# -*- coding: utf-8
import math
import numpy as np
import multiprocessing


def evaluate(user_embed,item_embed,train_U2I,testU2I,args):
    scores = np.matmul(user_embed,item_embed.T)
    topks = [5,10,15,20]
    test_user_set = list(testU2I.keys())
    perf_info = np.zeros((len(test_user_set),3*len(topks)),dtype=np.float32)
    test_paras = zip(test_user_set,)
    pool = multiprocessing.Pool(processes=args.cores,initializer=init,initargs=(scores,train_U2I,testU2I,topks))
    all_perf = pool.map(test_one_perf,test_paras)
    pool.close()
    pool.join()
    for i,one_perf in enumerate(all_perf):
        perf_info[i] = one_perf
    perf_info = np.mean(perf_info,axis=0)
    return perf_info


def test_one_perf(X):
    uid,max_topk = X[0],max(topks)
    score = scores[uid]
    uid_train_pos_itmes = list(train_user_item[uid])
    uid_test_pos_items = list(test_user_item[uid])
    score[uid_train_pos_itmes] = -np.inf
    indices = np.argpartition(score, -max_topk)[-max_topk:]
    score_indices = indices[np.argsort(-score[indices])]
    topks_eval = np.zeros(3 * len(topks), dtype=np.float32)
    for i, topk in enumerate(topks):
        topks_eval[i * 3 + 0] = precisionk(uid_test_pos_items,score_indices[:topk])
        topks_eval[i * 3 + 1] = recallk(uid_test_pos_items,score_indices[:topk])
        topks_eval[i * 3 + 2] = ndcgk(uid_test_pos_items,score_indices[:topk] )
    return topks_eval


def init(_scores,_train_U2I,_test_U2I,_topks):
    global scores, train_user_item, test_user_item, topks
    scores = _scores
    train_user_item = _train_U2I
    test_user_item = _test_U2I
    topks =_topks


def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)


def ndcgk(actual, predicted):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg


def get_metrics(actual,predict_list):
    topks = [5, 10, 15, 20]
    topks_eval = np.zeros(3 * len(topks), dtype=np.float32)
    for i, topk in enumerate(topks):
        topks_eval[i * 3 + 0] = precisionk(actual,predict_list[:topk])
        topks_eval[i * 3 + 1] = recallk(actual,predict_list[:topk])
        topks_eval[i * 3 + 2] = ndcgk(actual,predict_list[:topk])
    return topks_eval