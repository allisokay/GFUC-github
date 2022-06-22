# -*- coding: utf-8
"""
@author：67543
@date：  2022/4/18
@contact：675435108@qq.com
"""
from tools import *
import pickle


def train():
    f = open(result_path + args.dataset + "-graph_run.log", 'a')
    for epoch in range(args.num_epoch):
        start_time = time.time()
        gcn.train()
        all_loss, all_mf_loss, all_reg_loss = 0.0, 0.0, 0.0
        for data in loader:
            users = data[0].type(torch.long).to(device)
            pItems = data[1].type(torch.long).to(device),
            nItems = data[2].type(torch.long).to(device)
            user_embed, item_embed = gcn.propagate()
            user_embed, pItem_embed, nItem_embed = user_embed[users], item_embed[pItems], item_embed[nItems]
            loss, mf_loss, reg_loss = gcn.calculate_loss(user_embed, pItem_embed, nItem_embed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            all_mf_loss += mf_loss.item()
            all_reg_loss += reg_loss.item()
        mean_loss, mean_mf_loss, mean_reg_loss = all_loss / len(loader), all_mf_loss / len(loader), all_reg_loss / len(
            loader)
        one_epoch_time = time.time() - start_time
        loss_str = "epoch:{},loss:[{}]=mf:[{}]+reg:[{}],用时{}s".format(epoch + 1, mean_loss, mean_mf_loss, mean_reg_loss,
                                                                      one_epoch_time)
        print(loss_str + "\n")
        f.write(loss_str)
        if (epoch) % 100 == 0:
            gcn.eval()
            # 测试
            with torch.no_grad():
                user_embed, item_embed = gcn.propagate()
                user_embed, item_embed = user_embed.cpu().detach().numpy(), item_embed.cpu().detach().numpy()
                start_time = time.time()
                perf_info = evaluate(user_embed, item_embed,train_U2I, test_U2I, args)
                metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
                "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}".format(*perf_info)
                format_print("第{}轮metrics计算完成：用时{}s".format(epoch+1,time.time() - start_time)," ",6)
                print(metrics)
                f.write(metrics+"\n")
                graph_model_path = temp_path + args.dataset + "-graph_model.pth"
                torch.save((user_embed, item_embed), graph_model_path)
                print("图模型保存")
    f.close()


def predict():
    scores = np.matmul(user_embed,item_embed.T)
    results = []
    with open(result_path + "/gnn_result.txt", 'a') as f:
        for uid in test_uids:
            score = scores[uid]
            score[train_U2I[uid]] = -np.inf  # 将返回过的lid的概率值设为最小
            predict_result = list(reversed(score.argsort()))
            predict_topk, actual = predict_result[:top_k], test_U2I[uid]
            result = get_metrics(actual, predict_topk)
            results.append(result.tolist())
            str_reslut = '\t'.join(str(i) for i in result.tolist())
            print('\t'.join([str(uid), str_reslut]))
            f.write(str_reslut+"\n" )
        results = np.array(results)
        final_gnn_resuluts = np.mean(results, axis=0)
        metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
                  "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}".format(*final_gnn_resuluts)
        print(metrics)
        f.write(metrics)


if __name__ == '__main__':
    """
    1.基本配置
    """
    args = parse_args()
    temp_path, result_path = get_config(args,"GN")
    device = torch.device(args.cuda)
    """
    2.加载数据
    """
    data_name = args.dataset
    with open("processed/" + data_name + "/graph.pkl", 'r') as f:
        train_set = pickle.load(f)
        train_U2I = pickle.load(f)
        test_U2I = pickle.load(f)
        training_matrix2 = pickle.load(f)
        social_matrix = pickle.load(f)
        user_num,poi_num = pickle.load(f)
    # 预测数据
    test_uids =  test_U2I.keys()
    """
    3.初始化模块
    """
    gnn_dataloader = TrainLoader(train_set, train_U2I, poi_num)
    loader = DataLoader(gnn_dataloader, args.batch_size, num_workers=args.cores)
    start_time = time.time()  # 记录构建图谱起始时间
    graph = Graph(user_num, poi_num, train_U2I,training_matrix2)
    adj = graph.generate().to(device)
    gcn = GCN(user_num,poi_num, adj, args)  # gowalla数据集运行到这需要gpu显存575MB
    optimizer = optim.Adam(gcn.parameters(), lr=args.lr)
    gcn = gcn.to(device)
    format_print("图谱构建完成，用时{}s".format(time.time() - start_time))
    """
    4.模块训练
    """
    start_time = time.time()  # 记录图模型训练时间
    format_print("图神经网络训练开始...")
    train()
    format_print("图神经网络训练结束，共用时{}s".format(time.time()-start_time))
    """
    5.最终模型保存和预测
    """
    user_embed, item_embed = gcn.propagate()
    user_embed, item_embed = user_embed.cpu().detach().numpy(), item_embed.cpu().detach().numpy()
    graph_model_path = temp_path + args.dataset + "-graph_model.pth"
    torch.save((user_embed, item_embed), graph_model_path)
    format_print("最终图模型保存","",8)
    predict()