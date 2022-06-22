# -*- coding: utf-8
"""
@author：67543
@date：  2022/3/18
@contact：675435108@qq.com
"""
import scipy.sparse as sparse
from collections import defaultdict
import numpy as np
import time
import pickle
import pandas as pd
from tools import format_print


class Data:
    def __init__(self, data_name):
        self.data_name,data_dir = data_name,"Dataset/" + data_name + "/"
        self.check_in_file = data_dir + data_name + "_checkins.txt"
        self.train_file = data_dir + data_name + "_train.txt"
        self.tune_file = data_dir + data_name + "_tune.txt"
        self.test_file = data_dir + data_name + "_test.txt"
        self.poi_file = data_dir + data_name + "_poi_coos.txt"
        self.social_file = data_dir + data_name + "_social_relations_70.txt"
        self.num_file = data_dir + data_name +"_data_size.txt"
        self.training_tuples,user_num,poi_num = None, None,None

    def load_train_tune_data(self,use_tune=True):
        sparse_training_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        training_matrix = np.zeros((self.user_num, self.poi_num))
        training_matrix2 = np.zeros((self.user_num, self.poi_num))
        training_tuples = set()
        Users, Items, User2Itmes = [], [], {}
        def load_data(data):
            for eachline in data:
                uid, lid, freq = eachline.strip().split()
                uid, lid, freq = int(uid), int(lid), int(freq)
                Users.append(uid)
                Items.append(lid)
                User2Itmes.setdefault(uid,[])
                User2Itmes[uid].append(lid)
                sparse_training_matrix[uid, lid] = freq
                training_matrix[uid, lid], training_matrix2[uid, lid] = 1.0, freq
                training_tuples.add((uid, lid))
        load_data(open(self.train_file, 'r').readlines())
        if use_tune:
            load_data(open(self.tune_file, 'r').readlines())
        train_set = np.vstack([Users,Items]).T.tolist()
        self.training_tuples = training_tuples
        format_print("训练文本和验证文本加载完成", " ", 6)
        return sparse_training_matrix, training_tuples, training_matrix, training_matrix2,train_set,User2Itmes

    def load_checkin_data(self):
        check_in_data = open(self.check_in_file, 'r').readlines()
        training_tuples_with_day = defaultdict(int)
        training_tuples_with_time = defaultdict(int)
        for eachline in check_in_data:
            uid, lid, ctime = eachline.strip().split()
            uid, lid, ctime = int(uid), int(lid), float(ctime)
            if (uid, lid) in self.training_tuples:
                hour = time.gmtime(ctime).tm_hour
                training_tuples_with_time[(hour, uid, lid)] += 1.0
                if 8 <= hour < 18:
                    hour = 0  # 工作时间
                elif hour >= 18 or hour < 8:
                    hour = 1  # 闲暇时间
                training_tuples_with_day[(hour, uid, lid)] += 1.0
        # 默认时间被分为24小时.
        sparse_training_matrices = [sparse.dok_matrix((self.user_num, self.poi_num)) for _ in range(24)]
        for (hour, uid, lid), freq in training_tuples_with_time.items():
            sparse_training_matrices[hour][uid, lid] = 1.0 / (1.0 + 1.0 / freq)
        sparse_training_matrix_WT = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_training_matrix_LT = sparse.dok_matrix((self.user_num, self.poi_num))
        for (hour, uid, lid), freq in training_tuples_with_day.items():
            if hour == 0:
                sparse_training_matrix_WT[uid, lid] = freq
            elif hour == 1:
                sparse_training_matrix_LT[uid, lid] = freq
        format_print("签到文本加载完成，时间被分为24段、工作日和休息日"," ",6)
        return sparse_training_matrices, sparse_training_matrix_WT, sparse_training_matrix_LT

    def load_ground_truth(self):
        ground_truth = defaultdict(set)
        User2Itmes =  {}
        truth_data = open(self.test_file, 'r').readlines()
        for eachline in truth_data:
            uid, lid, _ = eachline.strip().split()
            uid, lid = int(uid), int(lid)
            ground_truth[uid].add(lid)
            User2Itmes.setdefault(uid, [])
            User2Itmes[uid].append(lid)
        format_print("测试文本加载完成"," ",6)
        return ground_truth,User2Itmes

    def load_poi_coos(self):
        poi_coos = {}
        poi_data = open(self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)
        format_print ("poi信息文本加载完成", " ", 6)
        return poi_coos

    def load_friend_data(self):
        social_data = open(self.social_file, 'r').readlines()
        social_matrix = np.zeros((self.user_num, self.user_num))
        for eachline in social_data:
            uid1, uid2, _, _, _, _, _ = eachline.strip().split()
            uid1, uid2 = int(uid1), int(uid2)
            social_matrix[uid1, uid2] = 1.0
            social_matrix[uid2, uid1] = 1.0
        format_print ("社交文本加载完成", " ", 6)
        return social_matrix

    def get_number(self):
        with open(self.num_file, 'r') as f:
            user_num, poi_num = f.readlines()[0].strip('\n').split()
            self.user_num, self.poi_num = int(user_num), int(poi_num)

    def generate_fine_grain_data(self):
        training_data_workday = {'day': sparse.dok_matrix((self.user_num, self.poi_num)),'night': sparse.dok_matrix((self.user_num, self.poi_num))}
        training_data_weekend = training_data_workday.copy()
        print ("工作日和休息日时间切割.....")
        start_time = time.time()
        check_in_data = pd.read_csv(self.check_in_file, delimiter='\t', encoding='utf-8',names=['users', 'pois', 'timestamp'])
        for _, user_group in check_in_data.groupby(by='users'):
            checkin_data = user_group.sort_values(by='timestamp')
            for row_num, (uid, lid, ctime) in checkin_data.iterrows():
                if (uid, lid) in self.training_tuples:
                    w_d, hour = time.gmtime(ctime).tm_wday, time.gmtime(ctime).tm_hour
                    if w_d == 5 or w_d == 6:  # 周六、周天休息日 ，or w_d == 6
                        if 8 <= hour < 18: training_data_weekend['day'][uid, lid] += 1.0
                        else:training_data_weekend['night'][uid, lid] += 1.0
                    else:
                        if 8 <= hour < 18:training_data_workday['day'][uid, lid] += 1.0
                        else:training_data_workday['night'][uid, lid] += 1.0
        print ("工作日和休息日时间切割完成，用时{}s".format(time.time() - start_time))
        return training_data_workday, training_data_weekend


if __name__ == '__main__':
    dataset = ['Gowalla','Yelp']
    for data_name in dataset:
        format_print("开始处理"+data_name+"数据集...")
        s_t,data = time.time(),Data(data_name)
        data.get_number()
        sparse_training_matrix, training_tuples, training_matrix, training_matrix2,training_set,training_U2I = data.load_train_tune_data(use_tune=False)
        sparse_training_matrices, sparse_training_matrix_WT, sparse_training_matrix_LT = data.load_checkin_data()
        training_data_workday, training_data_weekend = data.generate_fine_grain_data()
        ground_truth,test_U2I = data.load_ground_truth()
        poi_coos = data.load_poi_coos()
        social_matrix = data.load_friend_data()
        with open("processed/"+data_name+"/"+"graph.pkl","w") as f:
            pickle.dump(training_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(training_U2I, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_U2I, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(training_matrix2, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(social_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump([data.user_num,data.poi_num],f, pickle.HIGHEST_PROTOCOL)
        with open("processed/"+data_name+"/"+"fuc.pkl",'w') as f:
            pickle.dump(sparse_training_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(training_tuples, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(training_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(training_matrix2, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(sparse_training_matrices, f, pickle.HIGHEST_PROTOCOL)
            # pickle.dump(sparse_training_matrix_WT, f, pickle.HIGHEST_PROTOCOL)
            # pickle.dump(sparse_training_matrix_LT, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(training_data_workday, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(training_data_weekend, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(ground_truth, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(poi_coos, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(social_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump([data.user_num,data.poi_num],f, pickle.HIGHEST_PROTOCOL)
        format_print(data_name+"数据集处理完毕，用时{}s".format(time.time()-s_t))
