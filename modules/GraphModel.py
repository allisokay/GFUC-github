# -*- coding: utf-8
"""
@author：67543
@date：  2022/3/30
@contact：675435108@qq.com
"""
import random
import torch
import time
from torch import nn,optim
from torch.nn import Module,Embedding
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from metrics import *


class TrainLoader(Dataset):
    def __init__(self,train_set,train_U2I,n_items):
        self.train_set = train_set
        self.train_U2I = train_U2I
        self.all_items = list(range(0,n_items))

    def __getitem__(self, index):
        user = self.train_set[index][0]
        pos = self.train_set[index][1]
        neg = random.choice(self.all_items)
        while neg in self.train_U2I[user]:
            neg = random.choice(self.all_items)
        return [user,pos,neg]

    def __len__(self):
        return len(self.train_set)


class Graph():
    def __init__(self,n_users, n_items, train_U2I,training_matrix2):
        self.n_users = n_users
        self.n_items = n_items
        self.train_U2I = train_U2I
        self.n_nodes = n_users + n_items
        self.training_matrix2 = training_matrix2

    def to_edge(self):
        train_U,train_I = [],[]
        for u,item in self.train_U2I.items():
            train_U.extend([u]*len(item))
            train_I.extend(item)
        train_U,train_I = np.array(train_U),np.array(train_I)
        # 开始修改权重
        user_poi_weight,pois_user_weight = np.ones_like(train_U).tolist(),np.ones_like(train_I).tolist()
        users_pois = zip(train_U, train_I)
        for i, (uid, lid) in enumerate(users_pois):
            freq = self.training_matrix2[uid, lid]
            user_poi_weight[i] = freq if freq > 0 else user_poi_weight[i]
            pois_user_weight[i] = freq if freq > 0 else pois_user_weight[i]
        # 修改权重结束
        row = np.concatenate([train_U,train_I+self.n_users])
        col = np.concatenate([train_I+self.n_users,train_U])
        # edge_weight = np.ones_like(row).tolist() # 这里可以修改一下
        edge_weight = np.concatenate([user_poi_weight,pois_user_weight])
        edge_index = np.stack([row,col]).tolist()
        return edge_index,edge_weight

    def norm(self,edge_index,edge_weight):
        row,col = edge_index[0],edge_index[1]
        deg = torch.zeros(self.n_nodes,dtype=torch.float32)
        deg = deg.scatter_add(0,col,edge_weight) # 得到用户访问过的POI总数以及POI被访问的用户总数，这里忽略txt中的访问次数，即都是1
        # 上一步代码，权重边要改一下，我先把代码重构玩吧******
        deg_inv_sqrt = deg.pow(-0.5) # 数据规范处理，将值约束到0-1之间
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'),0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index,edge_weight

    def generate(self):
        """
        to_edge()返回两个值：
          edge_index:shape = 2*2n,dim = 2,n表示所有用户交互过的item总数，
            values = [[uid,uid,...,n,lid+用户数,lid+用户数,...,2n],[lid+用户数,lid+用户数,...,n,uid,uid...,2n] ]
          edge_weight:shape = 2n,dim = 1,n表示所有用户交互过的item总数,values = 1
        :return:
        """
        edge_index,edge_weight = self.to_edge()
        edge_index,edge_weight = torch.tensor(edge_index,dtype=torch.long),torch.tensor(edge_weight,dtype=torch.float32)
        loop_index,loop_weight = torch.arange(0,self.n_nodes,dtype=torch.long),torch.ones(self.n_nodes,dtype=torch.float32)
        loop_index = loop_index.unsqueeze(0).repeat(2,1)  # 增加一个维度，复制一列
        # loop_index:2*m,m为用户数+Item数，values = 0,1,2,3，...,m
        edge_weight = torch.cat([edge_weight,loop_weight],dim=-1)  # cat后变成2n+m个1
        # eige_weight:
        edge_index  = torch.cat([edge_index,loop_index],dim=-1)
        # cat后变成shape=[2*(2n+m)]个1,dim = 2,后面的[2,m]是loop_index[[0,..m],[0,..m]]
        #标准化：更新权重
        edge_index,edge_weight = self.norm(edge_index,edge_weight)
        print ("        图谱渲染完毕.")
        return torch.sparse.FloatTensor(edge_index,edge_weight,torch.Size([self.n_nodes,self.n_nodes]))


class GCN(Module):
    def __init__(self,n_users,n_items,adj,args):
        super(GCN,self).__init__()
        self.adj = adj
        self.n_users,self.n_items = n_users,n_items,
        self.embed_size,self.batch_size  = args.embed_size,args.batch_size
        self.decay,self.layers = args.decay,args.layers
        user_embed_weight = nn.init.normal_(torch.empty(n_users, args.embed_size), std=0.01)
        item_embed_weight = nn.init.normal_(torch.empty(n_items, args.embed_size), std=0.01)
        self.user_embeddings = Embedding(self.n_users,self.embed_size,_weight=user_embed_weight)
        self.item_embeddings = Embedding(self.n_items, self.embed_size, _weight=item_embed_weight)
        print ("    图神经网洛权重-embedding已引入，等待训练...")

    def propagate(self):
        x = torch.cat([self.user_embeddings.weight,self.item_embeddings.weight],dim=0)
        all_embed = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(self.adj,x)
            all_embed += [x]
        embeddings = torch.stack(all_embed,dim=1)
        f_embed = torch.mean(embeddings,dim=1)
        user_embed,item_embed = torch.split(f_embed,[self.n_users,self.n_items])
        return user_embed,item_embed

    def calculate_loss(self,user_embed,pItem_embed,nItem_embed):
        pos_score = torch.sum(user_embed * pItem_embed,dim=1)
        neg_score = torch.sum(user_embed * nItem_embed,dim=1)
        mf_loss = torch.mean(F.softplus(neg_score-pos_score))
        reg_loss = (1/2) * (user_embed.norm(2).pow(2) +
                            pItem_embed.norm(2).pow(2)+
                            nItem_embed.norm(2).pow(2))\
                   /user_embed.shape[0]*self.decay
        loss = mf_loss + reg_loss
        return loss,mf_loss,reg_loss
