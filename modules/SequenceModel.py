# -*- coding: utf-8

import time
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Embedding, Dense
import pandas as pd
import util
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from matplotlib import pyplot


def get_init_data(input_data,poi_num):
    truth_input_seq = input_data['vists_list']
    truth__user = input_data['users']
    turn_input, turn_user, input_lenth = [], [], 6
    for time_slot in truth_input_seq:
        visits_list, users = truth_input_seq[time_slot], truth__user[time_slot]
        users_visits = zip(users, visits_list)
        for uid, visit_list in users_visits:
            if len(visit_list) < input_lenth:
                while len(visit_list) < input_lenth:
                    visit_list.insert(0, poi_num)
                visit_list.insert(0, uid)
                turn_input.append(visit_list)
            elif len(visit_list) == input_lenth:
                visit_list.insert(0, uid)
                turn_input.append(visit_list)
            else:
                for i in range(len(visit_list) - (input_lenth - 1)):
                    temp_list = visit_list[i:i + input_lenth]
                    temp_list.insert(0, uid)
                    turn_input.append(temp_list)
        print ("    模型输入数据处理完毕")

        return turn_input


def process_data(turn_data):
     turn_data = turn_data.astype('float32')
     scaler = MinMaxScaler(feature_range=(0,1))
     scaled = scaler.fit_transform(turn_data)
     train_X,train_Y = scaled[:,:-1],scaled[:,-1]
     train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
     return train_X,train_Y,scaler


class SequenceModel():
    def __init__(self):
        self.user_num,self.poi_num = None,None
        model = Sequential([
            LSTM(150,input_shape=(1,6)),
            Dense(100),
            Dense(1)
        ])
        self.model = model

    def train(self,input_data,data,train_tuples,ground_truth):
        self.user_num,self.poi_num = data.user_num,data.poi_num
        turn_data = np.array(get_init_data(input_data, data.poi_num))
        train_X,train_Y,scaler = process_data(turn_data)
        self.model.compile(loss='mae', optimizer='adam')
        history = self.model.fit(train_X,train_Y,validation_split=0.2,verbose=1,nb_epoch=1,batch_size=64,shuffle=False)
        pyplot.plot(history.history['loss'],label='train')
        pyplot.plot(history.history['val_loss'], label='validation')
        pyplot.legend()
        pyplot.show()
        self.predict(scaler,data,train_tuples,ground_truth)

    def predict(self,scaler,data,training_tuples,ground_truth):
        all_uids, all_lids = list(range(data.user_num)), list(range(data.poi_num))
        poi = data.poi_num
        for cnt, uid in enumerate(all_uids):
            if uid in ground_truth:
                for lid in all_lids:
                    if (uid, lid) not in training_tuples:
                        test_X_raw = np.array([[uid,poi,poi,poi,poi,lid]])
                        test_X_min_max = scaler.fit_transform(test_X_raw)
                        test_X = test_X_min_max.reshape((test_X_min_max.shape[0],1,test_X_min_max.shape[1]))
                        _predict = self.model.predict(test_X)
                        predict = np.concatenate((_predict,test_X_min_max[:,1:]),axis=1)
                        predict = scaler.inverse_transform(predict)
                        print predict[0]
                        return 1


