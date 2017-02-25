# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:47:04 2017

@author: WangFeichi
for Weibo Predict 
method: only used means

"""

import pandas as pd
import numpy as np

# load training data
row_train_data = pd.read_table('../Weibo Data/Weibo Data/weibo_train_data.txt', \
names = ['uid','mid','time','forward_count','comment_count','like_count','content'], \
encoding = 'utf-8', \
iterator = True)
train_data = row_train_data.read()
train_data = train_data.drop('content',axis=1)
train_data = train_data.set_index(['uid','time','mid'])
user_mean = train_data.mean(axis = 0,level = 0)
user_mean = user_mean.apply(np.round)
del train_data


# load testing data
row_test_data = pd.read_table('../Weibo Data/Weibo Data/weibo_predict_data.txt', \
names = ['uid','mid','time','content'], \
encoding = 'utf-8', \
#chunksize = 1, \
iterator = True)
test_data = row_test_data.read()
test_data = test_data.drop('content',axis=1)

count_new_user = 0
with open('results/model_1_results.txt','w') as fw:
    for index in test_data.iterrows(): # 无法使用行索引，不可用
        t_uid = test_data.ix[index[0],'uid']
        t_mid = test_data.ix[index[0],'mid']
        if t_uid in user_mean.index:
            t_predict = str(int(user_mean.ix[t_uid,'forward_count'])) + ',' +  \
            str(int(user_mean.ix[t_uid,'comment_count'])) + ',' + \
            str(int(user_mean.ix[t_uid,'like_count']))
        else:
            t_predict = '0,0,0'
            count_new_user +=1
        fw.writelines(t_uid+'\t'+t_mid+'\t'+t_predict+'\n')

#predict_results = user_mean.ix[test_data['uid']]
#predict_results.to_csv('../Weibo Data/Weibo Data/results/only_mean.txt', \
#float_format = int, header = False, na_rep = 0, sep = '\t')
#
##predict_results = pd.DataFrame(columns = ['forward_count','comment_count','like_count'])
##for chunk in row_test_data:
##    tar_uid = chunk['uid']
##    predict_results = predict_results.append(user_mean.ix[tar_uid])

    
