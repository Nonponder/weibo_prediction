# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 10:16:05 2017

@author: WangFeichi
for Weibo Predict
method: use own cloest weibos

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn import model_selection
import time

# 微博内容的相关性：余弦函数
def weibo_distance(x1,x2):
    s_x1 = 0
    dict_x1 = {}
    for t_word,t_tfidf in x1:
        s_x1 += t_tfidf*t_tfidf
        dict_x1[t_word] = t_tfidf
    if s_x1==0:
#        raise Exception("Chinese word segmentation error: x1")
        return 0
    s_x2 = 0
    s_x1x2 = 0
    for t_word,t_tfidf in x2:
        s_x2 += t_tfidf*t_tfidf
        if dict_x1.has_key(t_word):
            s_x1x2 += t_tfidf*dict_x1[t_word]
    if s_x2==0:
#        raise Exception("Chinese word segmentation error: x2")
        return 0
    w_dis = s_x1x2/(s_x1*s_x2)**0.5
    return w_dis
    

def get_keyword(content):
    if content is not np.nan and isinstance(content,str):
        tags = jieba.analyse.extract_tags(content,withWeight = True)
    else:
        tags = []
    return tags
    
    
def dis_1toN(test_data,train_data):
    closely_thre = 0.7
    cal_dist = lambda x: weibo_distance(test_data,x)
    test_dist = train_data['keyword'].apply(cal_dist)
    close_set = train_data.ix[test_dist>closely_thre, \
                              ['forward_count','comment_count','like_count']]
    return [close_set['forward_count'].mean(), \
            close_set['comment_count'].mean(), \
            close_set['like_count'].mean()]


def weibo_predict_evaluation(test_set,predict_results):
    predict_results = predict_results.round()
    dev = predict_results-test_set
    forward_dev = dev['forward_count']/(test_set['forward_count']+5)
    comment_dev = dev['comment_count']/(test_set['comment_count']+3)
    like_dev = dev['like_count']/(test_set['like_count']+3)
    precision = 1-forward_dev.abs()*0.5-comment_dev.abs()*0.25-like_dev.abs()*0.25
    count = test_set.sum(axis = 1)
    count[count>100] = 100
    count = count+1
    right_count = count[precision>0.8].sum()
    return float(right_count)/count.sum()
    

if __name__ == "__main__":
    time_start = time.clock()
    #%% load training data
    row_train_data = pd.read_table('../Weibo Data/Weibo Data/weibo_train_data.txt', \
    names = ['uid','mid','time','forward_count','comment_count','like_count','content'], \
    dtype = {'uid':str,'mid':str,'time':str,'forward_count':int, \
    'comment_count':int,'like_count':int,'content':str}, \
    iterator = True)
    train_data = pd.DataFrame(columns={'uid','mid','time','forward_count','comment_count','like_count','content'})
    train_data = row_train_data.read()
#    train_data = row_train_data.read(nrows = 100000)
    train_data = train_data.set_index(['uid','time','mid'])
    train_data = train_data.sortlevel(level=['uid','time'])
    # 提取关键词
    train_data['keyword'] = train_data['content'].apply(get_keyword)
    train_data = train_data.drop('content',axis=1)
    
    #%% training  ----cross validation-----
    cv_train_data,cv_test_data = model_selection.train_test_split(train_data,test_size=0.1,random_state = 100)
    
    # 均值预测法  as baseline
    user_mean = cv_train_data[['forward_count','comment_count','like_count']].mean(axis = 0,level = ['uid'])
    t_test = cv_test_data.reset_index()
    mean_results = user_mean.loc[t_test['uid']]
    mean_results = mean_results.set_index(cv_test_data.index)
    preformance_1 = weibo_predict_evaluation(cv_test_data[['forward_count','comment_count','like_count']],mean_results)
    print preformance_1
    
    # KNN预测——————user_KNN
    like_thur = 0.1 # 近邻阈值
    user_sum = cv_train_data[['forward_count','comment_count','like_count']].sum(axis = 0,level = ['uid'])
    user_sum_all = user_sum.sum(axis = 1)
    zombie = user_sum[user_sum_all==0]# 僵尸粉
    general = user_sum[user_sum_all!=0]
    cv_test_data = cv_test_data.reset_index('time')
    cv_test_data = cv_test_data.drop('time',axis = 1)
    test_iter = cv_test_data['keyword'].groupby(level='uid')# 排序计算，减少索引次数
    residual_results = cv_test_data[['forward_count','comment_count','like_count']]
    residual_results = residual_results-residual_results
    for t_iter in test_iter:
        if t_iter[0] in zombie.index:
            # 僵尸粉狗带
            pass
        elif t_iter[0] not in general.index:
            # 新用户
            pass #---------------???
        else:
            t_history = cv_train_data.loc[t_iter[0],:,:]
            t_mean = user_mean.loc[t_iter[0]]
            for t_ind,t_keyword in t_iter[1].iteritems():
                if t_keyword==[]:
                    # 关键词不存在
                    residual_results.loc[t_ind] = t_mean  #均值插值or最小值插值???
                else:
                    cal_dist = lambda x: weibo_distance(t_keyword,x)
                    t_distance = t_history['keyword'].apply(cal_dist)
                    t_like = t_history[t_distance>like_thur]
                    t_distance = t_distance[t_distance>like_thur]
                    if t_like.shape[0]==0:
                        residual_results.loc[t_ind] = t_mean
                    else:
                        sum_dist = t_distance.sum()
                        t_weight = t_distance/sum_dist
                        t_forward = t_like['forward_count']*t_weight
                        t_comment = t_like['comment_count']*t_weight
                        t_likec = t_like['like_count']*t_weight
                        residual_results.loc[t_ind] = [t_forward.sum(), \
                                            t_comment.sum(), \
                                            t_likec.sum()]
    residual_results = residual_results.ix[cv_test_data.index]
    preformance_2 = weibo_predict_evaluation(cv_test_data[['forward_count','comment_count','like_count']], \
                                             residual_results)
    print preformance_2

    #%% test----- output
#    row_test_data = pd.read_table('../Weibo Data/Weibo Data/weibo_predict_data.txt', \
#    names = ['uid','mid','time','content'], \
#    dtype = {'uid':str,'mid':str,'time':str,'content':str}, \
#    iterator = True)
#    test_data = pd.DataFrame(columns={'uid','mid','time','content'})
#    test_data = row_test_data.read()
##    test_data = row_test_data.read(nrows = 100)
#    pre_test_data = test_data.set_index(['uid','mid'])
#    pre_test_data = pre_test_data.drop('time',axis = 1)
#    
#    # 提取关键词
#    pre_test_data['keyword'] = pre_test_data['content'].apply(get_keyword)
#    pre_test_data = pre_test_data.drop('content',axis=1)
#    
#    # predict------ user_KNN
#    like_thur = 0.1 # 近邻阈值
#    user_mean = train_data[['forward_count','comment_count','like_count']].mean(axis = 0,level = ['uid'])
#    user_sum = train_data[['forward_count','comment_count','like_count']].sum(axis = 0,level = ['uid'])
#    user_sum_all = user_sum.sum(axis = 1)
#    zombie = user_sum[user_sum_all==0]# 僵尸粉
#    general = user_sum[user_sum_all!=0]
#    test_iter = pre_test_data['keyword'].groupby(level='uid')# 排序计算，减少索引次数
#    residual_results = pd.DataFrame(np.zeros([pre_test_data.shape[0],3]), \
#                                    columns = ['forward_count','comment_count','like_count'], \
#                                    index = pre_test_data.index)
#    for t_iter in test_iter:
#        if t_iter[0] in zombie.index:
#            # 僵尸粉狗带
#            pass
#        elif t_iter[0] not in general.index:
#            # 新用户
#            pass #---------------???
#        else:
#            t_history = train_data.loc[t_iter[0],:,:]
#            t_mean = user_mean.loc[t_iter[0]]
#            for t_ind,t_keyword in t_iter[1].iteritems():
#                if t_keyword==[]:
#                    # 关键词不存在
#                    residual_results.loc[t_ind] = t_mean  #均值插值or最小值插值???
#                else:
#                    cal_dist = lambda x: weibo_distance(t_keyword,x)
#                    t_distance = t_history['keyword'].apply(cal_dist)
#                    t_like = t_history[t_distance>like_thur]
#                    t_distance = t_distance[t_distance>like_thur]
#                    if t_like.shape[0]==0:
#                        residual_results.loc[t_ind] = t_mean
#                    else:
#                        sum_dist = t_distance.sum()
#                        t_weight = t_distance/sum_dist
#                        t_forward = t_like['forward_count']*t_weight
#                        t_comment = t_like['comment_count']*t_weight
#                        t_likec = t_like['like_count']*t_weight
#                        residual_results.loc[t_ind] = [t_forward.sum(), \
#                                            t_comment.sum(), \
#                                            t_likec.sum()]
#    residual_results = residual_results.ix[pre_test_data.index]
#    residual_results = residual_results.round()
#    to_str = lambda x:str(int(x))
#    residual_results = residual_results.applymap(to_str)
#    out_put = residual_results['forward_count']+','+residual_results['comment_count']+ \
#                              ','+residual_results['like_count']
#    out_put = out_put.reset_index()
#    out_put.to_csv('results/model_2_results_1.txt',sep = '\t',header = False,index = False)
    
    
    
    time_end = time.clock()
    print '\nElapsed time: %.8f seconds\n' % (time_end-time_start)
