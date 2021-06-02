#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  https://www.zhihu.com/people/z285098346/posts
import time
t1 = time.perf_counter()

from icecream import ic

import logging
#声明了一个 Logger 对象
logger = logging.getLogger('wf_logger_name')
import sys
logger.setLevel(logging.DEBUG)
# 创建一个流处理器handler并将其日志级别设置为DEBUG
handler = logging.FileHandler('./wf_12folds.log', mode='w', encoding=None, delay=False)
#  handler.setLevel(logging.CRITICAL)  ; handler.setFormatter( logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='Day_%d %H:%M:%S') )
handler.setLevel(logging.CRITICAL)  ; handler.setFormatter( logging.Formatter(fmt='[%(asctime)s] - %(message)s', datefmt='Day_%d %H:%M:%S') )
logger.addHandler(handler)


import datetime
import math
import os
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists('out/'):
    os.makedirs('out/')

print('begin')
import pandas as pd
# 没快，反而慢了
# import modin.pandas as pd
import numpy as np
LOW_MAPE_PE = 99999
LOW_MAPE_DAY = 99999
# pd.set_option('display.min_rows', 100
pd.set_option('display.max_rows', 15)
num_round = 4000
early_stopping_rounds = 300
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'min_child_weight':2,
    'max_depth': 12,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'seed': 2021,
    'nthread': 8,
}
params2 = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'min_child_weight':10,
    'max_depth': 10,
    'colsample_bylevel':0.7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'seed': 2021,
    'nthread': 8,
}
folds = 12
#--------------------特征工程函数-------------------
def pe_level(x):
    if x in range(18,25):
        return 3
    elif x in range(25,30):
        return 2
    elif x in range(30,37):
        return 3
    elif x in range(37,39):
        return 1
    else:
        return 0
def date_feature(data):
    data['work_holi']=data['WKD_TYP_CD'].map({'WN':0,'SN': 1, 'NH': 1, 'SS': 1, 'WS': 0})
    data['date']=pd.to_datetime(data['date'])     # 变成形如 2020-11-30
    data['dayofweek']=data['date'].dt.dayofweek+1
    data['day']=data['date'].dt.day
    data['month']=data['date'].dt.month
    data['year']=data['date'].dt.year
    # 年底业务量增大，刻画出向上递增的趋势。 week_of_year：一年内的周数（1-52） day_of_year：一年内的天数（1-365）
    # 模型太蠢，不能从原始的年月日，学到整年趋势？
    data['week_of_year']=data['date'].dt.weekofyear
    data['day_of_year']=data['date'].dt.dayofyear

    # period_m24：与中间时段的绝对值差，能体现正午对称性。 week_mWes：以周三为对称轴的绝对值差，周末置为0。
    data['period_m24']=data['periods'].apply(lambda x:abs(x-24))
    # +1 避免出现0？
    data['week_mWes']=data['dayofweek'].apply(lambda x:0 if x in [6,7] else abs(x-3)+1)
    data['pe_level']=data['periods'].apply(pe_level)
    data['sin_period']=data['periods'].apply(lambda x:math.sin(x*math.pi/48)) #可以更好刻画每天的周期？
    # q1,q2：特征交叉，强特增益。（效果不明显，不过其他比赛可以用）
    data['q1']=data['period_m24']**2+data['week_mWes']**2
    data['q2']=data['day_of_year']**2+data['periods']**2
    # 4个季度
    data['quarter']=data['date'].dt.quarter
    data.drop(['date','post_id'],axis=1,inplace=True)
    return data
def xgb_model_A(train_x, train_y, test_x,test_y):
    predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    # 不是说时序预测不要交叉验证吗？
    seed = 2021
    # shuffle真的好？
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train = np.zeros(train_x.shape[0])
    test_pre = np.zeros(test_x.shape[0])
    test_pre_total = np.zeros((folds, test_x.shape[0]))
    total_pre_value_test = np.zeros((folds, test_x.shape[0]))
    #  cv_scores = []
    #  cv_rounds = []


    for i, (train_index, val_index) in enumerate(kf.split(train_x, train_y)):
        #  print("Fold", i)
        X = train_x[train_index]
        Y = train_y[train_index]
        fold_x = train_x[val_index]
        fold_y = train_y[val_index]
        train_matrix = xgb.DMatrix(X, label=Y)
        test_matrix = xgb.DMatrix(fold_x, label=fold_y)
        evals = [(train_matrix, 'train'), (test_matrix, 'val')]
        if test_matrix:
            #  xgboost.train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
                    #  maximize=None, early_stopping_rounds=None, evals_result=None, verbose_eval=True, xgb_model=None, callbacks=None)
            model = xgb.train(params, train_matrix, num_round, evals=evals, verbose_eval=2000,feval=wf_eval_mape,
                            early_stopping_rounds=early_stopping_rounds
                            )
            cv_pre = model.predict(xgb.DMatrix(fold_x),ntree_limit = model.best_iteration)
            # 每次交叉验证，都在要提交的数据集上预测，最后取mean
            test_pre_total[i, :] = model.predict(xgb.DMatrix(test_x),ntree_limit = model.best_iteration)


            #  cv_scores.append(mean_squared_error (fold_y, cv_pre))
            #  cv_rounds.append(model.best_iteration)
    test_pre[:] = test_pre_total.mean(axis=0)
    return test_pre
def my_mape(real_value, pre_value):
    real_value, pre_value = np.array(real_value), np.array(pre_value)
    return np.mean(np.abs((real_value - pre_value) /( real_value+1)))
def wf_eval_mape(pre, train_set):
    real = train_set.get_label()
    score = my_mape(real, pre)
    return 'wf_eval_mape', score
def xgb_model_B(train_x, train_y, test_x, test_y):
    predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    # 不是说时序预测不要交叉验证吗？
    seed = 2021
    # shuffle真的好？
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train = np.zeros(train_x.shape[0])
    test_pre = np.zeros(test_x.shape[0])
    test_pre_total = np.zeros((folds, test_x.shape[0]))
    total_pre_test = np.zeros((folds, test_x.shape[0]))
    cv_scores = []
    cv_rounds = []


    for i, (cv_train_index, cv_val_index) in enumerate(kf.split(train_x, train_y)):
        X = train_x[cv_train_index]
        Y = train_y[cv_train_index]
        fol_x = train_x[cv_val_index]
        fol_y = train_y[cv_val_index]
        train_matrix = xgb.DMatrix(X, label=Y)
        test_matrix = xgb.DMatrix(fol_x, label=fol_y)
        evals = [(train_matrix, 'train'), (test_matrix, 'val')]
        if test_matrix:
            model = xgb.train(params2, train_matrix, num_round, evals=evals, verbose_eval=2000,feval=wf_eval_mape,
                            early_stopping_rounds=early_stopping_rounds
                            )
            pre = model.predict(xgb.DMatrix(fol_x),ntree_limit = model.best_iteration)
            # 每次交叉验证，都在要提交的数据集上预测，最后取mean
            pred = model.predict(xgb.DMatrix(test_x),ntree_limit = model.best_iteration)
            train[cv_val_index] = pre
            #  cv_scores.append(mean_squared_error (fol_y, pre))
            #  cv_rounds.append(model.best_iteration)
            total_pre_test[i, :] = pred
    test_pre[:] =total_pre_test.mean(axis=0)
    #-----------------------------------------

    #todo
    '''
    # 现在是多个模型都在11月预测一遍，取平均
    # 交叉验证后，在所有训练数据上训练，在11月算指标
    train_matrix = xgb.DMatrix(train_x, label=train_y)
    test_matrix = xgb.DMatrix(test_x, label=test_y)
    evals = [(train_matrix, 'train'), (test_matrix, 'val')]
    model = xgb.train(params2, train_matrix, num_round, evals=evals, verbose_eval=200,feval=wf_eval_mape,
                        early_stopping_rounds=early_stopping_rounds
                        )
    wf_pred = model.predict(xgb.DMatrix(test_x),ntree_limit = model.best_iteration)
    #todo
    '''
    return test_pre
#读取数据
train=pd.read_csv('./data/train_v2.csv')
test_pe=pd.read_csv('./data/wf_test_Nov_peri.csv')#按0.5h计算
week=pd.read_csv('./data/wkd_v1.csv')
week=week.rename(columns={'ORIG_DT':'date'})
train['date']=pd.to_datetime(train['date'], format='%Y/%m/%d')
test_pe['date']=pd.to_datetime(test_pe['date'], format='%Y/%m/%d')
week['date']=pd.to_datetime(week['date'], format='%Y/%m/%d')
#数据处理
train_period_A=train[train['post_id']=='A'].copy()
train_period_A.reset_index(drop=True,inplace=True)
train_period_A=train_period_A.groupby(by=['date','post_id','periods'], as_index=False)['amount'].agg('sum')
train_period_B=train[train['post_id']=='B'].copy()
train_period_B.reset_index(drop=True,inplace=True)
train_period_B.drop(['biz_type'],axis=1,inplace=True)
train_period_A=train_period_A.merge(week)
train_period_B=train_period_B.merge(week)
train_period_A['amount']=train_period_A['amount']/1e4
train_period_B['amount']=train_period_B['amount']/1e4
# mean : .036136426056337836 max: 0.4 除1e4干啥？ 不除呢？

# 由于预测的11、12月均没有节日和调休情况出现，所以去掉'NH','SS','WS'三种类型的数据。
train_period_A=train_period_A[~train_period_A['WKD_TYP_CD'].isin(['NH','SS','WS'])]
train_period_B=train_period_B[~train_period_B['WKD_TYP_CD'].isin(['NH','SS','WS'])]
train_period_A=date_feature(train_period_A)
train_period_B=date_feature(train_period_B)
test_period_A=test_pe[test_pe['post_id']=='A'].reset_index(drop=True)
test_period_B=test_pe[test_pe['post_id']=='B'].reset_index(drop=True)
test_period_A=test_period_A.merge(week)
test_period_B=test_period_B.merge(week)
test_period_A=date_feature(test_period_A)
test_period_B=date_feature(test_period_B)
# test_period_A.drop(['amount'],axis=1,inplace=True)
# test_period_B.drop(['amount'],axis=1,inplace=True)

#-----------------------树模型-----------------------
all_features=['periods','work_holi','year','month','day','dayofweek','week_of_year','day_of_year','period_m24','week_mWes','pe_level','q1','q2']
#-------筛选数据月份---------
month_num=3
#----------------------------

# main
BEST_FOLD_NUM_PE = -1
BEST_FOLD_NUM_DAY = -1
ft_scores =[]
#  find best k folds
#  for idx_drop in range(len(all_features)):
for idx_drop in range(1,2):
    #  if idx_drop ==  len(all_features)-1:
    #      feature =  all_features[:-1]
    #  else:
    #      feature =  all_features[:idx_drop] + all_features[idx_drop+1:]
    feature =  all_features
    train_input=train_period_A#训练集
    # 由于跨越了2020年初的这段时间（疫情影响），所以，业务量会有不规则突变.只保留疫情后的，训练集数据仅选取2020年4月以后的数据。
    #  train_input=train_input[(train_input['year']==2020) & (train_input['month']>month_num)].reset_index(drop=True)
    train_input=train_input.reset_index(drop=True)
    test_input=test_period_A #测试集
    train_x = train_input[feature].copy()
    train_y = train_input['amount']
    test_x = test_input[feature].copy()
    test_y = test_input['amount'].copy()

    xgb_test = xgb_model_A(train_x, train_y, test_x, test_y)
    pre_hour_A=[max(i,0) for i in xgb_test]

    train_input=train_period_B #训练集
    #  train_input=train_input[(train_input['year']==2020) & (train_input['month']>month_num)].reset_index(drop=True)
    train_input=train_input.reset_index(drop=True)
    test_input=test_period_B#测试集
    train_x = train_input[feature].copy()
    train_y = train_input['amount']
    test_x = test_input[feature].copy()
    test_y = test_input['amount'].copy()

    xgb_test = xgb_model_B(train_x, train_y, test_x, test_y)
    pre_hour_B=[max(i,0) for i in xgb_test]


    #------------------拼接文件------------------
    pre_period=[]
    for i in range( int(len(pre_hour_A)/48 ) ):#每一天
        for j in range(48):#每一个时间段
            pre_period.append(1e4*pre_hour_A[48*i+j])
        for j in range(48):
            pre_period.append(1e4*pre_hour_B[48*i+j])

    test_pe['amount']=pre_period
    test_pe['amount']=(test_pe['amount']).astype(int)

    # test_pe['date']=test_pe['date'].dt.strftime('%Y/%#m/%#d') # Return an Index of formatted strings specified by date_format
    #汇总预测结果得到test_day
    test_pe['date']=pd.to_datetime(test_pe['date'], format='%Y/%m/%d')
    test_day=test_pe.groupby(by=['date','post_id'],as_index=False)['amount'].agg('sum')
    #调整test_day
    test_day_A=test_day[test_day.post_id=='A'].copy()
    test_day_B=test_day[test_day.post_id=='B'].copy()


    #-----------------------------强行改结果(加规则)------------------------
    # 调整系数 由于年底业务量增加，特征和模型对于这部分识别不到位，需要人为的去调整一下。A/B类型业务分开乘以系数。 0.08225-->0.063左右
    test_day_A['amount']=test_day_A['amount']*1.07
    test_day=pd.merge(test_day,test_day_A, suffixes=('', '_A'),on=['date','post_id'],how='left')
    test_day_B['amount']=test_day_B['amount']*1.032
    test_day=pd.merge(test_day,test_day_B, suffixes=('', '_B'),on=['date','post_id'],how='left')

    test_day.fillna(0,inplace=True) # Replace all NaN elements with 0.
    test_day['amount_day_scaled']=(  test_day['amount_A']+test_day['amount_B']  ).astype(int).apply(lambda x:0 if x<200 else x)
    test_day.drop(['amount','amount_A','amount_B'],axis=1,inplace=True)

    # 放缩操作
    # 按天粗预测 调整系数后 预测准确率较高，将按period预测任务的每个时段占全天业务量的比例计算出来，乘任务一中每天的业务量，对于任务二的数据按比例放缩。 0.163-->0.147左右

    # 因为上面把test_day的amount放大了，所以amount_sum较小   test_day_A['amount']=test_day_A['amount']*1.07
    temp=test_pe.groupby(by=['date','post_id'],as_index=False)['amount'].agg({'amount_day':'sum'})
    # merged的2个dataFrame 行数不同时，会在行少的dataFrame生成NaN
    test_day=pd.merge(test_day,temp,on=['date','post_id'],how='left')
    test_pe=pd.merge(test_pe,test_day,on=['date','post_id'],how='left')
    test_pe['amount']= (test_pe['amount']/test_pe['amount_day'] )*test_pe['amount_day_scaled']   #按比例放缩
    test_pe.fillna(0,inplace=True)
    test_pe['amount']=test_pe['amount'].astype(int)
    #
    test_day['amount']=test_day['amount_day_scaled'].astype(int)

    test_day.drop(['amount_day', 'amount_day_scaled'],axis=1,inplace=True)
    test_pe.drop(['amount_day','amount_day_scaled'],axis=1,inplace=True)
    test_day['date']=test_day['date'].dt.strftime('%Y/%#m/%#d')
    test_pe['date']=test_pe['date'].dt.strftime('%Y/%#m/%#d')
    #输出结果


    gt_day = pd.read_csv('./data/wf_test_Nov_day.csv')
    day_score =my_mape(test_day['amount'], gt_day['amount'])

    gt_pe=pd.read_csv('./data/wf_test_Nov_peri.csv')#按0.5h计算
    pe_score =my_mape(test_pe['amount'], gt_pe['amount'])

    LOW_MAPE_PE  = pe_score
    BEST_FOLD_NUM_PE = folds
    logger.critical('drop: %s'%(all_features[idx_drop]))
    logger.critical('mape_pe  : %.2f'%(LOW_MAPE_PE*100  ))
    test_pe.to_csv('out/period.txt',sep=',',index=False)

    LOW_MAPE_DAY = day_score
    BEST_FOLD_NUM_DAY = folds
    logger.critical('mape_day : %.2f'%(LOW_MAPE_DAY*100 ))
    test_day.to_csv('out/day.txt',sep=',',index=False)

    logger.critical('------------------------------')

t2 = time.perf_counter()
print(t2-t1,' seconds')
print('BEST_FOLD_NUM_PE',BEST_FOLD_NUM_PE )
print('BEST_FOLD_NUM_DAY',BEST_FOLD_NUM_DAY )
