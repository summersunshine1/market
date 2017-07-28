import numpy as np
import os
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.metrics import f1_score

from getPath import *
pardir = getparentdir()

train_path = pardir+'/data/libsvm/train#dtrain.cache'
test_path = pardir+'/data/libsvm/test#dtest.cache'
log_path = pardir+'/data/xgblog.txt'

def f1_score1(y_predict,y_true):
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y_predict] 
    return f1_score(y_true.get_label(), y_bin)

def train_model():
    dtrain = xgb.DMatrix(train_path)
    dtest = xgb.DMatrix(test_path)
    param = {'max_depth':5,'verbose':True,'min_child_weight':1,'gamma':0,'n_estimators':1000,
    'subsample':0.8, 'colsample_bytree':0.8,'scale_pos_weight':6.7,'nthread':4}
    num_round=100
    watchlist = [(dtest,'eval'), (dtrain,'train')]
    bst = xgb.train(param, dtrain, num_round,watchlist)
    bst.save_model(pardir+'/model/xgb1.model')
    print(bst)
    y_pred = bst.predict(dtest)
    print(f1_score1(y_pred, dtest))
    
train_model()

