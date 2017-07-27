import numpy as np
import os
import pandas as pd
import xgboost as xgb
import pickle

from getPath import *
pardir = getparentdir()

train_path = pardir+'/data/libsvm/train#dtrain.cache'
test_path = pardir+'/data/libsvm/test#dtest.cache'
log_path = pardir+'/data/xgblog.txt'

def train_model():
    dtrain = xgb.DMatrix(train_path)
    dtest = xgb.DMatrix(test_path)
    param = {'max_depth':8,'verbose':True}
    num_round=50
    watchlist = [(dtest,'eval'), (dtrain,'train')]
    bst = xgb.train(param, dtrain, num_round,watchlist)
    bst.save_model(pardir+'/model/xgb.model')
    clf = xgb.XGBClassifier()
    clf.fit(train, ans_train, eval_metric=xgb_f1,
                    eval_set=[(train, ans_train), (test, y_test)],
                    early_stopping_rounds=900)
        y_pred = clf.predict(test)
        predictions.append(y_pred)
        scores.append(f1_score(y_test, y_pred))

