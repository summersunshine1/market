from sklearn.externals import joblib
import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from sklearn import linear_model
from factorAnalyze import *
import pickle
import xgboost as xgb
from onehotencoder import *
from sklearn.datasets import dump_svmlight_file
import os

model_path = pardir+'/model/xgb1.model'
combine_dir = pardir+'/data/combine'
order_path = pardir+'/data/orders.csv'
scalerpath = pardir + '/model/scaler.save'
testsvmpath = pardir+'/data/libsvm/finaltest'

def get_data(file):
    data = readData(file)
    return data
    
def get_predict_order_feature():
    order_hour = encodetime(24)
    order_dow = encodetime(7)
    attr = []
    order_features = pd.DataFrame()
    for i in range(24+7):
        attr.append(str(i))
        order_features[str(i)] = 0
    orders = get_data(order_path)  
    temp = orders[['user_id','order_id','order_dow','order_hour_of_day','days_since_prior_order']][orders['eval_set']=='test']
    del orders
    order_hours = np.array([order_hour[o] for o in temp['order_hour_of_day']])
    order_dows = np.array([order_dow[o] for o in temp['order_dow']])
    total = np.hstack((order_hours,order_dows))
    order_features[attr] = total
    order_features['order_id'] = temp['order_id']
    order_features['user_id'] = temp['user_id']
    del temp
    return order_features

def get_user_product_list(data):
    total = pd.DataFrame({'totalcount':data.groupby(['user_id','product_id']).size()})
    total.reset_index(['user_id','product_id'],inplace=True)
    return total
    
def write_res_to_file(resdic):
    orders = get_data(order_path)
    order_ids = list(orders['order_id'][orders['eval_set']=='test'])
    order_ids = sorted(order_ids)
    del orders
    resPath = pardir+'/data/res.csv'
    fw = open(resPath, 'w', encoding = 'utf-8')
    fw.writelines(','.join(['"order_id"', '"products"']) + '\n')
    for order_id in order_ids:
        if not order_id in resdic:
            fw.writelines(','.join([str(order_id), "None"])+'\n')
            continue
        products = resdic[order_id]
        products = sorted(products)
        products = [str(p) for p in products]
        proline = ' '.join(products)
        line = ','.join([str(order_id), proline]) + '\n'
        fw.writelines(line)
    fw.close()

def writelibsvm(trainlist):
    samplenum = np.shape(trainlist)[0]
    y = []
    for i in range(samplenum):
        y.append(-1)
    if os.path.exists(testsvmpath):
        os.remove(testsvmpath)
    dump_svmlight_file(trainlist, y,testsvmpath, zero_based=True,query_id=None)
    del trainlist,y
    
def convertPredict(predict):
    y_bin = np.array([1 if y_cont > 0.5 else 0 for y_cont in predict])
    return y_bin

def gettestdata():
    filelist = listfiles(combine_dir)
    # clf = joblib.load(model_path)
    # scaler = joblib.load(scalerpath)
    clf = xgb.Booster(model_file=model_path)
    resdic ={}
    for file in filelist:
        data = get_data(file)
        total = get_user_product_list(data)
        users = get_user_feature(data)
        first = pd.merge(total,users,on='user_id')
        del users,total
        prods = get_product_feature(data)
        second = pd.merge(first,prods,on='product_id')
        del prods, first
        orders = get_predict_order_feature()
        third = pd.merge(second,orders,on='user_id')
        del orders, second
        up_features = get_user_product_feature(data)
        fourth = pd.merge(up_features,third,on=['user_id','product_id'])
        del up_features,third
        fourth['up_orders_ratio'] = fourth['user_product_num']/fourth['user_orders']
        fourth['up_orders_since_lastorder'] = fourth['user_orders']-fourth['max_order_num']
        fourth['up_orders_since_firstorder'] = fourth['user_orders']-fourth['min_order_num']
        features = ['user_total_items','average_days_between_orders','user_orders','average_items_num','total_distinct_items',
        'product_orders','reorders','reorder_ratio','add_to_cart_order','user_product_num','product_orders_num','add_cart_average',
        'average_order_num','up_orders_ratio','up_orders_since_lastorder','up_orders_since_firstorder']
        for i in range(24+7):
            features.append(str(i))
        # print(fourth.head())
        finalfourth = fourth[features]
        order_id = np.array(list(fourth['order_id']))
        print(len(order_id))
        product_id = np.array(list(fourth['product_id']))
        del fourth
        trainlist = finalfourth.values.tolist()
        del finalfourth
        trainlist = np.nan_to_num(trainlist)
        print(np.shape(trainlist))
        writelibsvm(trainlist)
        dtest = xgb.DMatrix(testsvmpath)
        predict = clf.predict(dtest)
        predicts = convertPredict(predict)
        del predict
        index = np.where(predicts==1)
        del predicts
        bought_order_ids = order_id[index]
        bought_product_ids = product_id[index]
        l = len(bought_order_ids)
        for i in range(l):
            bought_oid = bought_order_ids[i]
            if not bought_oid in resdic:
                resdic[bought_oid]=[]
            resdic[bought_oid].append(bought_product_ids[i])
    write_res_to_file(resdic)
    
if __name__=="__main__":
    gettestdata()
    
        