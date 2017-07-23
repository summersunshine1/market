from sklearn.externals import joblib
import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from sklearn import linear_model
from factorAnalyze import *

model_path = pardir+'/model/lr.pkl'
combine_dir = pardir+'/data/combine'
order_path = pardir+'/data/orders.csv'

def get_data(file):
    data = readData(file)
    return data
    
def get_predict_order_feature():
    orders = get_data(order_path)
    order_features = orders[['user_id','order_id','order_hour_of_day','days_since_prior_order']][orders['eval_set']=='test']
    return order_features
    
def get_user_product_list(data):
    total = pd.DataFrame({'totalcount':data.groupby(['user_id','product_id']).size()})
    total.reset_index(['user_id','product_id'],inplace=True)
    return total
    
def gettestdata():
    filelist = listfiles(combine_dir)
    clf = joblib.load(model_path)
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
        features = ['user_total_items','average_days_between_orders','user_orders','average_items_num','total_distinct_items',
        'product_orders','reorders','reorder_ratio','add_to_cart_order','user_product_num','product_orders_num','add_cart_average',
        'average_order_num']
        # print(fourth.head())
        finalfourth = fourth[features]
        order_id = np.array(list(fourth['order_id']))
        product_id = np.array(list(fourth['product_id']))
        del fourth
        trainlist = finalfourth.values.tolist()
        del finalfourth
        predict = clf.predict(trainlist)
        index = np.where(predict==1)
        bought_order_ids = order_id[index]
        bought_product_ids = product_id[index]
        l = len(bought_order_ids)
        for i in range(l):
            bought_oid = bought_order_ids[i]
            if not bought_oid in resdic:
                resdic[bought_oid]=[]
            resdic[bought_oid].append(bought_product_ids[i])
    print(resdic)
    
if __name__=="__main__":
    gettestdata()
    
        