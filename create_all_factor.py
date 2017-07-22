import pandas as pd
import time
import os

from getPath import *
pardir = getparentdir()
from commonLib import *

filter_dir = pardir+'/data/combine'
order_path = pardir+'/data/orders.csv'
order_train_path = pardir + '/data/order_products__train.csv'
factor_path = pardir + '/data/all_factor.csv'

def get_data(file):
    data = readData(file)
    return data
    
def get_train_data():
    order_train = pd.read_csv(order_train_path, encoding = 'utf-8')
    order = pd.read_csv(order_path, encoding = 'utf-8')
    res = pd.merge(order_train, order, on = 'order_id')
    del order_train
    del order
    data = res[['user_id', 'product_id']].apply(tuple, axis=1) 
    del res
    return data
    
def get_test_data(data):
    test = data[[data['eval_set'] == 'test']]
    print(test)
    
def get_item_importance(data):
    items = pd.DataFrame({'count':data.groupby(['user_id','product_id']).size()}).reset_index()
    user_items = pd.DataFrame({'totalcount':data.groupby(['user_id']).size()}).reset_index()
    res = pd.merge(items, user_items, on = 'user_id')
    item_importance = pd.DataFrame(res['count']/res['totalcount'])
    item_importance['user_id'] = res['user_id']
    item_importance['product_id'] = res['product_id']
    columns = item_importance.columns.values
    item_importance = item_importance.rename(columns={0: "item_importance"})
    return item_importance
    
def get_reorder_ratio(data):
    order_num = pd.DataFrame({'totalcount':data.groupby(['user_id','order_id']).size()}).reset_index()
    order_total = pd.DataFrame({'totalcount':order_num.groupby(['user_id']).size()}).reset_index()
    del order_num
    items = pd.DataFrame({'count':data.groupby(['user_id','product_id']).size()}).reset_index()
    res = pd.merge(items, order_total, on = 'user_id')
    del order_total,items
    res['order_importance'] = res['count']/res['totalcount']
    finalres = res.drop(['count','totalcount'],1)
    del res
    return finalres
    
def get_interval_average(data):
    time_mean = data.groupby(['user_id', 'order_id']).mean().astype(np.float32)['days_since_prior_order'].reset_index()
    totaltime_mean = time_mean.groupby(['user_id']).mean().astype(np.float32)['days_since_prior_order'].reset_index()#[days_since_prior_order].reset_index()
    totaltime_mean = totaltime_mean.rename(columns={'days_since_prior_order': "order_mean_since_prior"})
    del time_mean
    item_mean = data.groupby(['user_id', 'product_id']).mean().astype(np.float32)
    item_time_and_order = item_mean[['days_since_prior_order','add_to_cart_order']].reset_index()
    item_time_and_order = item_time_and_order.rename(columns={'days_since_prior_order': "item_mean_since_prior"})
    del item_mean
    return totaltime_mean, item_time_and_order
    
def get_all_user_product():
    filelist = listfiles(filter_dir)
    finalres = []
    for file in filelist:
        data = get_data(file)
        groups = data.groupby(["user_id","product_id"]).groups
        del data
        res = [k for k,v in groups.items()]
        del groups
        finalres += res
    return finalres
  
def analyze_main():
    train_info = get_train_data()
    filelist = listfiles(filter_dir)
    res = pd.DataFrame()
    i = 0
    for file in filelist:
        data = get_data(file)
        item_importance = get_item_importance(data)
        reorder_ratio = get_reorder_ratio(data)
        totaltime_mean, item_time_and_order = get_interval_average(data)
        del data
        first = pd.merge(item_importance, reorder_ratio, on = ['user_id','product_id'])
        del item_importance
        del reorder_ratio
        second = pd.merge(first, totaltime_mean, on="user_id")
        del first
        del totaltime_mean
        third = pd.merge(second, item_time_and_order, on=['user_id','product_id'])
        del second
        del item_time_and_order
        third['user_product'] = third[['user_id', 'product_id']].apply(tuple, axis=1)
        third['label'] = third["user_product"].isin(train_info).astype(int)
        third.drop('user_product',1,inplace=True)
        if i==0:
            third.to_csv(factor_path,encoding='utf-8',mode = 'w', index = False)
        else:
            third.to_csv(factor_path,encoding='utf-8',mode = 'a', header=False, index = False)
        print(i)
        i+=1
        del third
        
def form_predict():
    order = pd.read_csv(order_path, encoding = 'utf-8')
    test = order[order['eval_set'] == 'test']
    print(test)
        
         
if __name__=="__main__":
    form_predict()