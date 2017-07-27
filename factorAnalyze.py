#!/usr/bin/python
import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn import preprocessing
from onehotencoder import *

combine_dir = pardir+'/data/combine'
order_path = pardir+'/data/orders.csv'
order_train_path = pardir + '/data/order_products__train.csv'
product_path = pardir + '/data/products.csv'

factor_path = pardir+'/data/factor4.csv'

def get_data(file):
    data = readData(file)
    return data
    
def get_user_feature(data):
    users = pd.DataFrame()
    users['user_total_items'] = data.groupby('user_id').size().astype(np.int16)
    orders = get_data(order_path)
    users['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
    users['user_orders'] = orders.groupby('user_id').size().astype(np.int16)
    del orders
    users['average_items_num'] = round(users.user_total_items/users.user_orders,2).astype(np.float32)
    users['all_products'] = data.groupby('user_id')['product_id'].apply(set)
    users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)
    users.reset_index(level=['user_id'],inplace = True)
    return users
    
def get_product_feature(data):
    prods = get_data(product_path)
    products = pd.DataFrame()
    products['product_orders'] = data.groupby('product_id').size().astype(np.int32)
    products['reorders'] = data.groupby('product_id')['reordered'].sum().astype(np.int32)
    products['reorder_ratio'] = round(products.reorders/products.product_orders,2).astype(np.float32)
    products['add_to_cart_order'] = data.groupby('product_id')['add_to_cart_order'].mean().astype(np.float32)
    prods = prods.join(products,on='product_id')
    del products
    return prods
    
    
def get_order_feature():
    orders = get_data(order_path)
    # order_features = orders['order_id']#,'order_hour_of_day','order_dow','days_since_prior_order']]
    order_hour = encodetime(24)
    order_dow = encodetime(7)
    attr = []
    order_features = pd.DataFrame()
    for i in range(24+7):
        attr.append(str(i))
        order_features[str(i)] = 0
    order_hours = np.array([order_hour[o] for o in orders['order_hour_of_day']])
    order_dows = np.array([order_dow[o] for o in orders['order_dow']])
    total = np.hstack((order_hours,order_dows))
    order_features[attr] = total
    order_features['order_id'] = orders['order_id']
    del orders
    return order_features
    
def get_user_product_feature(data):
    up_features = pd.DataFrame()
    up_features['user_product_num'] = data.groupby(['user_id','product_id']).size().astype(np.int16)
    up_features['product_orders_set'] = data.groupby(['user_id','product_id'])['order_id'].apply(set)
    up_features['product_orders_num'] = (up_features.product_orders_set.map(len)).astype(np.int16)
    
    up_features['add_cart_average'] = data.groupby(['user_id','product_id'])['add_to_cart_order'].mean().astype(np.float32)
    
    up_features['min_order_num'] = data.groupby(['user_id','product_id'])['order_number'].min().astype(np.int16)
    up_features['max_order_num'] = data.groupby(['user_id','product_id'])['order_number'].max().astype(np.int16)
    up_features['average_order_num'] = ((up_features['max_order_num']-up_features['min_order_num'])/up_features['product_orders_num']).astype(np.float32)
    # up_features.drop(['min_order_num','max_order_num'],1,inplace=True)
    up_features.reset_index(level=['user_id', 'product_id'],inplace = True)
    return up_features
    
def get_user_order_product_list(data):
    total = pd.DataFrame({'totalcount':data.groupby(['user_id','order_id','product_id']).size()})
    total.reset_index(['user_id','order_id','product_id'],inplace=True)
    return total

def get_train_data():
    order_train = pd.read_csv(order_train_path, encoding = 'utf-8')
    order = pd.read_csv(order_path, encoding = 'utf-8')
    res = pd.merge(order_train, order, on = 'order_id')
    del order_train
    del order
    data = res[['user_id', 'product_id']].apply(tuple, axis=1) 
    del res
    return data
    
def split_train_and_test(data):
    kf = KFold(n_splits=5,random_state=1,shuffle=True)
    train_indexs = []
    test_indexs = []
    for train_index, test_index in kf.split(data):
        # return train_index, test_index
        train_indexs.append(train_index)
        test_indexs.append(test_index)
    return train_indexs,test_indexs 
    
def f1_score1(y_true, y_predict):
    return f1_score(y_true, y_predict)

def factor_analyze_main():
    filelist = listfiles(combine_dir)
    clf = linear_model.SGDClassifier()
    j = 0
    for file in filelist:
        data = get_data(file)
        total = get_user_order_product_list(data)
        users = get_user_feature(data)
        first = pd.merge(total,users,on='user_id')
        del users,total
        prods = get_product_feature(data)
        second = pd.merge(first,prods,on='product_id')
        del prods, first
        orders = get_order_feature()
        third = pd.merge(second,orders,on='order_id')
        del orders, second
        up_features = get_user_product_feature(data)
        fourth = pd.merge(up_features,third,on=['user_id','product_id'])
        del up_features,third
        fourth['user_product'] = fourth[['user_id', 'product_id']].apply(tuple, axis=1)
        train_info = get_train_data()
        fourth['label'] = fourth["user_product"].isin(train_info).astype(int)
        del train_info
        fourth['up_orders_ratio'] = fourth['user_product_num']/fourth['user_orders']
        fourth['up_orders_since_lastorder'] = fourth['user_orders']-fourth['max_order_num']
        fourth['up_orders_since_firstorder'] = fourth['user_orders']-fourth['min_order_num']
        
        features = ['user_id','product_id','order_id','user_total_items','average_days_between_orders','user_orders','average_items_num','total_distinct_items',
        'product_orders','reorders','reorder_ratio','add_to_cart_order','user_product_num','product_orders_num','add_cart_average',
        'average_order_num','up_orders_ratio','up_orders_since_lastorder','up_orders_since_firstorder','days_since_prior_order']
        for i in range(24+7):
            features.append(str(i))
        features.append('label')
        finalfourth = fourth[features]
        if j==0:
           finalfourth.to_csv(factor_path,encoding='utf-8',mode = 'w', index = False)
        else:
           finalfourth.to_csv(factor_path,encoding='utf-8',mode = 'a', header=False, index = False)
        j+=1
        del fourth
        del finalfourth
        continue
        # print(finalfourth)
        trainlist = np.array(finalfourth.values.tolist())
        train_data = trainlist[:,:-1]
        # process_train_data = preprocessing.scale(train_data)
        # print(process_train_data)
        label = trainlist[:,-1]
        del finalfourth,trainlist
        # del train_data
        train_indexs,test_indexs = split_train_and_test(train_data)
        l = len(train_indexs)
        scores = []
        for i in range(l):
            traindata = train_data[train_indexs[i]]
            testdata = train_data[test_indexs[i]]
            train_label = label[train_indexs[i]]
            test_label = label[test_indexs[i]]
            clf.partial_fit(traindata,train_label.ravel(),classes = [0,1])
            predict = clf.predict(testdata)
            score = f1_score1(test_label, predict)
            scores.append(score)
        print(np.mean(scores))
    # path = pardir+'/model/lr.pkl'
    # joblib.dump(clf, path)

if __name__=="__main__":
    factor_analyze_main()
    

    
