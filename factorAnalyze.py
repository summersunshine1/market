import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from sklearn import linear_model
from sklearn.model_selection import KFold

combine_dir = pardir+'/data/combine'
order_path = pardir+'/data/orders.csv'
order_train_path = pardir + '/data/order_products__train.csv'
product_path = pardir + '/data/products.csv'

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
    users['average_items_num'] = users.user_total_items/users.user_orders
    users['all_products'] = data.groupby('user_id')['product_id'].apply(set)
    users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)
    users.reset_index(level=['user_id'],inplace = True)
    return users
    
def get_product_feature(data):
    prods = get_data(product_path)
    products = pd.DataFrame()
    products['product_orders'] = data.groupby('product_id').size().astype(np.int32)
    products['reorders'] = data.groupby('product_id')['reordered'].sum().astype(np.int32)
    products['reorder_ratio'] = products.reorders/products.product_orders
    products['add_to_cart_order'] = data.groupby('product_id')['add_to_cart_order'].mean().astype(np.float32)
    prods = prods.join(products,on='product_id')
    del products
    return prods
    
def get_order_feature():
    orders = get_data(order_path)
    order_features = orders[['order_id','order_hour_of_day','days_since_prior_order']]
    return order_features
    
def get_user_product_feature(data):
    up_features = pd.DataFrame()
    up_features['user_product_num'] = data.groupby(['user_id','product_id']).size().astype(np.int16)
    up_features['product_orders_set'] = data.groupby(['user_id','product_id'])['order_id'].apply(set)
    up_features['product_orders_num'] = (up_features.product_orders_set.map(len)).astype(np.int16)
    up_features['add_cart_average'] = data.groupby(['user_id','product_id'])['add_to_cart_order'].mean().astype(np.float32)
    
    up_features['min_order_num'] = data.groupby(['user_id','product_id'])['order_number'].min()
    up_features['max_order_num'] = data.groupby(['user_id','product_id'])['order_number'].max()
    up_features['average_order_num'] = (up_features['max_order_num']-up_features['min_order_num'])/up_features['product_orders_num']
    up_features.drop(['min_order_num','max_order_num'],1,inplace=True)
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
    kf = KFold(n_splits=10,random_state=1,shuffle=True)
    train_indexs = []
    test_indexs = []
    for train_index, test_index in kf.split(data):
        train_indexs.append(train_index)
        test_indexs.append(test_index)
    return train_indexs,test_indexs 

def factor_analyze_main():
    filelist = listfiles(combine_dir)
    clf = linear_model.SGDClassifier()
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
        features = ['user_total_items','average_days_between_orders','user_orders','average_items_num','all_products','total_distinct_items',
        'product_orders','reorders','reorder_ratio','add_to_cart_order','user_product_num','product_orders_set','product_orders_num','add_cart_average',
        'average_order_num','label']
        finalfourth = fourth[features]
        del fourth
        trainlist = finalfourth.values.tolist()
        del finalfourth
        train_indexs,test_indexs = split_train_and_test(finalfourth.values.tolist())
        traindata = trainlist[train_indexs]
        testdata = testlist[test_indexs]
        train = traindata[:,:-1]
        train_label =  traindata[:,-1]
        test = testdata[:,:-1]
        test_label = testdata[:,-1]
        print(train_label)
        print(test_label)
        break
        
        
if __name__=="__main__":
    factor_analyze_main()
    

    