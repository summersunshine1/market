import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *

combine_dir = pardir+'/data/combine'
order_path = pardir+'/data/orders.csv'
order_train_path = pardir + '/data/order_products__train.csv'
product_path = pardir + '/data/products.csv'

def get_data(file):
    data = readData(file)
    return data
    
def get_user_feature(data):
    users = pd.DataFrame()
    users['total_items'] = data.groupby('user_id').size().astype(np.int16)
    orders = get_data(order_path)
    users['average days between orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
    users['orders'] = orders.groupby('user_id').size().astype(np.int16)
    del orders
    users['average_items_num'] = users.total_items/users.orders
    users['all_products'] = data.groupby('user_id')['product_id'].apply(set)
    users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)
    return users
    
def get_product_feature(data):
    prods = get_data(product_path)
    products = pd.DataFrame()
    products['orders'] = data.groupby('product_id').size().astype(np.int32)
    products['reorders'] = data.groupby('product_id')['reordered'].sum().astype(np.int32)
    products['reorder_ratio'] = products.reorders/products.orders
    products['add_to_cart_order'] = data.groupby('product_id')['add_to_cart_order'].mean().astype(np.float32)
    prods = prods.join(products,on='product_id')
    del products
    prods.set_index('product_id',drop = True, inplace = True)
    return prods
    
def get_order_feature():
    orders = get_data(order_path)
    order_features = orders[['order_id','order_hour_of_day','days_since_prior_order']]
    return order_features
    
def get_user_product_feature(data):
    up_features = pd.DataFrame()
    up_features['user_product_num'] = data.groupby(['user_id','product_id']).count().astype(np.int16)
    up_features['product_orders_set'] = data.groupby(['user_id','product_id'])['order_id'].apply(set)
    up_features['product_orders_num'] = (up_features.product_orders_set.map(len)).astype(np.int16)
    up_features['add_cart_average'] = data.groupby(['user_id','product_id'])['add_to_cart_order'].mean().astype(np.float32)
    # up_features['day_since_prior_average']=data.groupby(['user_id','product_id'])['day_since_prior_order'].sum().astype(np.float32)
    
    

def factor_analyze_main():
    filelist = listfiles(combine_dir)
    for file in filelist:
        data = get_data(file)
        # users = get_user_feature(data)
        # prods = get_product_feature(data)
        # orders = get_order_feature()
        
        break
        
if __name__=="__main__":
    factor_analyze_main()
    

    