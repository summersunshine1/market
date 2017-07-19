import pandas as pd 
import gc
from getPath import *
pardir = getparentdir()

order_path = pardir+'/data/orders.csv'
order_prior_path = pardir+'/data/order_products__prior.csv'
order_train_path = pardir + '/data/order_products__train.csv'
user_path = pardir+'/data/user/'

def readData():
    order_prior = pd.read_csv(order_prior_path, encoding = 'utf-8',iterator=True)
    order = pd.read_csv(order_path, encoding = 'utf-8')
    print(order.info())
    
    loop = True
    chunkSize = 10000
    chunks = []
    print("begin loop")
    while loop:
        try:
            chunk = order_prior.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            # print "Iteration is stopped."
    print("end loop")
    df = pd.concat(chunks, ignore_index=True)
    print("end concat")
    del chunks
    res = pd.merge(df, order, on = 'order_id')
    del df
    del order
    userid = pd.unique(res['user_id'])
    for user in userid:
        temp = res[res['user_id'] == user]
        temp.to_csv(user_path+str(user)+'.csv',encoding='utf-8')
        del temp
    
    
if __name__=="__main__":
    readData()