import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import linear_model

from getPath import *
pardir = getparentdir()
from commonLib import *

# trainpath = pardir + '/data/train.csv'
# testpath = pardir + '/data/test.csv'
data_path = pardir+'/data/factor1.csv'

features = ['user_total_items','average_days_between_orders','user_orders','average_items_num','total_distinct_items',
    'product_orders','reorders','reorder_ratio','add_to_cart_order','user_product_num','product_orders_num','add_cart_average',
    'average_order_num','up_orders_ratio','up_orders_since_lastorder','up_orders_since_firstorder','label']
    
def sampleTest(list):
    X_train, X_test= train_test_split(list, test_size=0.25,random_state=2)
    return X_train,X_test

def TrainData():
    data = pd.read_csv(data_path, encoding = 'utf-8',iterator=True)
    loop = True
    chunkSize = 10000
    print("begin loop")
    i = 0
    clf = linear_model.SGDClassifier()
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            train,test = sampleTest(chunk)
            trainvalue = np.array(train.values.to_list())
            testvalue = np.array(test.values.to_list())
            train_x = preprocessing.scale(trainvalue[:,:-1])
            train_y = trainvalue[:,-1]
            test_x = preprocessing.scale(testvalue[:,:-1])
            test_y = preprocessing.scale(testvalue[:,-1])
            for i in range(50):
                clf.partial_fit(train_x,train_y.ravel(),classes = [0,1])  
                if i%10==0:
                    predict = clf.predict(test_x)
                    score = f1_score1(test_y, predict)
                    print(str(i)+" iters" + str(score))
        except StopIteration:
            loop = False
    path = pardir+'/model/sgd.pkl' 
    
if __name__=="__main__":
   TrainData() 
    