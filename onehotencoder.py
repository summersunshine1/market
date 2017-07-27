import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from factorAnalyze import *

from sklearn.preprocessing import OneHotEncoder
def encoder(arr):
    ohe = OneHotEncoder(sparse=False)#categorical_features='all',
    ohe.fit(arr)
    return ohe.transform(arr)  

def encodetime(bins):    
    arr = [[a] for a in range(bins)]
    res = encoder(arr)
    return res
    
if __name__=="__main__":
    get_order_feature()
    arr = encodetime(3)
    arr1 = encodetime(4)
    data = [0,1,1,2]
    datac = np.array([arr[d] for d in data])
    print(datac)
    df = pd.DataFrame()
    attr = []
    for i in range(7):
        attr.append(str(i))
        df[str(i)]=0
    # print(attr)
    # df['1']=0
    # df['2']=0
    # df['3']=0
    # df[attr]=datac
    # print(df)
    a=np.array([arr[d] for d in data])
    b = np.array([arr1[d] for d in data])
    c = np.hstack((a,b))
    print(np.array(c))
    # print(c)
    # print(attr)
    # print(df)
    df[attr] = c
    print(df)
    # print(c)
    # get_order_feature()