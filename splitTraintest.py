import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import linear_model

from getPath import *
pardir = getparentdir()
from commonLib import *

datapath = pardir + '/data/factor1.csv'
trainpath = pardir + '/data/train.csv'
testpath = pardir + '/data/test.csv'

def readData():
    data = pd.read_csv(datapath, encoding = 'utf-8',iterator=True)
    loop = True
    chunkSize = 10000
    print("begin loop")
    i = 0
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            train,test = sampleTest(chunk)
            if i==0:
                train.to_csv(trainpath,encoding='utf-8',mode = 'w', index = False)
                test.to_csv(testpath,encoding='utf-8',mode = 'w', index = False)
            else:
                train.to_csv(trainpath,encoding='utf-8',mode = 'a', index = False,header= False)
                test.to_csv(testpath,encoding='utf-8',mode = 'a', index = False,header= False)
            i+=1
        except StopIteration:
            loop = False
         
def sampleTest(list):
    X_train, X_test= train_test_split(list, test_size=0.25,random_state=2)
    return X_train,X_test
           
if __name__=="__main__":
    readData()
