import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib

from getPath import *
pardir = getparentdir()
from commonLib import *
trainpath = pardir + '/data/train.csv'
scalerpath = pardir + 'model/scaler.save'

def ScaleData():
    data = pd.read_csv(trainpath, encoding = 'utf-8',iterator=True)
    loop = True
    chunkSize = 10000
    standard_scalar = preprocessing.StandardScaler()
    print("begin loop")
    i = 0
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            values = chunk.values.tolist()
            del chunk
            standard_scalar.partial_fit(values)
        except StopIteration:
            loop = False
    joblib.dump(standard_scalar, scalerpath)

if __name__=="__main__":
    ScaleData()
            
    
