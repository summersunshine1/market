import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import linear_model
import os

from getPath import *
pardir = getparentdir()
from commonLib import *

dataPath = pardir+'/data/train.csv'

def aggregateLabels():
    data = pd.read_csv(dataPath, encoding = 'utf-8',iterator=True)
    loop = True
    chunkSize = 10000
    labelzero = 0
    labelone = 0
    print("begin loop")
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            values = np.array(chunk.values.tolist())
            del chunk
            labels = values[:,-1]
            labelzero += len(np.where(labels==0)[0])
            labelone += len(np.where(labels==1)[0])
        except StopIteration:
            loop = False
    print("zero:"+str(labelzero)+" one:"+str(labelone))
    
if __name__=="__main__":
    aggregateLabels()