import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import linear_model

from getPath import *
pardir = getparentdir()
from commonLib import *
from validateModel import *
scalerpath = pardir + '/model/scaler.save'
trainpath = pardir + '/data/train.csv'

def train_sgd():
    data = pd.read_csv(trainpath, encoding = 'utf-8',iterator=True)
    loop = True
    chunkSize = 10000
    scaler = joblib.load(scalerpath)
    clf = linear_model.SGDClassifier()
    print("begin loop")
    i = 0
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            values = np.array(chunk.values.tolist())
            del chunk
            scale_values = scaler.transform(values[:,:-1])
            for i in range(200):
                clf.partial_fit(scale_values,values[:,-1].ravel(),classes = [0,1])
                if i%50==0:
                    validate(scaler, clf)
        except StopIteration:
            loop = False
            
if __name__=="__main__":
    train_sgd()
    

