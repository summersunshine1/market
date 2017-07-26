import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import linear_model
import os

from getPath import *
pardir = getparentdir()
from commonLib import *
from validateModel import *
scalerpath = pardir + '/model/scaler.save'
trainpath = pardir + '/data/shuffled/shuffle_orders0.csv'
logPath = pardir +'/data/log.txt'
modelpath = pardir + '/model/scalesgd.pkl'

def train_sgd(iter,clf,scaler):
    data = pd.read_csv(trainpath, encoding = 'utf-8',iterator=True)
    loop = True
    chunkSize = 10000
    
    print("begin loop")
    f = open(logPath, 'a', encoding='utf-8')
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            values = np.array(chunk.values.tolist())
            del chunk
            scale_values = scaler.transform(values[:,:-1])
            clf.partial_fit(scale_values,values[:,-1].ravel(),classes = [0,1])    
        except StopIteration:
            loop = False
    score = validate(scaler, clf)
    f.writelines("iter:"+str(iter)+" score:"+str(score)+'\n')
    f.flush()
    f.close()
        
if __name__=="__main__":
    scaler = joblib.load(scalerpath)
    clf = linear_model.SGDClassifier()
    for i in range(15):
        train_sgd(i,clf,scaler)
    joblib.dump(clf, modelpath)
    
    

