import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing

from getPath import *
pardir = getparentdir()
from commonLib import *

testpath = pardir + '/data/test.csv'
modelpath = pardir + '/model/sgd.pkl'

def validate(scale, clf):
    data = pd.read_csv(testpath, encoding = 'utf-8')
    dataarr = np.array(data.values.tolist())
    del data
    if scale == None:
        train = preprocessing.scale(dataarr[:,:-1])
    else:
        train = scaler.transform(dataarr[:,:-1])
    label = dataarr[:,-1]
    del dataarr
    predict = clf.predict(train)
    score = f1_score1(label, predict)
    print(score)
    
if __name__=="__main__":
    clf = joblib.load(modelpath)
    validate(None, clf)
    
    
    
