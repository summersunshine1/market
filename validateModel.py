import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing

from getPath import *
pardir = getparentdir()
from commonLib import *

testpath = pardir + '/data/test.csv'
modelpath = pardir + '/model/scalesgd.pkl'
scaler_path = pardir+'model/scaler.save'
logPath = pardir +'/data/log.txt'

def validate(scale, clf):
    data = pd.read_csv(testpath, encoding = 'utf-8')
    dataarr = np.array(data.values.tolist())
    del data
    if scale == None:
        train = preprocessing.scale(dataarr[:,:-1])
    else:
        train = scale.transform(dataarr[:,:-1])
    label = dataarr[:,-1]
    del dataarr
    f = open(logPath, 'a', encoding='utf-8')
    predict = clf.predict(train)
    f.writelines(str(predict)+'\n')
    score = f1_score1(label, predict)
    f.close()
    return score
    
if __name__=="__main__":
    clf = joblib.load(modelpath)
    scale = joblib.load(scaler_path)
    score = validate(scale, clf)
    print(score)
    
    
    
