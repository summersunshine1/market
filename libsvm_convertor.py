from sklearn.datasets import dump_svmlight_file

import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *

train_path = pardir+'/data/orders.csv'

def readData():
    data = pd.read_csv(train_path, encoding = 'utf-8',iterator=True)
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
    
if __name__=="__main__":
    convert_libsvm()
    