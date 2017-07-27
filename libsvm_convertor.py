from sklearn.datasets import dump_svmlight_file
import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *

train_path = pardir+'/data/train4.csv'
test_path =pardir+'/data/test4.csv'
train_svm_file = pardir+'/data/libsvm/train'
test_svm_file = pardir+'/data/libsvm/test'
temp_path = pardir+'/data/libsvm/temp'

def convert_libsvm(source_path, target_path):
    data = pd.read_csv(source_path, encoding = 'utf-8',iterator=True)
    loop = True
    chunkSize = 10000
    i = 0
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            datavalues = np.array(chunk.values.tolist())
            del chunk
            x = np.nan_to_num(datavalues[:,3:-1])
            y = datavalues[:,-1]
            del datavalues
            dump_svmlight_file(x, y,temp_path, zero_based=True,query_id=None)
            del x,y
            with open(target_path, 'a') as W:
                with open(temp_path, 'r') as R:
                    for line in R:
                        W.write(line)
                os.remove(temp_path)
        except StopIteration:
            loop = False
 
if __name__=="__main__":
    convert_libsvm(train_path,train_svm_file)
    convert_libsvm(test_path,test_svm_file)
    