import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *

res1_path = pardir+'/res/res.csv'
res2_path = pardir+'/res/reswithxgboost.csv'
combine_path = pardir+'/res/combine.csv'

def getdic(path):
    data = pd.read_csv(path, encoding='utf-8')
    order_ids = list(data['order_id'])

    products = list(data['products'])
    dictionary = dict(zip(order_ids, products))
    return dictionary
    
def merge():
    dic = getdic(res1_path)
    dic2 = getdic(res2_path)
    newdic = {}

    for k,v in dic.items():
        if v=='None':
            newdic[k] = dic2[k]
        else:
            if dic2[k]=='None':
                newdic[k]=dic[k]
            else:
                arr1 = dic2[k].split()
                res = [int(t) for t in arr1]
                arr2 = dic[k].split()
                res += [int(t) for t in arr2]
                res = sorted(res)
                res = [str(t) for t in res]
                newdic[k] = (' ').join(res)
    newtuple = sorted(newdic.items(), key=lambda key_value: key_value[0])
    # print(newtuple)
    f = open(combine_path,'w',encoding='utf-8')
    f.writelines(','.join(['"order_id"', '"products"']) + '\n')
    for k,v in newtuple:
        info = [str(k),v]
        line = (',').join(info)
        f.writelines(line+'\n')
    f.close()
    
merge()
