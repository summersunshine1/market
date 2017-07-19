import pandas as pd
import time
import os

from getPath import *
pardir = getparentdir()
from commonLib import *
combine_dir= pardir+'/data/combine/'
reorderuser_path = pardir + '/data/reorderuser.csv'

def getreorderuser():
    data = open(reorderuser_path,'r',encoding = 'utf-8')
    line = data.readline()
    arr = line.strip('\n').split(',')
    arr = [int(t) for t in arr]
    return arr

def readData(path):
    data = pd.read_csv(path,encoding = 'utf-8')
    return data
    
def get_rest_data(path):
    reorder_user = getreorderuser()
    data = readData(path)
    newdata = data[~data['user_id'].isin(reorder_user)]
    columns = newdata.columns.values
    return newdata[columns[1:]]
    
def filterthree(newdata):
    groups = newdata.groupby(["user_id","product_id"]).groups
    user_product_id = []
    res = [k for k,v in groups.items() if len(v)>=2]
    newdata["user_product"] = newdata[['user_id', 'product_id']].apply(tuple, axis=1) 
    filterdata = newdata[newdata["user_product"].isin(res)]  
    return filterdata

def handle_data(path):
    newdata = get_rest_data(path)
    filterdata = filterthree(newdata)
    del newdata
    file_name = os.path.basename(path)
    path = pardir+'/data/filter/'+file_name
    print(path)
    columns = filterdata.columns.values
    filterdata[columns[:-1]].to_csv(path,encoding='utf-8',index = False)    
    
if __name__=="__main__":
    filelist = listfiles(combine_dir)
    for file in filelist:
        handle_data(file)
    
