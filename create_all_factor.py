import pandas as pd
import time
import os

from getPath import *
pardir = getparentdir()
from commonLib import *

filter_dir = pardir+'/data/filter'

def analyze(file):
    data = readData(file)
    items = pd.DataFrame({'count':data.groupby(['user_id','product_id']).size()}).reset_index()
    user_items = pd.DataFrame({'totalcount':data.groupby(['user_id']).size()}).reset_index()
    res = pd.merge(items, user_items, on = 'user_id')
    item_importance = pd.DataFrame(res['count']/res['totalcount'])
    # item_importance
    item_importance['user_id'] = res['user_id']
    item_importance['product_id'] = res['product_id']
    columns = item_importance.columns.values
    item_importance = item_importance.rename(columns={columns[0]: "ratio"})
    print(item_importance)
    
def analyze_main():
    filelist = listfiles(filter_dir)
    for file in filelist:
        analyze(file)
        break
        
if __name__=="__main__":
    analyze_main()