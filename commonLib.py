import os
import numpy as np
import pandas as pd

def listfiles(datadir):
    list_dirs = os.walk(datadir) 
    filepath_list = []
    for root, dirs, files in list_dirs:
        for f in files:
            filepath_list.append(os.path.join(root,f))
    return filepath_list
    
def readData(path):
    data = pd.read_csv(path,encoding = 'utf-8')
    return data