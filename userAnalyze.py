import pandas as pd
from getPath import *
pardir = getparentdir()
from commonLib import *
user_dir = pardir+'/data/user'
combine_dir= pardir+'/data/combine/'

def readrawData(path):
    data = pd.read_csv(path,encoding = 'utf-8')
    return data

def combinefile():
    filelist = listfiles(user_dir)
    length = len(filelist)
    tempfiles = []
    for i in range(length):
        if i%10000==0:
            index = int(i/10000)
            filepath = combine_dir+str(index)+'.csv'
            data = readrawData(filelist[i])
            data.to_csv(filepath,encoding='utf-8',mode = 'w', index = False)
            continue
        data = readrawData(filelist[i])
        data.to_csv(filepath,encoding='utf-8',mode = 'a',header=False, index = False)

def readData(path):
    data = pd.read_csv(path,encoding = 'utf-8')
    groups = data.groupby(['user_id','product_id']).groups
    users = groups.keys()
    exceptuser = set()
    user_set = set([x for x,_ in users])
    for user in users:
        if len(groups[user]) == 1:
            exceptuser.add(user[0])
    reorderuser = user_set-exceptuser
    return reorderuser
   
def get_reorder_user():
    reorder_user = []
    filelist = listfiles(combine_dir)
    reorderuser = []
    for file in filelist:
        temp = readData(file)
        reorder_user+=list(temp)
    write_reorder_user(reorder_user)
    return reorder_user
    
def write_reorder_user(reorder_user):
    f = open(pardir+'/data/reorderuser.csv','w',encoding = 'utf-8')
    reorder_user = [str(t) for t in reorder_user]
    line = ','.join(reorder_user)
    f.writelines(line)
    f.close()
  
if __name__=="__main__":
    # combinefile()
    # print(get_reorder_user())
    get_reorder_user()  