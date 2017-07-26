import numpy as np
import os
import pandas as pd

from getPath import *
pardir = getparentdir()

def disk_shuffle(filename_in, filename_out, header=True, iterations =3, CHUNK_SIZE = 10000, SEP=','):
    for i in range(iterations):
        with open(filename_in, 'r') as R:
            iterator = pd.read_csv(R, chunksize=CHUNK_SIZE)
            for n, df in enumerate(iterator):
                if n==0 and header:
                    header_cols =SEP.join(df.columns)+'\n'
                df.iloc[np.random.permutation(len(df))].to_csv(pardir+'/data/shuffle/'+str(n)+'_chunk.csv', index=False, header=False, sep=SEP)
        ordering = list(range(0,n+1))
        np.random.shuffle(ordering)
        with open(filename_out, 'w') as W:
            if header:
                W.write(header_cols)
            for f in ordering:
                with open(pardir+'/data/shuffle/'+str(f)+'_chunk.csv', 'r') as R:
                    for line in R:
                        W.write(line)
                os.remove(pardir+'/data/shuffle/'+str(f)+'_chunk.csv')
        filename_in = filename_out
        CHUNK_SIZE = int(CHUNK_SIZE / 2)
import os

if __name__=="__main__":
    source = pardir+'/data/train.csv'
    target = pardir+'/data/shuffled/shuffle_orders'
    for i in range(20):
        disk_shuffle(source, filename_out=target+str(i)+'.csv', header=True)
    