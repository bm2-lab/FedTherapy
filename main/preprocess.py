import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

if sys.platform != 'win32':
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt

import utils

def getLim(arr, raid):
    arr = sorted(arr)
    r = int(raid*len(arr))
    return arr[r], arr[-r]

def norm(mat):
    mat = (mat - mat.mean(0)) / mat.std(0)
    mat[np.isnan(mat)] = 0
    return mat

ccleDataPath = '../data/rawdata'
savePath = '../data/processeddata'

expDi = {}
dr = []
cclGeneNames = []
for name in os.listdir(ccleDataPath):
    if name.startswith('CCLE'):
        df = pd.read_csv(os.path.join(ccleDataPath, name), header=0)
        if not cclGeneNames:
            cclGeneNames = list(df.columns)[3:]
        for line in df.itertuples():
            expDi[line[2]] = np.array(line[4:])
    elif name.startswith('GDSC'):
        df = pd.read_csv(os.path.join(ccleDataPath, name), header=0)
        for line in df.itertuples():
            dr.append((line[3], line[5], line[7], line[8]))
                
expLenSet = set(map(len, expDi.values()))
assert len(expLenSet) == 1
assert len(cclGeneNames) == next(iter(expLenSet))

#utils.pklSave(cclGeneNames, os.path.join(savePath, 'cclGeneNames.pkl'))

drugDict = utils.pklLoad(os.path.join(savePath, 'drugDict.pkl'))
patExpDi = utils.pklLoad(os.path.join(savePath, 'patExpDi.pkl'))
geneIdx = utils.pklLoad(os.path.join(savePath, 'geneIdx.pkl'))
patResponseMat = utils.pklLoad(os.path.join(savePath, 'patResponseMat.pkl'))

with open(os.path.join(ccleDataPath, 'tpm_new.csv')) as file:
    file.readline()
    patNames = [x.split(',')[0].strip('"') for x in file ]

cclGeneNameSet = set(cclGeneNames)
assert all(x in cclGeneNameSet for x in patNames)

idxDi = {x:i for i,x in enumerate(cclGeneNames)}
idxs = [idxDi[x] for x in patNames]
expDi = {k:v[idxs][geneIdx] for k,v in expDi.items()}

dr = [x for x in dr if x[1] in drugDict]

utils.pklSave(expDi, os.path.join(savePath, 'cclExpDi.pkl'))
utils.pklSave(dr, os.path.join(savePath, 'cclDrLi.pkl'))


#pca = PCA(n_components=2)
#cclMat = norm(np.array(list(expDi.values())))
#patMat = norm(np.array(list(patExpDi.values())))
#
#p = pca.fit_transform(cclMat)
#plt.scatter(p[:,0], p[:,1], alpha=0.25, label='source')  
#plt.legend()
#plt.xlim(getLim(p[:,0], 0.01))
#plt.ylim(getLim(p[:,1], 0.01))
#plt.title(f'source')
#plt.savefig(f'./source.jpg')
#plt.show()
#plt.cla()
#
#p = pca.fit_transform(patMat)
#plt.scatter(p[:,0], p[:,1], alpha=0.25, label='target')    
#plt.legend()
#plt.xlim(getLim(p[:,0], 0.01))
#plt.ylim(getLim(p[:,1], 0.01))
#plt.title(f'target')
#plt.savefig(f'./target.jpg')
#plt.show()
#plt.cla()
#
#p = pca.fit_transform(np.concatenate((cclMat, patMat), axis=0))
#splitPoint = cclMat.shape[0]
#plt.scatter(p[:splitPoint,0], p[:splitPoint,1], alpha=0.25, label='source')
#plt.scatter(p[splitPoint:,0], p[splitPoint:,1], alpha=0.25, label='target')    
#plt.legend()
#plt.xlim(getLim(p[:,0], 0.01))
#plt.ylim(getLim(p[:,1], 0.01))
#plt.title(f'st')
#plt.savefig(f'./st.jpg')
#plt.show()
#plt.cla()


