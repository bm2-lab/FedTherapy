import sys
sys.path.append("..")

import random
import numpy as np
from os import path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
from torch.utils.data import DataLoader, TensorDataset
import utils

srcDataPath = "../data/processeddata"

class NameGetDict(dict):
    """
    Data structure for quickly getting data from dict by plurality of keys.
    Accessed by format as NameGetDict[keys].
    'keys' should be iterable object which contains keys.
    Return numpy array.
    Except for the dict[key] form, the method of the original dict can still be normally used.
    """
    def __init__(self, di, dtype='float32'):
        super().__init__(di)
        self.idxDi = {k:i for i,k in enumerate(self.keys())}
        self.mat = np.array(list(self.values()), dtype=dtype)
    def __getitem__(self, idx):
        idxDi = self.idxDi
        return self.mat[np.array([idxDi[i] for i in idx])]

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def npArrayShuffle(array, dim=0, inplace=False):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    idx = np.arange(array.shape[dim])
    np.random.shuffle(idx)
    if inplace:
        array[:] = array[(slice(None),)*dim + (idx,)]
        return array
    else:
        return array[(slice(None),)*dim + (idx,)]

def discretization(array, threshold=None):
    nparray = array if isinstance(array, np.ndarray) else np.array(array, dtype="float32")
    if threshold is None:
        threshold = np.median(nparray)
    res = np.zeros(nparray.shape, dtype=nparray.dtype)
    res[nparray>threshold] = 1
    return res

def norm(mat):
    mat = (mat - mat.mean(0)) / mat.std(0)
    mat[np.isnan(mat)] = 0
    return mat

def expDiPreprocess(fileName):
    return utils.pklLoad(path.join(srcDataPath, fileName))
#    di = utils.pklLoad(path.join(srcDataPath, fileName))
#    mat = np.array(list(di.values()), dtype='float32')
#    min_ = mat.min(0)
#    mat = (mat - min_) / (mat.max(0) - min_)
#    return {k:line for k, line in zip(di, mat)}

class FedTherapyDataSet:
    def __init__(self, fmt='norm', seed=None, batchSize=128, num_workers=4, shuffle=False):
        self.seed = seed
        if seed is not None:
            setSeed(seed)
        self.batchSize = batchSize
        self.num_workers = num_workers
        
        self.drugDict = NameGetDict(utils.pklLoad(path.join(srcDataPath, "drugDict.pkl")))
        self.cclExpDi = NameGetDict(expDiPreprocess("cclExpDi.pkl"))
        self.patExpDi = NameGetDict(expDiPreprocess("patExpDi.pkl"))
        self.cclDrMat = np.array([[x[0], x[1].replace('5-', ''), x[2],x[3]] for x in utils.pklLoad(path.join(srcDataPath, "cclDrLi.pkl"))])
        self.patDrMat = np.array(utils.pklLoad(path.join(srcDataPath, "patResponseMat.pkl")))

        if fmt=='norm':
            self.cclExpDi.mat = norm(self.cclExpDi.mat)
            self.patExpDi.mat = norm(self.patExpDi.mat)
            self.drugDict.mat = norm(self.drugDict.mat)
        
        if shuffle:
            npArrayShuffle(self.cclDrMat, True)
            npArrayShuffle(self.patDrMat, True)
        
        self.toDataLoaderF = lambda x,**kargs:DataLoader(
                TensorDataset(*map(torch.FloatTensor, x)), 
                batch_size = self.batchSize, 
                num_workers = self.num_workers, 
                pin_memory = True, 
                **kargs)
        
    def load_fed_unlabel_data(self, n_fedClient=3):
        p = len(self.patExpDi.mat) / n_fedClient
        sep = [round(p * i) for i in range(n_fedClient+1)]
        return self.cclExpDi.mat, [self.patExpDi.mat[sep[i]:sep[i+1]] for i in range(n_fedClient)]
        
    def load_unlabel_data(self, trainSize=0.8):
        def process(data):
            x_train, x_test = train_test_split(data, train_size=trainSize, test_size=1-trainSize, random_state=self.seed)
#            return self.toDataLoaderF([x_train], drop_last=True), self.toDataLoaderF([x_test], drop_last=True)
            return self.toDataLoaderF([data], drop_last=True), self.toDataLoaderF([x_test], drop_last=True)
        # Xs_train, Xs_test, Xt_train, Xt_test
        return process(self.cclExpDi.mat) + process(self.patExpDi.mat)
        
    def load_finetune_data(self, trainSize=0.8, y_label="IC50", drug=None, kFlodNSplit=5):
        
        assert drug in self.drugDict, f"Can't found drug {drug}."
        assert y_label.upper() in ("IC50", "AUC"), "y_label only support IC50 or AUC"
        
        ccl_y_index = 2 if y_label.upper()=="IC50" else 3
        cclMat = self.cclDrMat[self.cclDrMat[:,1]==drug][:,(0,1,ccl_y_index)]
        patMat = self.patDrMat[self.patDrMat[:,1]==drug]
        cclX = np.array(self.cclExpDi[(x[0] for x in cclMat)], dtype='float32')
        ccly = discretization(cclMat[:,-1])
        
        kfoldCclData = []
        kfold = StratifiedKFold(n_splits=kFlodNSplit, random_state=self.seed, shuffle=True)
        for train_index, test_index in kfold.split(cclX, ccly):
            train_labeled_ccl_dataloader = self.toDataLoaderF(
                    [cclX[train_index], 
                     ccly[train_index].flatten()], 
                    drop_last=True)
            test_labeled_ccl_dataloader = self.toDataLoaderF(
                    [cclX[test_index], 
                     ccly[test_index].flatten()], 
                    drop_last=True)
            kfoldCclData.append((train_labeled_ccl_dataloader, test_labeled_ccl_dataloader))
        
        patDs = TensorDataset(
            torch.from_numpy(self.patExpDi[patMat[:,0]]),
            torch.from_numpy(discretization(patMat[:,-1])))
        labeled_tcga_dataloader = DataLoader(patDs, batch_size=self.batchSize)
    
        return [(train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader) for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader in kfoldCclData]

    
    def load_labeled_data(self, trainSize, y_label="IC50", drug=None):
        assert drug is None or drug in self.drugDict, f"Can't found drug {drug}."
        assert y_label.upper() in ("IC50", "AUC"), "y_label only support IC50 or AUC"
        
        ccl_y_index = 2 if y_label.upper()=="IC50" else 3
        def process(mat, mappingDict, discretize=False):
            y = np.array(mat[:,2], dtype=np.float32)
#            if discretize:
#                y = np.array(y<np.median(y), dtype=np.float32) # np.median(y)
            return self.toDataLoaderF([mappingDict[mat[:,0]], 
                                       self.drugDict[mat[:,1]], 
                                       y])
        if drug: # single drug
            cclMat = self.cclDrMat[self.cclDrMat[:,1]==drug, (0,1,ccl_y_index)]
            patMat = self.patDrMat[self.patDrMat[:,1]==drug]
        else: # muti drug, mut & drug-fingerprint, batch-split
            cclMat = self.cclDrMat[:,(0,1,ccl_y_index)]
            patMat = self.patDrMat
        y = np.array(cclMat[:,2], dtype=np.float32)
        cclMat[:,2] = np.array(y<np.median(y), dtype=np.float32)
        s_train_mat, s_test_mat = train_test_split(cclMat, train_size=trainSize, random_state=self.seed)
        return process(s_train_mat, self.cclExpDi, True), \
               process(s_test_mat, self.cclExpDi, True), \
               process(patMat, self.patExpDi)

    def load_drugSplited_pat_data(self):
        def process(mat, mappingDict):
            y = np.array(mat[:,2], dtype=np.float32)
            return [torch.FloatTensor(x) 
                    for x in [mappingDict[mat[:,0]], self.drugDict[mat[:,1]], y]]
        idxDi = {}
        for d in self.drugDict:
            currentIdx = self.patDrMat[:,1]==d
            if currentIdx.any():
                idxDi[d] = process(self.patDrMat[currentIdx], self.patExpDi)
        return idxDi
        
        