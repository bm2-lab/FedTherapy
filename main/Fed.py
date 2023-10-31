import ae
import os
import data
import copy
import torch
import pickle
import itertools
import numpy as np
#from collections import defaultdict

from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

def mergeDi(container):
    return {k:sum(c[k] for c in container)/len(container) for k in container[0]}
        
def mergeLi(li):
    return sum(li)/len(li)

def dataPreprocess(s_data, t_data, trainSize=0.8, batch_size=64, kfold=None):
    if kfold:
        def F(x):
            kf = KFold(n_splits=kfold[0], random_state=42, shuffle=True)
            kfIter = iter(kf.split(x))
            for _ in range(kfold[1]+1):
                trainIdx, testIdx = next(kfIter)
            return x[trainIdx], x[testIdx]      
    else:
        F = lambda x:train_test_split(x, train_size=trainSize, test_size=1-trainSize, random_state=42)
    s_train, s_test = map(torch.FloatTensor, F(s_data))
    t_train, t_test = map(torch.FloatTensor, F(t_data))
    s_train, t_train = map(TensorDataset, (s_train, t_train))
    exchangeFlag = False
    if len(s_train) > len(t_train):
        exchangeFlag = True
        s_train, t_train = t_train, s_train
    s = DataLoader(
            s_train, 
            sampler = RandomSampler(s_train, True, len(t_train)), 
            batch_size = batch_size, 
            num_workers = 1, 
            pin_memory = True)
    t = DataLoader(
            t_train, 
            batch_size = batch_size, 
            num_workers = 1, 
            pin_memory = True)
    if exchangeFlag:
        s, t = t, s
    return (s, t), (s_test, t_test)

class Server:
    def __init__(self, modelClass, trainArgs):
        self.clientList = []
        self.modelClass = modelClass
        self.trainArgs = trainArgs
        
    def run(self):
        res = []
        if self.trainArgs['kfold']:
            for i in range(self.trainArgs['kfold']):
                self.trainArgs['currentKFold'] = (self.trainArgs['kfold'], i)
                res.append(self.train())
        else:
            res.append(self.train())
        meanScore = mergeLi(res)
        tag = '_'.join(str(self.trainArgs[x]) for x in ("lr", "dop", "noise_flag"))
        print(tag+ '\n' + str(meanScore))
        with open('result.txt', 'a') as file:
            file.write(tag+ '\n' + str(meanScore) + '\n')

    def train(self):
        # init self model and clients
        self.model = self.modelClass(self.trainArgs)
        for c in self.clientList:
            c.init(copy.deepcopy(self.model), self.trainArgs)
        # training
        self.model = self.model.to(self.trainArgs["device"])
        train_score = []
        test_score = []
        bestScore = 1e10
        tag = '_'.join(str(self.trainArgs[x]) for x in ("lr", "dop", "noise_flag", 'currentKFold'))
        print(tag)
        for epoch in range(int(self.trainArgs['train_num_epochs'])):
            grads, trainLoss = zip(*[c.trainEpoch() for c in self.clientList])
            grads = list(map(mergeLi, zip(*grads)))
            trainLoss = mergeLi(trainLoss)
            for para, grad in zip(self.model.parameters(), grads):
                para.data += grad
            for c in self.clientList:
                c.modelUpdate(self.model.state_dict())
            test = mergeLi([c.test() for c in self.clientList])
            train_score.append(trainLoss)
            test_score.append(test)
            if test < bestScore:
                bestIdx = epoch
                bestScore = test
                torch.save(self.model.state_dict(), os.path.join(self.trainArgs['model_save_folder'], f'ae_{tag}.pt'))
            elif self.trainArgs['early_stop'] and epoch - bestIdx > self.trainArgs['early_stop']:
                break            
            print(epoch, trainLoss, test, bestIdx)
        self.model = self.model.cpu()
        self.model.load_state_dict(torch.load(os.path.join(self.trainArgs['model_save_folder'], f'ae_{tag}.pt')))
        with open(os.path.join(self.trainArgs['model_save_folder'], f'ae_{tag}.pkl'), 'bw') as file:
            pickle.dump((self.model.encoder, train_score, test_score), file)
        return bestScore
    
class Client:
    def __init__(self, publicData, privateData):
        self.publicData = publicData
        self.privateData = privateData
        
    def init(self, model, trainArgs):
        self.trainArgs = trainArgs
        (s_train, t_train), (s_test, t_test) = dataPreprocess(self.publicData, self.privateData, kfold=trainArgs['currentKFold'])
        self.s_train = s_train
        self.t_train = t_train
        self.s_test = s_test
        self.t_test = t_test
        model.initOptim()
        self.model = model.to(trainArgs["device"])
        
    def trainEpoch(self):
        return self.model.trainEpoch(self.s_train, self.t_train)
    
    def test(self):
        return self.model.testEpoch(self.s_test, self.t_test)
    
    def modelUpdate(self, modelStateDict):
        self.model.load_state_dict(modelStateDict)
        
class FederatedSimulator:
    def __init__(self, modelClass, trainArgs, n_client=3):
        self.modelClass = modelClass
        self.trainArgs = trainArgs
        self.n_client = n_client
        
    def run(self):
        self.server = Server(self.modelClass, self.trainArgs)
        ds = data.FedTherapyDataSet(seed=0, batchSize=64, num_workers=1, shuffle=True)
        publicData, privateData = ds.load_fed_unlabel_data(self.n_client)
        for pri in privateData:
            self.server.clientList.append(Client(publicData, pri))
#        self.server.init()
        self.server.run()
        
def main(di):
    trainArgs = {
            "lr": 0.001, 
            "noise_flag": True,
            "dop": 0.1,
            "hidden_dims": [512, 256, 128],
            "input_dim": 2009,
            "latent_dim": 64,
            "early_stop": 20,
            "model_save_folder": './ae0',
            "train_num_epochs": 1000,
            "batch_size": 64, 
            "device": 0, 
            'kfold': 5, 
            }
    trainArgs.update(di)
#    print(trainArgs)
    fs = FederatedSimulator(ae.AE, trainArgs)
    fs.run()
    
if __name__ == "__main__":
    
    params_grid = {
        "dop": [0., 0.1], 
        "lr": [1e-1, 1e-3, 1e-4, 1e-5,], 
#        "lr": [1e-4,], 
        "noise_flag": [True, False]
    }
    params_grid = {
        "dop": [0.], 
        "lr": [1e-4], 
        "noise_flag": [True], 
        "early_stop": [2000],
        "model_save_folder": ['./ae_fedGraph'],
        "train_num_epochs": [400],
    }
    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for param_dict in update_params_dict_list:
        main(param_dict)
