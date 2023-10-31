import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True

class CCL1(nn.Module):
    def __init__(self, inputDim):
        super(CCL1, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 2048), 
                nn.BatchNorm1d(2048),
#                nn.ReLU(inplace=True),
                nn.Linear(2048, 256), 
                nn.BatchNorm1d(256),
#                nn.ReLU(inplace=True),
                nn.Linear(256, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),
                nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layers(x)
    
class CCL2(nn.Module):
    def __init__(self, inputDim):
        super(CCL2, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 2048), 
                nn.BatchNorm1d(2048),
#                nn.ReLU(inplace=True),
                nn.Linear(2048, 512), 
                nn.BatchNorm1d(512),
#                nn.ReLU(inplace=True),
                nn.Linear(512, 128), 
                nn.BatchNorm1d(128),
#                nn.ReLU(inplace=True),
                nn.Linear(128, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),)
#                nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layers(x)
    
class CCL2EndRelu(nn.Module):
    def __init__(self, inputDim):
        super(CCL2EndRelu, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 2048), 
                nn.BatchNorm1d(2048),
#                nn.ReLU(inplace=True),
                nn.Linear(2048, 512), 
                nn.BatchNorm1d(512),
#                nn.ReLU(inplace=True),
                nn.Linear(512, 128), 
                nn.BatchNorm1d(128),
#                nn.ReLU(inplace=True),
                nn.Linear(128, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),
                nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layers(x)
        
class CCL2AllRelu(nn.Module):
    def __init__(self, inputDim):
        super(CCL2AllRelu, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 2048), 
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 512), 
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128), 
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),
                nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layers(x)    
            
class CCLAllReluDroup(nn.Module):
    def __init__(self, inputDim):
        super(CCLAllReluDroup, self).__init__()
        self.outputDim = 64
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 2048), 
                nn.BatchNorm1d(2048),
                nn.Dropout(0.1),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 512), 
                nn.BatchNorm1d(512),
                nn.Dropout(0.1),
                nn.ReLU(inplace=True),
                nn.Linear(512, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),
                nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layers(x)    
    
class CCLBig(nn.Module):
    def __init__(self, inputDim):
        super(CCLBig, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 2048), 
                nn.BatchNorm1d(2048),
#                nn.ReLU(inplace=True),
                nn.Linear(2048, 2048), 
                nn.BatchNorm1d(2048),
#                nn.ReLU(inplace=True),
                nn.Linear(2048, 2048), 
                nn.BatchNorm1d(2048),
#                nn.ReLU(inplace=True),
                nn.Linear(2048, 512), 
                nn.BatchNorm1d(512),
#                nn.ReLU(inplace=True),
                nn.Linear(512, 128), 
                nn.BatchNorm1d(128),
#                nn.ReLU(inplace=True),
                nn.Linear(128, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),)
#                nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layers(x)
    
class Drug1(nn.Module):
    def __init__(self, inputDim):
        super(Drug1, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 100), 
                nn.BatchNorm1d(100),
#                nn.Dropout(0.1),
                nn.Linear(100, 50), 
                nn.BatchNorm1d(50),
#                nn.Dropout(0.1),
                nn.Linear(50, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),)
#                nn.ReLU(inplace=True),)
    def forward(self, x):
        return self.layers(x)

class Drug2(nn.Module):
    def __init__(self, inputDim):
        super(Drug2, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 100), 
                nn.BatchNorm1d(100),
#                nn.ReLU(inplace=True), 
                nn.Linear(100, 50), 
                nn.BatchNorm1d(50),
#                nn.ReLU(inplace=True), 
                nn.Linear(50, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),)
#                nn.ReLU(inplace=True),)
    def forward(self, x):
        return self.layers(x)
    
class Drug2AllRelu(nn.Module):
    def __init__(self, inputDim):
        super(Drug2AllRelu, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 100), 
                nn.BatchNorm1d(100),
                nn.ReLU(inplace=True), 
                nn.Linear(100, 50), 
                nn.BatchNorm1d(50),
                nn.ReLU(inplace=True), 
                nn.Linear(50, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),
                nn.ReLU(inplace=True),)
    def forward(self, x):
        return self.layers(x)
    
class Drug2EndRelu(nn.Module):
    def __init__(self, inputDim):
        super(Drug2EndRelu, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 100), 
                nn.BatchNorm1d(100),
#                nn.ReLU(inplace=True), 
                nn.Linear(100, 50), 
                nn.BatchNorm1d(50),
#                nn.ReLU(inplace=True), 
                nn.Linear(50, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),
                nn.ReLU(inplace=True),)
    def forward(self, x):
        return self.layers(x)

class DrugBig(nn.Module):
    def __init__(self, inputDim):
        super(DrugBig, self).__init__()
        self.outputDim = 32
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 512), 
                nn.BatchNorm1d(512),
#                nn.ReLU(inplace=True), 
                nn.Linear(512, 128), 
                nn.BatchNorm1d(128),
#                nn.ReLU(inplace=True), 
                nn.Linear(128, self.outputDim), 
                nn.BatchNorm1d(self.outputDim),)
#                nn.ReLU(inplace=True),)
    def forward(self, x):
        return self.layers(x)

class Pred1(nn.Module):
    def __init__(self, inputDim):
        super(Pred1, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 64), 
                nn.BatchNorm1d(64),
#                nn.ReLU(inplace=True),
                nn.Linear(64, 32), 
                nn.BatchNorm1d(32),
#                nn.ReLU(inplace=True),
                nn.Linear(32, 1))
    def forward(self, x):
        return self.layers(x)
    
class PredRelu(nn.Module):
    def __init__(self, inputDim):
        super(PredRelu, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 64), 
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32), 
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1))
    def forward(self, x):
        return self.layers(x)
    
class PredBin(nn.Module):
    def __init__(self, inputDim):
        super(PredBin, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(inputDim, 64), 
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32), 
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 2))
    def forward(self, x):
        x = self.layers(x)
        return x[:,[0]], x[:,[1]].sigmoid()

class Module(nn.Module):
    def __init__(self, cclModel, drugModel, predModel):
        super(Module, self).__init__()
        self.cm = cclModel
        self.dm = drugModel
        self.pm = predModel
        
    def forward(self, Xc, Xd):
        return self.pm(torch.cat((self.cm(Xc), self.dm(Xd)), dim=1))
    
class ModuleC(nn.Module):
    def __init__(self, cclModel, predModel):
        super(ModuleC, self).__init__()
        self.cm = cclModel
        self.pm = predModel
        
    def forward(self, X):
        return self.pm(self.cm(X))
