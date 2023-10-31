import copy
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error as skmse

def buildEncoder(dims, dop):
    return nn.Sequential(
            *[nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dop)
                    )
            for i in range(len(dims)-1)]
            )

class AE(nn.Module):
#    def __init__(self, input_dim, latent_dim, hidden_dims=None, dop=0.1, noise_flag=False, trainingArgs, **kwargs):
    def __init__(self, kwargs):
        super(AE, self).__init__()
        self.latent_dim = kwargs["latent_dim"]
        self.noise_flag = kwargs["noise_flag"]
        self.dop = kwargs["dop"]
        self.trainingArgs = kwargs
        
        hidden_dims = kwargs["hidden_dims"]
        input_dim = kwargs["input_dim"]
        latent_dim = kwargs["latent_dim"]
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]

        self.encoder = buildEncoder([input_dim] + hidden_dims + [latent_dim], self.dop)
        self.decoder = buildEncoder([latent_dim] + hidden_dims[::-1], self.dop)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.dop),
            nn.Linear(hidden_dims[0], input_dim)
        )
        
    def initOptim(self):
        # Initializing on server may leading unclear optimizer parameter pointing.
        # Called on the client!
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.trainingArgs['lr'])

    def encode(self, input):
        if self.noise_flag and self.training:
            return self.encoder(input + torch.randn_like(input, requires_grad=False)*0.1 - 0.05)
        else:
            return self.encoder(input)

    def decode(self, z):
        return self.final_layer(self.decoder(z))

    def forward(self, input, **kwargs):
        z = self.encode(input)
        return [input, self.decode(z), z]

    def loss_function(self, *args, **kwargs) -> dict:
        loss = F.mse_loss(args[0], args[1])
        return {'loss': loss, 'recons_loss': loss}

    def sample(self, num_samples, current_device, **kwargs):
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        return self.forward(x)[1]
    
    def trainEpoch(self, s_dataloader, t_dataloader):
        paraSave = copy.deepcopy(list(self.parameters()))
        device = next(self.parameters()).device
#        print(type(s_dataloader), type(t_dataloader))
        for s_batch, t_batch in zip(s_dataloader, t_dataloader):
            self.train()
#            print(type(s_batch))
#            print(s_batch)
        
            s_x = s_batch[0].to(device)
            t_x = t_batch[0].to(device)
        
            s_loss_dict = self.loss_function(*self(s_x))
            t_loss_dict = self.loss_function(*self(t_x))
        
            self.opt.zero_grad()
            loss = s_loss_dict['loss'] + t_loss_dict['loss']
        
            loss.backward()
            self.opt.step()
        
#            loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

        return [b-p for p,b in zip(paraSave, self.parameters())], loss.cpu().detach().item()
        
    def testEpoch(self, s_data, t_data):
        device = next(self.parameters()).device
        self.eval()
        s_x = self(s_data.to(device))
        t_x = self(t_data.to(device))
        s_loss_dict = self.loss_function(*s_x)
        t_loss_dict = self.loss_function(*t_x)
        loss = s_loss_dict['loss'] + t_loss_dict['loss']
        
        return loss.cpu().detach().item()
#        yTest = np.concatenate((s_x[0].cpu().detach().numpy(), t_x[0].cpu().detach().numpy()))
#        yTestPred = np.concatenate((s_x[1].cpu().detach().numpy(), t_x[1].cpu().detach().numpy()))
#        res = {'loss':loss.cpu().detach().item()}
#        res['mse'] = skmse(yTest, yTestPred)
#        res['r2'] = r2_score(yTest, yTestPred)
##        res['pearson'] = pearsonr(yTest, yTestPred)[0]
##        res['spearman'] = spearmanr(yTest, yTestPred)[0]
#
#        return res

