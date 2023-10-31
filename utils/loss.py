import torch
from torch import nn

def spearmanT(x, y):
    n = x.size(0)
    r1 = torch.zeros(n, dtype=torch.float)
    r2 = torch.zeros(n, dtype=torch.float)
    r1[x.flatten().sort().indices] = r2[y.flatten().sort().indices] = torch.FloatTensor(range(n))
    return 1 - 6 * ((r1-r2)**2).sum()/(n*n*n-n)    

def r2T(y_true, y_hat):
    return 1 - ((y_true-y_hat)**2).sum()/((y_true-y_true.mean())**2).sum()

def pearsonT(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return (x*y).sum() / ((x**2).sum().sqrt() * ((y**2).sum().sqrt()))

def mseT(x, y):
    return ((x.flatten()-y.flatten())**2).mean()

def bceT(ytrue, yhat):
    ytrue = ytrue.flatten()
    yhat = yhat.flatten()
    return (- (1-ytrue) * (1-yhat).log() - ytrue * yhat.log()).mean()

def PearsonMat(mat):
    l = mat.size(0)
    mat = mat - mat.mean(dim=1).reshape((l, 1))
    triMask = torch.triu(torch.ones((l,l)).bool(), diagonal=1).flatten()
    x = mat.repeat((l, 1)).reshape((l*l, -1))[triMask] 
    y = mat.repeat((1, l)).reshape((l*l, -1))[triMask]
    
    res = torch.zeros(l*l)
    res[triMask] = (x*y).sum(dim=1) / ((x**2).sum(dim=1).sqrt() * ((y**2).sum(dim=1).sqrt()))
    res = res.reshape((l, l))
    return res + res.T + torch.eye(l)

def JaccardMat(mat):
    l = mat.size(0)
    triMask = torch.triu(torch.ones((l,l)).bool(), diagonal=1).flatten()
    m = torch.cat((mat.repeat((l, 1)).reshape((l*l, -1, 1))[triMask], 
                   mat.repeat((1, l)).reshape((l*l, -1, 1))[triMask]),
                  dim=2) # high time cost
            
    res = torch.zeros((l*l))
    res[triMask] = (m.min(dim=2).values.sum(dim=1) + 1e-40) / \
                   (m.max(dim=2).values.sum(dim=1) + 1e-40) # high time cost
    res = res.reshape((l, l))
    return res + res.T + torch.eye(l)

def CosMat(mat):
    mod = (mat * mat).sum(dim=1).sqrt()
    x, y = torch.meshgrid(mod, mod)
    return mat.mm(mat.T) / x / y

class CoralLoss(nn.Module):
    def __init__(self):
        super(CoralLoss, self).__init__()
    
    def forward(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)
    
        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)
    
        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4*d*d)
        return loss
    
class MMDrbfLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, **kwargs):
        super(MMDrbfLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul

    def guassian_kernel(self, source, target, kernel_mul, kernel_num):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        bandwidth = L2_distance.data.sum() / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [(-L2_distance / bandwidth_temp).exp()
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num)
        XX = kernels[:batch_size, :batch_size].mean()
        YY = kernels[batch_size:, batch_size:].mean()
        XY = kernels[:batch_size, batch_size:].mean()
        YX = kernels[batch_size:, :batch_size].mean()
        return (XX + YY - XY - YX).mean()
        