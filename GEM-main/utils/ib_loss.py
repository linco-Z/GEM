import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
import numpy as np

def distmat(X):
    r = torch.sum(X*X)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D) 
    return D

def sigma_estimation(X, Y):
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def GaussianMatrix(X,Y,sigma):
    size1 = X.size()
    size2 = Y.size()
    G = (X*X).sum(-1)
    H = (Y*Y).sum(-1)
    Q = G.unsqueeze(-1).repeat(1,size2[0])
    R = H.unsqueeze(-1).T.repeat(size1[0],1)
    
    
    H = Q + R - 2*X@(Y.T)
    H = torch.clamp(torch.exp(-H/2/sigma**2),min=0)
    
    return H

def CS_QMI(x,y,sigma = None):
    
    N = x.shape[0]
    if not sigma:
        sigma_x = 10*sigma_estimation(x,x)
        sigma_y = 10*sigma_estimation(y,y)
       
        Kx = GaussianMatrix(x,x,sigma_x)
        Ky = GaussianMatrix(y,y,sigma_y)
    
    else:
        Kx = GaussianMatrix(x,x,sigma)
        Ky = GaussianMatrix(y,y,sigma)
    
    self_term1 = torch.trace(Kx@Ky.T)/(N**2)
    
    self_term2 = (torch.sum(Kx)*torch.sum(Ky))/(N**4)
    
    term_a = torch.ones(1,N).to(x.device)
    term_b = torch.ones(N,1).to(x.device)
    cross_term = (term_a@Kx.T@Ky@term_b)/(N**3)
    CS_QMI = -2*torch.log2(cross_term) + torch.log2(self_term1) + torch.log2(self_term2)
    
    return CS_QMI
    
def CS_QMI_normalized(x,y,sigma=None):
    QMI = CS_QMI(x, y, sigma)
    var1 = torch.sqrt(CS_QMI(x, x, sigma))
    var2 = torch.sqrt(CS_QMI(y, y, sigma))
    
    eps = 1e-8
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)
    
    normalized_qmi = QMI / (var1 * var2)
    
    normalized_qmi = torch.clamp(normalized_qmi, min=-10.0, max=10.0)
    
    if torch.isnan(normalized_qmi) or torch.isinf(normalized_qmi):
        print(f"Warning: QMI computation resulted in NaN/Inf. QMI: {QMI}, var1: {var1}, var2: {var2}")
        return torch.tensor(0.0, device=x.device)
    
    return normalized_qmi