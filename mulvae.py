import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class MultipleVAE(nn.Module):
    def __init__(self,xyvae,xvae,yvae):
        self,xyvae=xyvae
        self.xvar=xyvae
        self.yvar=yvae
    def forward(self, x,y): 
        

