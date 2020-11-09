import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class MultipleVAE(nn.Module):
    def __init__(self, xyvae, xvae, yvae):
        super(MultipleVAE, self).__init__()
        """The model consists of three VAE's, one for every possible permutation of a sample (only x is given,
        only y is given, both are given) """
        self.xyvae = xyvae
        self.xvae = xvae
        self.yvae = yvae

        """Given a complete sample(x and y) forward all VAE's"""
    def forward(self, x, y):
        return {'x': self.xvae(x), 'y': self.yvae(y), 'xy': self.xyvae(torch.cat([x, y]))}

    """we're trying to minimize two losses, the  reconstruction losses of all VAE's together with the differences 
    between the latent representation of the VAE's """

    def loss_function(self, forward_dict):
        l_x = self.xvae.loss_function(forward_dict['x'])['loss']
        l_y = self.xvae.loss_function(forward_dict['y'])['loss']
        l_xy = self.xvae.loss_function(forward_dict['xy'])['loss']
        similarity_loss_x_y = F.mse_loss(forward_dict['x'][0], forward_dict['y'][0])
        similarity_loss_x_xy = F.mse_loss(forward_dict['x'][0], forward_dict['xy'][0])
        similarity_loss_y_xy = F.mse_loss(forward_dict['y'][0], forward_dict['xy'][0])
        return l_x+l_y+l_xy+similarity_loss_x_xy+similarity_loss_x_y+similarity_loss_y_xy