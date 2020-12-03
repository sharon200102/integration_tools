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
        return {'x': self.xvae(x), 'y': self.yvae(y), 'xy': self.xyvae(torch.cat([x, y], dim=1))}

    """we're trying to minimize two losses, the  reconstruction losses of all VAE's together with the differences 
    between the latent representation of the VAE's """

    def loss_function(self, forward_dict):
        loss_dict = {'xvae_loss': self.xvae.loss_function(*(forward_dict['x'])),
                     'yvae_loss': self.xvae.loss_function(*(forward_dict['y'])),
                     'xyvae_loss': self.xvae.loss_function(*(forward_dict['xy'])),
                     'similarity_loss_x_y': F.mse_loss(forward_dict['x'][4], forward_dict['y'][4]),
                     'similarity_loss_x_xy': F.mse_loss(forward_dict['x'][4], forward_dict['xy'][4]),
                     'similarity_loss_y_xy': F.mse_loss(forward_dict['y'][4], forward_dict['xy'][4])}
        loss_dict['total_loss'] = loss_dict['xvae_loss']['loss'] + loss_dict['yvae_loss']['loss'] + \
                                  loss_dict['xyvae_loss']['loss'] + loss_dict['similarity_loss_x_y'] + loss_dict[
                                      'similarity_loss_x_xy'] + loss_dict['similarity_loss_y_xy']
        return loss_dict
