import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from .ProjectionModels.stdvae import StandardVAE
from .Constants import activation_fn_dict,optimizers_dict


class MultipleVAE(LightningModule):
    def __init__(self, config: dict, core_parameters: dict):
        super().__init__()
        """The model consists of three VAE's, one for every possible permutation of a sample (only x is given,
        only y is given, both are given) """
        default_config = {'latent_layer_size': 10, 'klb_coefficient': 1, 'learning_rate': 0.01}
        default_core_parameters = {'activation_fn': 'relu', 'optimizer': 'adam'}

        config = self._set_default_parameters(config,default_config)
        core_parameters = self._set_default_parameters(core_parameters,default_core_parameters)

        xy_architecture = core_parameters.get('xy_architecture')
        x_architecture = core_parameters.get('x_architecture')
        y_architecture = core_parameters.get('y_architecture')
        activation_fn = activation_fn_dict[core_parameters.get('activation_fn', 'relu')]()
        self.optimizer = optimizers_dict[core_parameters.get('optimizer', 'adam')]

        latent_layer_size = config.get('latent_layer_size', 10)
        klb_coefficient = config.get('klb_coefficient', 1)

        self.lr = config.get('learning_rate', 0.01)

        self.xyvae = StandardVAE(xy_architecture + [latent_layer_size], activation_fn,
                                 kld_coefficient=klb_coefficient)
        self.xvae = StandardVAE(x_architecture + [latent_layer_size], activation_fn,
                                kld_coefficient=klb_coefficient)
        self.yvae = StandardVAE(y_architecture + [latent_layer_size], activation_fn,
                                kld_coefficient=klb_coefficient)

    """Given a complete sample(x and y) forward all VAE's"""

    def forward(self, x, y):
        return {'x': self.xvae(x), 'y': self.yvae(y), 'xy': self.xyvae(torch.cat([x, y], dim=1))}

    """we're trying to minimize two losses, the  reconstruction losses of all VAE's together with the differences 
    between the latent representation of the VAE's """

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        field0_batch, field1_batch = batch['FIELD0'], batch['FIELD1']
        # If for some reason the batch has more dimensions than expected, squeeze it.
        if len(field0_batch.shape) > 2:
            field0_batch, field1_batch = batch['FIELD0'].squeeze(), batch['FIELD1'].squeeze()

        forward_dict = self(field0_batch, field1_batch)
        # Computes loss
        loss_dict = self.loss_function(forward_dict)
        return {'loss': loss_dict['total_loss']}

    def validation_step(self, batch, batch_index):
        field0_batch, field1_batch = batch['FIELD0'].squeeze(), batch['FIELD1'].squeeze()
        forward_dict = self(field0_batch, field1_batch)
        # Computes reconstruction loss and according to it decide whether to stop.
        loss_dict = self.loss_function(forward_dict)
        validation_batch_loss = float(loss_dict['xvae_loss']['Reconstruction_Loss'] + \
                                      loss_dict['xyvae_loss']['Reconstruction_Loss'] + \
                                      loss_dict['yvae_loss']['Reconstruction_Loss'])

        return {'loss': validation_batch_loss}

    def validation_epoch_end(self, val_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_outputs]).mean()
        self.log('val_loss', avg_val_loss, prog_bar=True)

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


    @classmethod
    def get_hyper_parameter_search_dict(cls):
        return {'latent_layer_size':'linear_scale','klb_coefficient':'log_scale','learning_rate':'log_scale'}
