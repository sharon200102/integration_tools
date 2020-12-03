import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class StandardVAE(nn.Module):
    def __init__(self, layers_structure, activation_fn, kld_coefficient=1):
        """
        :param layers_structure: An iterable which will form the structure of the encoder, As a result, the structure
        of the decoder will be symmetrical.

        :param activation_fn: A Pointer to the activation function that will be used between the layers.
        """

        super(StandardVAE, self).__init__()
        self.activation_fn = activation_fn
        reversed_layers_structure = layers_structure[::-1]  # the reversed list will form the structure of the decoder.
        self.encoding_layers = nn.ModuleList(
            [nn.Linear(layers_structure[i], layers_structure[i + 1]) for i in
             range(0, len(layers_structure) - 2)])  # Don't create the last layer because it needs to be split.

        # Layers for the mean and the variance of the normal distribution.

        self.mu_layer = nn.Linear(layers_structure[-2], layers_structure[-1])

        self.logvar_layer = nn.Linear(layers_structure[-2], layers_structure[-1])

        self.decoding_layers = nn.ModuleList(
            [nn.Linear(reversed_layers_structure[i], reversed_layers_structure[i + 1]) for i in
             range(0, len(reversed_layers_structure) - 1)])
        self.kld_coefficient = kld_coefficient

    def encode(self, x):
        """
        :param x The input to the network, The function will output the corresponding mu, and logvar.
        """
        for encoding_layer in self.encoding_layers:
            x = self.activation_fn(encoding_layer(x))
        return self.mu_layer(x), self.logvar_layer(x)

    def reparameterize(self, mu, logvar):
        """

        :param mu The first output of the encoding function, represents the mean of the latent distribution
        :param logvar: The second output of the encoding function, will be transformed into the distribution variance.
        :return: The latent representation of the input of the network, using the parameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def decode(self, z):
        """

        :param z: the latent representation of the input which is the output of the reparameterize function.
        :return:
        """
        for layer_number, decoding_layer in enumerate(self.decoding_layers):
            if layer_number < len(self.decoding_layers) - 1:
                z = self.activation_fn(decoding_layer(z))
            else:
                z = decoding_layer(z)
        return z

    def forward(self, input):
        """

        :param input: The input to the first layer.
        :return: List which consists of, the reconstructed input,
        the input itself, the mu and the sigma constructing the distribution from which the output was sampled,
        and the latent representation of the input.
        """
        mu, logvar = self.encode(input)  # self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), input, mu, logvar, z]

    def loss_function(self, *args):
        """
        A function the given the forward array of a sample, calculates the loss. :param args: Forwrad array :return:
        A dictionary which describes ever component of the loss i.e Reconstruction_Loss, KLD loss, and the total loss.
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = self.kld_coefficient
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}
