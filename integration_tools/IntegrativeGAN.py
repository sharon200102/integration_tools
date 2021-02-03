import torch.nn as nn
import pytorch_lightning as pl
from .Constants import activation_fn_dict, optimizers_dict
import torch
X_LABELS=0
Y_LABELS=1
XY_LABELS=2

class Generator(nn.Module):
    def __init__(self, layers_structure, activation_fn):
        """
                :param layers_structure: An iterable which will form the structure of the generator

                :param activation_fn: A Pointer to the activation function that will be used between the layers.
        """

        super().__init__()

        self.activation_fn = activation_fn
        modules = []

        for i in range(0, len(layers_structure) - 1):
            if i == len(layers_structure) - 2:
                modules.extend(self.block(layers_structure[i], layers_structure[i + 1], activation=False))
            else:
                modules.extend(self.block(layers_structure[i], layers_structure[i + 1]))
        self.modules = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def block(self, in_feat, out_feat, activation=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if activation:
            layers.append(self.activation_fn)
        return layers


class Discriminator(nn.Module):
    def __init__(self, layers_structure, activation_fn):
        super().__init__()

        self.activation_fn = activation_fn
        modules = []

        for i in range(0, len(layers_structure) - 1):
            if i == len(layers_structure) - 2:
                modules.extend(self.block(layers_structure[i], layers_structure[i + 1], activation=False))
            else:
                modules.extend(self.block(layers_structure[i], layers_structure[i + 1]))

        self.modules_list = nn.ModuleList(modules)
    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

    def block(self, in_feat, out_feat, activation=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if activation:
            layers.append(self.activation_fn)
        return layers


class IntegrativeGAN(pl.LightningModule):
    def __init__(self, config: dict, core_parameters: dict):
        super().__init__()
        discriminator_output_size = 3
        xgenerator_architecture = core_parameters.get('xgenerator_architecture')
        ygenerator_architecture = core_parameters.get('ygenerator_architecture')
        discriminator_architecture = core_parameters.get('discriminator_architecture')

        activation_fn = activation_fn_dict[core_parameters.get('activation_fn', 'relu')]()
        self.optimizer = optimizers_dict[core_parameters.get('optimizer', 'adam')]

        latent_layer_size = core_parameters.get('latent_layer_size')
        self.lr = config.get('learning_rate', 0.01)

        self.xgenerator = Generator(xgenerator_architecture + [latent_layer_size], activation_fn)
        self.ygenerator = Generator(ygenerator_architecture + [latent_layer_size], activation_fn)
        self.discriminator = Discriminator(
            [latent_layer_size] + discriminator_architecture + [discriminator_output_size],activation_fn)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        return {'x': self.xgenerator(x), 'y': self.ygenerator(y)}

    def configure_optimizers(self):
        x_gen_optimizer = self.optimizer(self.xgenerator.parameters(), lr=self.lr)
        y_gen_optimizer = self.optimizer(self.ygenerator.parameters(), lr=self.lr)
        discriminator_optimizer = self.optimizer(self.discriminator.parameters(), lr=self.lr)

        return [discriminator_optimizer, x_gen_optimizer, y_gen_optimizer], []

    def adversarial_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def training_step(self,batch,batch_idx,optimizer_idx):
        raw_samples,ground_truth_representation = batch
        ground_truth_representation=ground_truth_representation[0]
        x, y = raw_samples['FIELD0'], raw_samples['FIELD1']
        b_size = len(batch)
        label = torch.full((b_size,), XY_LABELS, dtype=torch.float)

        # train the discriminator
        if optimizer_idx == 0:
            # Forward pass ground_truth batch through D
            output = self.discriminator(ground_truth_representation)
            err_ground_truth=self.adversarial_loss(output,label)

            # Train discriminator to identify x_generated_representation from ground_truth
            self.x_generated_representation=self.xgenerator(x)
            label.fill_(X_LABELS)
            output = self.discriminator(self.x_generated_representation.detach())

            err_disc_x_generated_representation=self.adversarial_loss(output,label)

            # Train discriminator to identify y_generated_representation from ground_truth
            self.y_generated_representation = self.ygenerator(y)
            label.fill_(Y_LABELS)
            output = self.discriminator(self.y_generated_representation.detach())

            err_disc_y_generated_representation = self.adversarial_loss(output, label)

            errD = err_ground_truth + err_disc_x_generated_representation + err_disc_y_generated_representation
            tqdm_dict = {'d_loss': errD}

            return {'loss': errD,'progress_bar':tqdm_dict}

        # Train the X generator
        if optimizer_idx == 1:
            label.fill_(XY_LABELS)
            output = self.discriminator(self.x_generated_representation.detach())
            err_gx = self.adversarial_loss(output,label)
            err_gx.backward()
            tqdm_dict = {'gx_loss': err_gx}

            return {'loss': err_gx, 'progress_bar': tqdm_dict}
        # Train The Y generator
        if optimizer_idx == 2:
            label.fill_(XY_LABELS)
            output = self.discriminator(self.y_generated_representation.detach())
            err_gy = self.adversarial_loss(output, label)
            err_gy.backward()
            tqdm_dict = {'gy_loss': err_gy}

            return {'loss': err_gy, 'progress_bar': tqdm_dict}

    @classmethod
    def get_hyper_parameter_search_dict(cls):
        return {'learning_rate': 'log_scale'}







