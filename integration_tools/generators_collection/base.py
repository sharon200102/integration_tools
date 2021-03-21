import torch.nn as nn
from integration_tools.Constants import activation_fn_dict,batch_norm_before_or_after_activation


class Generator(nn.Module):
    def __init__(self, layers_structure, activation_fn='relu',use_batch_norm = True):
        """
                :param layers_structure: An iterable which will form the structure of the generator

                :param activation_fn: A Pointer to the activation function that will be used between the layers.
                :param use_batch_norm: whether to use a batch norm layer.
        """

        super().__init__()
        self.activation_fn_name = activation_fn
        self.activation_fn = activation_fn_dict[activation_fn]()
        self.use_batch_norm = use_batch_norm
        modules = []

        for i in range(0, len(layers_structure) - 1):
            if i == len(layers_structure) - 2:
                # Do not use an activation function in the last layer
                modules.extend(self.block(layers_structure[i], layers_structure[i + 1], activation=False,
                                          use_batch_norm=False))
            else:
                modules.extend(self.block(layers_structure[i], layers_structure[i + 1],
                                          use_batch_norm=self.use_batch_norm))
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

    def block(self, in_feat, out_feat, activation=True,use_batch_norm = True):
        layers = [nn.Linear(in_feat, out_feat)]
        if use_batch_norm and activation and batch_norm_before_or_after_activation[self.activation_fn_name] == 'after':
            layers.extend([self.activation_fn,nn.BatchNorm1d(out_feat)])
        elif use_batch_norm and activation and batch_norm_before_or_after_activation[self.activation_fn_name] == 'before':
            layers.extend([nn.BatchNorm1d(out_feat),self.activation_fn])
        elif use_batch_norm:
            layers.append(nn.BatchNorm1d(out_feat))
        elif activation:
            layers.append(self.activation_fn)

        return layers
