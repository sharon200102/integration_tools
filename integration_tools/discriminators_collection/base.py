import torch.nn as nn
from integration_tools.Constants import activation_fn_dict, batch_norm_before_or_after_activation
from ..utils.models.learning_utils import MiniBatchDiscrimination


class Discriminator(nn.Module):

    def __init__(self, layers_structure: list, activation_fn: str = 'relu', use_batch_norm: bool = True,
                 use_mini_batch_discrimination: bool = False, layer_index: int = None,
                 similarity_features: int = 10, kernel_dims: int = 10, mini_batch_discrimination_kwargs: dict = None,
                 feature_matching_layer: int = None):
        super().__init__()
        self.activation_fn_name = activation_fn
        self.activation_fn = activation_fn_dict[activation_fn]()
        self.use_batch_norm = use_batch_norm
        self.feature_matching_layer = feature_matching_layer
        if mini_batch_discrimination_kwargs is None:
            mini_batch_discrimination_kwargs = {}

        modules = []

        for i in range(0, len(layers_structure) - 1):
            if i == layer_index and use_mini_batch_discrimination:
                mbd = MiniBatchDiscrimination(layers_structure[i], similarity_features, kernel_dims,
                                              **mini_batch_discrimination_kwargs)
                modules.append(mbd)
                # as a consequence of the mbd activation the input shape increases,
                # therefore a modification in the layers structure is made.
                layers_structure[i] = layers_structure[i] + similarity_features
            if i == len(layers_structure) - 2:
                modules.extend(self.block(layers_structure[i], layers_structure[i + 1], activation=False,
                                          use_batch_norm=False))
            else:
                modules.extend(self.block(layers_structure[i], layers_structure[i + 1],
                                          use_batch_norm=self.use_batch_norm))
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x, matching: bool = False):
        # keep track of the activation functions, for the feature-matching.
        activation_layer_counter = 0
        for module in self.modules_list:
            # Batch Norm requires a batch dimension, if missing, add one.
            try:
                x = module(x)

            except ValueError:
                x = x.unsqueeze(0)
                x = module(x)

            if module is self.activation_fn:
                activation_layer_counter += 1
                # IF the current layer is the feature matching layer, return immediately
                if activation_layer_counter == self.feature_matching_layer and matching == True:
                    return x
        return x

    def block(self, in_feat, out_feat, activation=True, use_batch_norm=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if use_batch_norm and activation and batch_norm_before_or_after_activation[self.activation_fn_name] == 'after':
            layers.extend([self.activation_fn, nn.BatchNorm1d(out_feat)])
        elif use_batch_norm and activation and batch_norm_before_or_after_activation[
            self.activation_fn_name] == 'before':
            layers.extend([nn.BatchNorm1d(out_feat), self.activation_fn])
        elif use_batch_norm:
            layers.append(nn.BatchNorm1d(out_feat))
        elif activation:
            layers.append(self.activation_fn)

        return layers
