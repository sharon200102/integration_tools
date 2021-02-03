from torch import nn, optim

# Global variables which translate the names into the corresponding functions and objects
activation_fn_dict = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid,
                      'leaky relu': nn.LeakyReLU}
optimizers_dict = {'adam': optim.Adam, 'Adadelta': optim.Adadelta, 'Adagrad': optim.Adagrad}
