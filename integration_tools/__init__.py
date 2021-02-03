from .mulvae import MultipleVAE
from .IntegrativeGAN import IntegrativeGAN
from .utils.data.data_classes import DualDataModule, ConcatDataModule

integration_models = {'MultipleVAE': MultipleVAE, 'IntegrativeGAN': IntegrativeGAN}

integration_data_modules = {'MultipleVAE': DualDataModule, 'IntegrativeGAN': ConcatDataModule}
