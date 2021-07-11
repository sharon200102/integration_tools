from .mulvae import MultipleVAE
from .IntegrativeGAN import IntegrativeGAN,FMIntegrativeGAN,MultiClassGAN
from .utils.data.data_classes import TrainOnlyDataModule, TrainAndValidateDataModule

GAN_models = {'IntegrativeGAN': IntegrativeGAN,'FMIntegrativeGAN':FMIntegrativeGAN,'MultiClassGAN':MultiClassGAN}
VAE_models = {'MultipleVAE': MultipleVAE}
