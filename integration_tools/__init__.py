from .mulvae import MultipleVAE
from .IntegrativeGAN import IntegrativeGAN
from.ProjectionModels.stdvae import StandardVAE_pl
from .utils.data.data_classes import TrainOnlyDataModule,TrainAndValidateDataModule
integration_models = {'MultipleVAE': MultipleVAE, 'IntegrativeGAN': IntegrativeGAN}
projection_models = {'VAE':StandardVAE_pl}
data_modules_for_integration_models = {'MultipleVAE': TrainAndValidateDataModule, 'IntegrativeGAN': TrainOnlyDataModule}
data_modules_for_projection_models = {'VAE':TrainAndValidateDataModule}

