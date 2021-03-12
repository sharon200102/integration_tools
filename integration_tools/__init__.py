from .mulvae import MultipleVAE
from .IntegrativeGAN import IntegrativeGAN
from .utils.data.data_classes import TrainOnlyDataModule,TrainAndValidateDataModule
integration_models = {'MultipleVAE': MultipleVAE, 'IntegrativeGAN': IntegrativeGAN}
data_modules_for_integration_models = {'MultipleVAE': TrainAndValidateDataModule, 'IntegrativeGAN': TrainOnlyDataModule}

