from .stdvae import StandardVAE_pl
from ..utils.data.data_classes import TrainOnlyDataModule,TrainAndValidateDataModule
projection_models = {'StandardVAE_pl':StandardVAE_pl}
data_modules_for_projection_models = {'StandardVAE_pl':TrainAndValidateDataModule}
early_stopping_dict = {'StandardVAE_pl':'val_loss'}
