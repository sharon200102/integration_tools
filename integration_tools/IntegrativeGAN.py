import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from ._types import Tensor
from torchvision.transforms import transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from integration_tools.ProjectionModels import data_modules_for_projection_models, early_stopping_dict, \
    projection_models
from .Constants import activation_fn_dict, optimizers_dict
from .utils.data.data_classes import DualDataset, ConcatDataset, ALL_FIELDS, FIELD_ONE_ONLY, FIELD_ZERO_ONLY
from .utils.transforms.transforms_classes import EntityToTensor, ToTensor, Todict
from .ProjectionModels.stdvae import StandardVAE_pl
import torch
from copy import deepcopy

GROUND_TRUTH = 0
GENERATED = 1

discriminator_output_size = 2


class IntegrativeGAN(pl.LightningModule):

    def __init__(self, xgenerator, ygenerator, discriminator, data: DualDataset,
                 projection_model, xgenerator_learning_rate=0.001, ygenerator_learning_rate=0.001,
                 discriminator_learning_rate=0.001,
                 xgenerator_optimizer_name='adam', ygenerator_optimizer_name='adam',
                 discriminator_optimizer_name='adam', train_percent=0.8, train_batch_size=10, validation_batch_size=10,
                 projection_train_percent=0.8,
                 projection_train_batch_size=10, projection_validation_batch_size=10):

        """The function doesn't receive the actual generators and discriminator due to the fact that they cannot be
        saved as hyper parameters """
        super().__init__()
        # Don't save the projection_model,data because It's not possible to save them properly.
        not_to_save_args = ['self', 'xgenerator', 'ygenerator', 'discriminator', 'projection_model', 'data']
        # https://stackoverflow.com/questions/582056/getting-list-of-parameter-names-inside-python-function

        all_init_parameters = self.__init__.__code__.co_varnames[:self.__init__.__code__.co_argcount]
        hyper_parameters_to_save = [param_to_save for param_to_save in all_init_parameters if
                                    param_to_save not in not_to_save_args]
        self.save_hyperparameters(*hyper_parameters_to_save)
        self.projection_model = deepcopy(projection_model)
        self.data = deepcopy(data)
        self.xgenerator = deepcopy(xgenerator)
        self.ygenerator = deepcopy(ygenerator)
        self.discriminator = deepcopy(discriminator)
        self.loss = nn.CrossEntropyLoss()
        self.xgenerator_optimizer_function = optimizers_dict[xgenerator_optimizer_name]
        self.ygenerator_optimizer_function = optimizers_dict[ygenerator_optimizer_name]
        self.discriminator_optimizer_function = optimizers_dict[discriminator_optimizer_name]

    def prepare_data(self) -> None:
        projection_model_class_name = type(self.projection_model).__name__
        model_checkpoints_folder = '{}_checkpoints'.format(projection_model_class_name)
        model_logs_folder = '{}_Logs'.format(projection_model_class_name)

        monitor = None
        callbacks = []
        # Separate the full samples from the Dual dataset
        xy_dataset = DualDataset.subset(self.data, self.data.separate_to_groups()[0],
                                        transform=transforms.Compose([ToTensor(), EntityToTensor()]))
        # Create datamodule specific to the projection module
        dm = data_modules_for_projection_models[projection_model_class_name](xy_dataset,
                                                                             self.hparams.projection_train_batch_size,
                                                                             self.hparams.projection_validation_batch_size,
                                                                             self.hparams.projection_train_percent)
        # If the projection model demands an Early stopping callback
        if projection_model_class_name in early_stopping_dict:
            monitor = early_stopping_dict[projection_model_class_name]
            callbacks.append(EarlyStopping(monitor=monitor))

        # Create a ModelCheckpoint callback in order to save the best checkpoint
        checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=model_checkpoints_folder)
        callbacks.append(checkpoint_callback)

        logger = TensorBoardLogger(save_dir=model_logs_folder,
                                   name='{} projection logs'.format(projection_model_class_name))

        trainer = pl.Trainer(callbacks=callbacks, logger=logger)
        trainer.fit(self.projection_model, datamodule=dm)

        # recreate the best model and exploit it to project the XY dataset
        trained_projection_model = projection_models[projection_model_class_name].load_from_checkpoint(
            checkpoint_callback.best_model_path)
        # The projection model inherits the project_the_data function from the BaseProjector class
        projected_dataset = trained_projection_model.project_the_data(
            torch.tensor(xy_dataset.simplified, dtype=torch.float64).type(
                torch.FloatTensor))
        xy_dataset.transform = transforms.Compose([ToTensor(), Todict()])
        self.gan_dataset = ConcatDataset(xy_dataset, projected_dataset)

    def setup(self, stage: str):
        train_size = int(self.hparams.train_percent * len(self.gan_dataset))
        validation_size = len(self.gan_dataset) - train_size
        self.gan_train_data, self.gan_validation_data = torch.utils.data.random_split(self.gan_dataset,
                                                                                      [train_size, validation_size])

    def train_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.gan_train_data, shuffle=True,
                                           batch_size=self.hparams.train_batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.gan_validation_data, shuffle=False,
                                           batch_size=self.hparams.validation_batch_size)

    def forward(self, x, y):
        return {'x': self.xgenerator(x), 'y': self.ygenerator(y)}

    def configure_optimizers(self):
        return [self.discriminator_optimizer_function(self.discriminator.parameters(),
                                                      lr=self.hparams.discriminator_learning_rate),
                self.xgenerator_optimizer_function(self.xgenerator.parameters(),
                                                   lr=self.hparams.xgenerator_learning_rate),
                self.ygenerator_optimizer_function(self.ygenerator.parameters(),
                                                   lr=self.hparams.ygenerator_learning_rate)], []

    def adversarial_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        raw_samples, ground_truth_representation = batch
        x, y = raw_samples['FIELD0'], raw_samples['FIELD1']
        b_size = x.shape[0]

        # train the discriminator
        if optimizer_idx == 0:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long)
            # Forward pass ground_truth batch through D
            err_ground_truth = self.adversarial_loss(self.discriminator(ground_truth_representation),
                                                     real_latent_representation_label)

            # Train discriminator to identify x_generated_representation from ground_truth
            fake_x_label = torch.full((b_size,), GENERATED, dtype=torch.long)
            err_disc_x_generated_representation = self.adversarial_loss(
                self.discriminator(self.xgenerator(x).detach()),
                fake_x_label)

            # Train discriminator to identify y_generated_representation from ground_truth
            fake_y_label = torch.full((b_size,), GENERATED, dtype=torch.long)
            err_disc_y_generated_representation = self.adversarial_loss(
                self.discriminator(self.ygenerator(y).detach()),
                fake_y_label)

            self.errD = err_ground_truth + err_disc_x_generated_representation + err_disc_y_generated_representation
            self.log('discriminator_loss', self.errD, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return {'loss': self.errD}

        # Train the X generator
        if optimizer_idx == 1:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long)
            self.err_gx = self.adversarial_loss(self.discriminator(self.xgenerator(x)),
                                                real_latent_representation_label)
            self.log('xgenerator_loss', self.err_gx, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return {'loss': self.err_gx}
        # Train The Y generator
        if optimizer_idx == 2:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long)
            self.err_gy = self.adversarial_loss(self.discriminator(self.ygenerator(y)),
                                                real_latent_representation_label)
            self.log('ygenerator_loss', self.err_gy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('total_loss', self.err_gx + self.err_gy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return {'loss': self.err_gy}

    def validation_step(self, batch, batch_idx):

        raw_samples, ground_truth_representation = batch
        x, y = raw_samples['FIELD0'], raw_samples['FIELD1']
        b_size = x.shape[0]
        real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long)
        err_gx = self.adversarial_loss(self.discriminator(self.xgenerator(x)),
                                       real_latent_representation_label)

        err_gy = self.adversarial_loss(self.discriminator(self.ygenerator(y)),
                                       real_latent_representation_label)

        return {'loss': err_gx + err_gy}

    def validation_epoch_end(self, outputs) -> None:
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_val_loss, logger=True, prog_bar=True)
        self.logger.experiment.add_scalar("validation_loss_per_epoch", avg_val_loss, self.current_epoch)

    def project_the_data(self) -> Tensor:
        self.eval()
        with torch.no_grad():
            outputs = []
            tensor_entities = [torch.tensor(simplified_entity, dtype=torch.float32, requires_grad=False)
                               for simplified_entity in self.data.simplified]
            entities_status = [entity.get_status() for entity in self.data.entities]
            for tensor_entity, entity_status in zip(tensor_entities, entities_status):
                if entity_status == ALL_FIELDS:
                    result = self.projection_model(tensor_entity)[4]
                elif entity_status == FIELD_ZERO_ONLY:
                    result = self.xgenerator(tensor_entity)
                elif entity_status == FIELD_ONE_ONLY:
                    result = self.ygenerator(tensor_entity)
                outputs.append(result)
        return torch.stack(outputs)

    """@classmethod
        def from_configuration_dictionaries(cls, tuned_parameters: dict, fixed_parameters: dict):
            # Fill tha missing parameters with default values.
            tuned_parameters = set_default_parameters(tuned_parameters, default_tuned_dict)
            fixed_parameters = set_default_parameters(fixed_parameters, default_fixed_dict)

            latent_representation_size = tuned_parameters['latent_representation_size']
            data = load_pickle(fixed_parameters['data_path'])

            xgenerator_architecture = fixed_parameters['xgenerator_internal_layers_structure'] + [
                latent_representation_size]

            ygenerator_architecture = fixed_parameters['ygenerator_internal_layers_structure'] + [
                latent_representation_size]

            discriminator_architecture = [latent_representation_size] + fixed_parameters[
                'discriminator_internal_layers_structure'] + [discriminator_output_size]

            projection_model_architecture = fixed_parameters['projector_internal_layers_structure'] + [
                latent_representation_size]
            projection_model = StandardVAE_pl(projection_model_architecture,
                                              fixed_parameters['projector_activation_function_name'],
                                              tuned_parameters['projector_klb_coefficient'],
                                              tuned_parameters['projector_learning_rate'],
                                              fixed_parameters['projector_optimizer_name'])

            return cls(xgenerator_architecture, ygenerator_architecture, discriminator_architecture,
                       fixed_parameters['xgenerator_activation_function_name'],
                       fixed_parameters['ygenerator_activation_function_name']
                       , fixed_parameters['discriminator_activation_function_name'], data, projection_model,
                       tuned_parameters['xgenerator_learning_rate'],
                       tuned_parameters['ygenerator_learning_rate'], tuned_parameters['discriminator_learning_rate'],
                       fixed_parameters['xgenertor_optimizer_name'], fixed_parameters['ygenertor_optimizer_name'],
                       fixed_parameters['discriminator_optimizer_name'], fixed_parameters['train_percent'],
                       tuned_parameters['train_batch_size'], fixed_parameters['validation_batch_size'],
                       fixed_parameters['projection_train_percent'], fixed_parameters['projection_train_batch_size'],
                       fixed_parameters['projection_validation_batch_size'])
    """
