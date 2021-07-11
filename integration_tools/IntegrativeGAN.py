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
from os.path import join

GROUND_TRUTH = 0
GENERATED = 1
XGENERATED = 1
YGENERATED = 2


class IntegrativeGAN(pl.LightningModule):

    def __init__(self, xgenerator, ygenerator, discriminator, data: DualDataset,
                 projection_model, xgenerator_learning_rate=0.001, ygenerator_learning_rate=0.001,
                 discriminator_learning_rate=0.001,
                 xgenerator_optimizer_name='adam', ygenerator_optimizer_name='adam',
                 discriminator_optimizer_name='adam', train_percent=0.8, train_batch_size=10, validation_batch_size=10,
                 projection_train_percent=0.8,
                 projection_train_batch_size=10, projection_validation_batch_size=10, results_dir='.',
                 projection_identifier=None):

        super().__init__()
        self.projection_model = deepcopy(projection_model)
        self.data = deepcopy(data)
        self.xgenerator = deepcopy(xgenerator)
        self.ygenerator = deepcopy(ygenerator)
        self.discriminator = deepcopy(discriminator)
        self.loss = nn.CrossEntropyLoss()
        self.xgenerator_optimizer_function = optimizers_dict[xgenerator_optimizer_name]
        self.ygenerator_optimizer_function = optimizers_dict[ygenerator_optimizer_name]
        self.discriminator_optimizer_function = optimizers_dict[discriminator_optimizer_name]
        self.xgenerator_learning_rate = xgenerator_learning_rate
        self.ygenerator_learning_rate = ygenerator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.train_percent = train_percent
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.projection_train_percent = projection_train_percent
        self.projection_train_batch_size = projection_train_batch_size
        self.projection_validation_batch_size = projection_validation_batch_size
        self.results_dir = results_dir
        self.projection_identifier = projection_identifier

    def prepare_data(self) -> None:
        projection_model_class_name = type(self.projection_model).__name__
        model_checkpoints_folder = join(self.results_dir, '{}_checkpoints'.format(projection_model_class_name),
                                        self.projection_identifier) if self.projection_identifier is not None \
            else join(self.results_dir, '{}_checkpoints'.format(projection_model_class_name))

        model_logs_folder = join(self.results_dir, '{}_Logs'.format(projection_model_class_name))

        monitor = None
        callbacks = []
        # Separate the full samples from the Dual dataset
        xy_dataset = DualDataset.subset(self.data, self.data.separate_to_groups()[0],
                                        transform=transforms.Compose([ToTensor(), EntityToTensor()]))
        # Create datamodule specific to the projection module
        dm = data_modules_for_projection_models[projection_model_class_name](xy_dataset,
                                                                             self.projection_train_batch_size,
                                                                             self.projection_validation_batch_size,
                                                                             self.projection_train_percent)
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
        train_size = int(self.train_percent * len(self.gan_dataset))
        validation_size = len(self.gan_dataset) - train_size
        self.gan_train_data, self.gan_validation_data = torch.utils.data.random_split(self.gan_dataset,
                                                                                      [train_size, validation_size])

    def train_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.gan_train_data, shuffle=True,
                                           batch_size=self.train_batch_size, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.gan_validation_data, shuffle=False,
                                           batch_size=self.validation_batch_size)

    def forward(self, x, y):
        return {'x': self.xgenerator(x), 'y': self.ygenerator(y)}

    def configure_optimizers(self):
        return [self.discriminator_optimizer_function(self.discriminator.parameters(),
                                                      lr=self.discriminator_learning_rate),
                self.xgenerator_optimizer_function(self.xgenerator.parameters(),
                                                   lr=self.xgenerator_learning_rate),
                self.ygenerator_optimizer_function(self.ygenerator.parameters(),
                                                   lr=self.ygenerator_learning_rate)], []

    def adversarial_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        raw_samples, ground_truth_representation = batch
        x, y = raw_samples['FIELD0'], raw_samples['FIELD1']
        b_size = x.shape[0]

        # train the discriminator
        if optimizer_idx == 0:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long,
                                                          requires_grad=False)
            # Forward pass ground_truth batch through D
            err_ground_truth = self.adversarial_loss(self.discriminator(ground_truth_representation.detach()),
                                                     real_latent_representation_label)

            # Train discriminator to identify x_generated_representation from ground_truth
            fake_x_label = torch.full((b_size,), GENERATED, dtype=torch.long, requires_grad=False)
            err_disc_x_generated_representation = self.adversarial_loss(
                self.discriminator(self.xgenerator(x).detach()),
                fake_x_label)

            # Train discriminator to identify y_generated_representation from ground_truth
            fake_y_label = torch.full((b_size,), GENERATED, dtype=torch.long, requires_grad=False)
            err_disc_y_generated_representation = self.adversarial_loss(
                self.discriminator(self.ygenerator(y).detach()),
                fake_y_label)

            self.errD = err_ground_truth + err_disc_x_generated_representation + err_disc_y_generated_representation
            self.log('discriminator_loss', self.errD, prog_bar=True, logger=True)
            return {'loss': self.errD}

        # Train the X generator
        if optimizer_idx == 1:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long,
                                                          requires_grad=False)
            self.err_gx = self.adversarial_loss(self.discriminator(self.xgenerator(x)),
                                                real_latent_representation_label)
            self.log('xgenerator_loss', self.err_gx, prog_bar=True, logger=True)

            return {'loss': self.err_gx}
        # Train The Y generator
        if optimizer_idx == 2:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long,
                                                          requires_grad=False)
            self.err_gy = self.adversarial_loss(self.discriminator(self.ygenerator(y)),
                                                real_latent_representation_label)
            self.log('ygenerator_loss', self.err_gy, prog_bar=True, logger=True)
            self.log('total_loss', self.err_gx + self.err_gy, prog_bar=True, logger=True)

            return {'loss': self.err_gy}

    def validation_step(self, batch, batch_idx):

        raw_samples, ground_truth_representation = batch
        x, y = raw_samples['FIELD0'], raw_samples['FIELD1']
        b_size = x.shape[0]
        real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long, requires_grad=False)
        err_gx = self.adversarial_loss(self.discriminator(self.xgenerator(x)),
                                       real_latent_representation_label)

        err_gy = self.adversarial_loss(self.discriminator(self.ygenerator(y)),
                                       real_latent_representation_label)

        return {'loss': err_gx + err_gy}

    def validation_epoch_end(self, outputs) -> None:
        avg_val_loss = torch.tensor([x['loss'] for x in outputs], requires_grad=False).mean()
        self.log('val_loss', avg_val_loss, logger=True, prog_bar=True)
        # log the vla loss with respect to the epoch number
        self.logger.experiment.add_scalar("validation_loss_per_epoch", avg_val_loss, self.current_epoch)

    def project_the_data(self) -> Tensor:
        self.eval_all()
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
                # Batch norm may cause a dimension increment, therefore the squeeze function is applied,
                result = result.squeeze()
                outputs.append(result)
        return torch.stack(outputs)

    def eval_all(self):
        self.xgenerator.eval()
        self.ygenerator.eval()
        self.discriminator.eval()


class FMIntegrativeGAN(IntegrativeGAN):
    def __init__(self, *args, **kwargs):

        super(FMIntegrativeGAN, self).__init__(*args, **kwargs)
        self.xgenerator_loss = nn.MSELoss(reduction='sum')
        self.ygenerator_loss = nn.MSELoss(reduction='sum')

    def training_step(self, batch, batch_idx, optimizer_idx):
        raw_samples, ground_truth_representation = batch
        x, y = raw_samples['FIELD0'], raw_samples['FIELD1']
        b_size = x.shape[0]

        # train the discriminator
        if optimizer_idx == 0:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long,
                                                          requires_grad=False)
            # Forward pass ground_truth batch through D
            err_ground_truth = self.adversarial_loss(self.discriminator(ground_truth_representation.detach()),
                                                     real_latent_representation_label)

            # Train discriminator to identify x_generated_representation from ground_truth
            fake_x_label = torch.full((b_size,), GENERATED, dtype=torch.long, requires_grad=False)
            err_disc_x_generated_representation = self.adversarial_loss(
                self.discriminator(self.xgenerator(x).detach()),
                fake_x_label)

            # Train discriminator to identify y_generated_representation from ground_truth
            fake_y_label = torch.full((b_size,), GENERATED, dtype=torch.long, requires_grad=False)
            err_disc_y_generated_representation = self.adversarial_loss(
                self.discriminator(self.ygenerator(y).detach()),
                fake_y_label)

            self.errD = err_ground_truth + err_disc_x_generated_representation + err_disc_y_generated_representation
            self.log('discriminator_loss', self.errD, prog_bar=True, logger=True)
            return {'loss': self.errD}

        # Train the X generator
        if optimizer_idx == 1:
            # Apply feature matching, that is to say, calculate the mse
            # between the expectation of the real features and the fake features.
            real_features = self.discriminator(ground_truth_representation.detach(), matching=True)
            fake_features = self.discriminator(self.xgenerator(x), matching=True)
            mean_of_real_features = torch.mean(real_features, dim=0)
            mean_of_fake_features = torch.mean(fake_features, dim=0)

            self.err_gx = self.xgenerator_loss(mean_of_real_features.detach(), mean_of_fake_features)
            self.log('xgenerator_loss', self.err_gx, prog_bar=True, logger=True)

            return {'loss': self.err_gx}
        # Train The Y generator
        if optimizer_idx == 2:
            # Apply feature matching, that is to say, calculate the ms
            # between the expectation of the real features and the fake features.
            real_features = self.discriminator(ground_truth_representation.detach(), matching=True)
            fake_features = self.discriminator(self.ygenerator(y), matching=True)
            mean_of_real_features = torch.mean(real_features, dim=0)
            mean_of_fake_features = torch.mean(fake_features, dim=0)

            self.err_gy = self.ygenerator_loss(mean_of_real_features.detach(), mean_of_fake_features)

            self.log('ygenerator_loss', self.err_gy, prog_bar=True, logger=True)

            self.log('total_loss', self.err_gx + self.err_gy, prog_bar=True, logger=True)
            return {'loss': self.err_gy}

    def validation_step(self, batch, batch_idx):

        raw_samples, ground_truth_representation = batch
        x, y = raw_samples['FIELD0'], raw_samples['FIELD1']

        real_features = self.discriminator(ground_truth_representation.detach(), matching=True).detach()
        mean_of_real_features = torch.mean(real_features, dim=0)

        x_fake_features = self.discriminator(self.xgenerator(x), matching=True)
        mean_of_xfake_features = torch.mean(x_fake_features, dim=0)

        y_fake_features = self.discriminator(self.ygenerator(y), matching=True)
        mean_of_yfake_features = torch.mean(y_fake_features, dim=0)

        err_gx = self.xgenerator_loss(mean_of_real_features, mean_of_xfake_features)
        err_gy = self.ygenerator_loss(mean_of_real_features, mean_of_yfake_features)

        return {'loss': err_gx + err_gy}


class MultiClassGAN(IntegrativeGAN):
    def training_step(self, batch, batch_idx, optimizer_idx):
        raw_samples, ground_truth_representation = batch
        x, y = raw_samples['FIELD0'], raw_samples['FIELD1']
        b_size = x.shape[0]

        # train the discriminator
        if optimizer_idx == 0:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long,
                                                          requires_grad=False)
            # Forward pass ground_truth batch through D
            err_ground_truth = self.adversarial_loss(self.discriminator(ground_truth_representation.detach()),
                                                     real_latent_representation_label)

            # Train discriminator to identify x_generated_representation from ground_truth
            fake_x_label = torch.full((b_size,), XGENERATED, dtype=torch.long, requires_grad=False)
            err_disc_x_generated_representation = self.adversarial_loss(
                self.discriminator(self.xgenerator(x).detach()),
                fake_x_label)

            # Train discriminator to identify y_generated_representation from ground_truth
            fake_y_label = torch.full((b_size,), YGENERATED, dtype=torch.long, requires_grad=False)
            err_disc_y_generated_representation = self.adversarial_loss(
                self.discriminator(self.ygenerator(y).detach()),
                fake_y_label)

            self.errD = err_ground_truth + err_disc_x_generated_representation + err_disc_y_generated_representation
            self.log('discriminator_loss', self.errD, prog_bar=True, logger=True)
            return {'loss': self.errD}

        # Train the X generator
        if optimizer_idx == 1:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long,
                                                          requires_grad=False)
            self.err_gx = self.adversarial_loss(self.discriminator(self.xgenerator(x)),
                                                real_latent_representation_label)
            self.log('xgenerator_loss', self.err_gx, prog_bar=True, logger=True)

            return {'loss': self.err_gx}
        # Train The Y generator
        if optimizer_idx == 2:
            real_latent_representation_label = torch.full((b_size,), GROUND_TRUTH, dtype=torch.long,
                                                          requires_grad=False)
            self.err_gy = self.adversarial_loss(self.discriminator(self.ygenerator(y)),
                                                real_latent_representation_label)
            self.log('ygenerator_loss', self.err_gy, prog_bar=True, logger=True)
            self.log('total_loss', self.err_gx + self.err_gy, prog_bar=True, logger=True)

            return {'loss': self.err_gy}

