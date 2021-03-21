from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from integration_tools.IntegrativeGAN import IntegrativeGAN
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import yaml
import nni
import os.path
from integration_tools.ProjectionModels import StandardVAE_pl
from integration_tools.utils.models.learning_utils import set_default_parameters
from integration_tools.utils.parser.parser import load_yaml, load_pickle
import pickle
from integration_tools.generators_collection.base import Generator
from integration_tools.discrimantors_collection.base import Discriminator


def main(fixed_args, tuned_args, trial_id):
    """""defining some constants"""
    """DEFAULT_VALUES"""
    default_latent_representation_size = 10
    default_projector_klb_coefficient = 1
    projector_default_learning_rate = 0.01
    xgenerator_default_learning_rate = 0.001
    ygenerator_default_learning_rate = 0.001
    discriminator_default_learning_rate = 0.001
    default_train_batch_size = 10
    discriminator_output_size = 2

    default_xgenerator_activation_function = 'relu'
    default_ygenerator_activation_function = 'relu'
    default_discriminator_activation_function = 'relu'
    default_projector_activation_function = 'relu'
    projector_default_optimizer = 'adam'
    default_xgenertor_optimizer_name = 'adam'
    default_ygenertor_optimizer_name = 'adam'
    default_discriminator_optimizer_name = 'adam'
    defualt_train_percent = 0.8
    default_validation_batch_size = 10
    default_projection_train_percent = 0.8
    default_projection_train_batch_size = 10
    default_projection_validation_batch_size = 10
    default_xgenerator_batch_norm = True
    default_ygenerator_batch_norm = True
    default_discriminator_batch_norm = True

    default_tuned_dict = {'latent_representation_size': default_latent_representation_size,
                          'projector_klb_coefficient': default_projector_klb_coefficient,
                          'projector_learning_rate': projector_default_learning_rate,
                          'xgenerator_learning_rate': xgenerator_default_learning_rate,
                          'ygenerator_learning_rate': ygenerator_default_learning_rate,
                          'discriminator_learning_rate': discriminator_default_learning_rate,
                          'train_batch_size': default_train_batch_size}

    default_fixed_dict = {'xgenerator_activation_function_name': default_xgenerator_activation_function,
                          'ygenerator_activation_function_name': default_ygenerator_activation_function,
                          'discriminator_activation_function_name': default_discriminator_activation_function,
                          'projector_activation_function_name': default_projector_activation_function,
                          'projector_optimizer_name': projector_default_optimizer,
                          'xgenertor_optimizer_name': default_xgenertor_optimizer_name,
                          'ygenertor_optimizer_name': default_ygenertor_optimizer_name,
                          'discriminator_optimizer_name': default_discriminator_optimizer_name,
                          'train_percent': defualt_train_percent,
                          'validation_batch_size': default_validation_batch_size,
                          'projection_train_percent': default_projection_train_percent,
                          'projection_train_batch_size': default_projection_train_batch_size,
                          'projection_validation_batch_size': default_projection_validation_batch_size,
                          'xgenerator_batch_norm': default_xgenerator_batch_norm,
                          'ygenerator_batch_norm': default_ygenerator_batch_norm,
                          'discriminator_batch_norm': default_discriminator_batch_norm

                          }

    tuned_parameters = set_default_parameters(tuned_args, default_tuned_dict)
    fixed_parameters = set_default_parameters(fixed_args, default_fixed_dict)

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

    xgenerator = Generator(xgenerator_architecture, fixed_parameters['xgenerator_activation_function_name'],
                           fixed_parameters['xgenerator_batch_norm'])
    ygenerator = Generator(ygenerator_architecture, fixed_parameters['ygenerator_activation_function_name'],
                           fixed_parameters['ygenerator_batch_norm'])
    discriminator = Discriminator(discriminator_architecture,
                                  fixed_parameters['discriminator_activation_function_name'],
                                  fixed_parameters['discriminator_batch_norm'])
    model = IntegrativeGAN(xgenerator, ygenerator, discriminator, data, projection_model,
                           tuned_parameters['xgenerator_learning_rate'],
                           tuned_parameters['ygenerator_learning_rate'],
                           tuned_parameters['discriminator_learning_rate'],
                           fixed_parameters['xgenertor_optimizer_name'], fixed_parameters['ygenertor_optimizer_name'],
                           fixed_parameters['discriminator_optimizer_name'], fixed_parameters['train_percent'],
                           tuned_parameters['train_batch_size'], fixed_parameters['validation_batch_size'],
                           fixed_parameters['projection_train_percent'],
                           fixed_parameters['projection_train_batch_size'],
                           fixed_parameters['projection_validation_batch_size'])

    callbacks = []
    monitor = 'val_loss'
    model_logs_folder = 'GAN_Logs'
    model_checkpoints_folder = os.path.join('GAN_checkpoints', '{}'.format(trial_id))
    max_epochs = 200
    # Creating the model.
    # Creating the callbacks.
    checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=model_checkpoints_folder, save_last=True)
    callbacks.extend([checkpoint_callback])

    # Create a logger for the IntegrativeGAN
    logger = TensorBoardLogger(save_dir=model_logs_folder, name='GAN_integration_logs')

    # Train the model
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, max_epochs=max_epochs)
    trainer.fit(model)

    best_result = checkpoint_callback.best_model_score
    print(best_result)

    nni.report_final_result(best_result.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runner for Integrative GAN')
    parser.add_argument('--fixed_parameters', '-f',
                        dest="filename",
                        metavar='FILE',
                        help='path to the fixed parameters file')

    args = parser.parse_args()
    fixed_parameters = load_yaml(args.filename)
    tuned_parameters = nni.get_next_parameter()
    main(fixed_parameters, tuned_parameters, nni.get_sequence_id())
