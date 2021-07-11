from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from integration_tools.IntegrativeGAN import IntegrativeGAN
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import nni
import os.path
import integration_tools
import integration_tools.ProjectionModels as Projection_models
from integration_tools.utils.parser.parser import load_yaml, load_pickle
import integration_tools.generators_collection as generators_collection
import integration_tools.discriminators_collection as discriminator_collection
import torch


def main(args: dict, trial_id=0):
    torch.manual_seed(args['logger_params']['manual_seed'])
    data = load_pickle(args['exp_params']['data_path'])

    projection_name = args['model_params']['projection_name']
    xgenerator_name = args['model_params']['x_generator_name']
    ygenerator_name = args['model_params']['y_generator_name']
    discriminator_name = args['model_params']['discriminator_name']
    GAN_name = args['model_params']['GAN_name']
    results_folder = args['exp_params']['results_folder']

    projection_model = Projection_models.projection_models[projection_name](**args['model_params']['projection_params'])

    xgenerator = generators_collection.generator_models[xgenerator_name](**args['model_params']['x_generator_params'])

    ygenerator = generators_collection.generator_models[ygenerator_name](**args['model_params']['y_generator_params'])

    discriminator = discriminator_collection.discriminator_models[discriminator_name] \
        (**args['model_params']['discriminator_params'])

    # Creating the model.
    model = integration_tools.GAN_models[GAN_name](xgenerator, ygenerator,
                                                   discriminator, data,
                                                   projection_model, results_dir=results_folder,
                                                   projection_identifier=str(trial_id),
                                                   **args['model_params']['GAN_params'])

    callbacks = []
    monitor = 'val_loss'
    model_logs_folder = os.path.join(results_folder, args['logger_params']['GAN_logs_name'])
    model_checkpoints_folder = os.path.join(results_folder, args['logger_params']['GAN_checkpoints_name'],
                                            '{}'.format(trial_id))
    # Creating the callbacks.
    checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=model_checkpoints_folder, save_last=True)
    callbacks.extend([checkpoint_callback])

    # Create a logger for the IntegrativeGAN
    logger = TensorBoardLogger(save_dir=model_logs_folder)

    # Train the model
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **args['trainer_params'])
    trainer.fit(model)

    best_result = checkpoint_callback.best_model_score
    print(best_result)

    nni.report_final_result(best_result.item())


def add_tuned_parameters(current_dict: dict, additional_dict: dict):
    current_dict = current_dict.copy()
    additional_dict = additional_dict.copy()
    latent_representation_size = additional_dict['latent_representation_size']
    current_dict['model_params']['x_generator_params']['layers_structure'].append(latent_representation_size)
    current_dict['model_params']['y_generator_params']['layers_structure'].append(latent_representation_size)
    current_dict['model_params']['projection_params']['layers_structure'].append(latent_representation_size)
    current_dict['model_params']['discriminator_params']['layers_structure'].insert(0, latent_representation_size)

    current_dict['model_params']['projection_params']['kld_coefficient'] = additional_dict['projector_klb_coefficient']
    current_dict['model_params']['projection_params']['learning_rate'] = additional_dict['projector_learning_rate']
    current_dict['model_params']['GAN_params']['xgenerator_learning_rate'] = additional_dict['xgenerator_learning_rate']
    current_dict['model_params']['GAN_params']['ygenerator_learning_rate'] = additional_dict['ygenerator_learning_rate']
    current_dict['model_params']['GAN_params']['discriminator_learning_rate'] = additional_dict[
        'discriminator_learning_rate']
    current_dict['model_params']['GAN_params']['train_batch_size'] = additional_dict['train_batch_size']
    return config_parameters


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runner for Integrative GAN')
    parser.add_argument('--config_path', '-c',
                        metavar='FILE',
                        help='The path to the config file')

    parser.add_argument('--apply_nni', '-n', dest='nni_flag', action='store_true')

    parser_args = parser.parse_args()
    config_parameters = load_yaml(parser_args.config_path)
    if parser_args.nni_flag:
        tuned_parameters = nni.get_next_parameter()
        config_parameters = add_tuned_parameters(config_parameters, tuned_parameters)

    main(config_parameters, nni.get_sequence_id())


#-c configs/integration_configs/GAN_configs/GDM/feature_matching_and_batch_discrimination/params.yaml  -n
"""
tuned_parameters = {'latent_representation_size': 8, 'projector_klb_coefficient': 0.0005,
                            'projector_learning_rate': 0.001, 'xgenerator_learning_rate': 0.0001,
                            'ygenerator_learning_rate': 0.0001, 'discriminator_learning_rate': 0.001,
                            'train_batch_size': 10}
        
"""