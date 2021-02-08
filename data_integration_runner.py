import yaml
import argparse
from pytorch_lightning import Trainer
from integration_tools import *
from integration_tools.utils.data.data_classes import TrainAndValidateDataModule, DualDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from integration_tools.utils.parser.parser import create_search_space_dict
import ray.tune as tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.loggers import TensorBoardLogger
import pickle
import os
import  ray
from integration_tools.utils.transforms.transforms_classes import Concatenate
from torchvision import transforms
#ray.init(local_mode=True)

def train_integration_model(model_hyper_parameters: dict, model_core_parameters: dict, data, program_parameters: dict,
                            trainer_params: dict = {},
                            logger_params: dict = {}):
    model = integration_models[program_parameters['model_name']](model_hyper_parameters, model_core_parameters)
    metrics = logger_params['metric_params']
    callbacks = [TuneReportCallback(metrics, on=logger_params['on_end'])]

    if 'patience' in program_parameters:
        callbacks.append(EarlyStopping('val_loss', patience=program_parameters['patience']))

    runner = Trainer(callbacks=callbacks, logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
                     **trainer_params)
    runner.fit(model, datamodule=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic runner for integration models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            experiment_parameters = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    data_path = experiment_parameters['program_params']['data_path']
    if os.path.isfile(data_path):
        # Load the dual_dataset object which upon it the learning will be performed.
        with open(data_path, 'rb') as data_file:
            dataset = pickle.load(data_file)
    # Not valid
    else:
        raise FileNotFoundError(data_path)

    model_name = experiment_parameters['program_params']['model_name']
    dm = data_modules_for_integration_models[model_name](dataset, **experiment_parameters['data_params'])

    integration_model_class = integration_models[model_name]

    search_space_dict = create_search_space_dict(experiment_parameters['tuned_params'],
                                                 integration_model_class.get_hyper_parameter_search_dict())
    parameters_configuration_dict = {key: tune.grid_search(search_space_dict[key]) for key in search_space_dict.keys()}

    trainable = tune.with_parameters(
        train_integration_model,
        model_core_parameters=experiment_parameters['core_params'],
        data=dm,
        program_parameters=experiment_parameters['program_params'],
        logger_params=experiment_parameters['logger_params'],
        trainer_params=experiment_parameters['trainer_params']

    )

    analysis = tune.run(
        trainable,
        metric="loss",
        mode="min",
        config=parameters_configuration_dict,
        name="tune_integration_model",
        local_dir=experiment_parameters['program_params']['save_dir'])

    print(analysis.best_config)
    print(analysis.best_trial)  # Get best trial
    print(analysis.best_logdir)  # Get best trial's logdir
    print(analysis.best_checkpoint)  # Get best trial's best checkpoint
    print(analysis.best_result)  # Get best trial's last results