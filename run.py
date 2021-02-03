import yaml
import argparse
from pytorch_lightning import Trainer
from integration_tools import *
from integration_tools.utils.data.data_classes import DualDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from integration_tools.utils.parser.parser import create_search_space_dict
import ray.tune as tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.loggers import TensorBoardLogger

def train_integration_model(model_hyper_parameters: dict, model_core_parameters: dict, data_parameters: dict,
                            trainer_parameters: dict):
    dm = integration_data_modules[model_core_parameters['name']](**data_parameters)
    model = integration_models[model_core_parameters['name']](model_hyper_parameters, model_core_parameters)

    patience = trainer_parameters.get('patience')
    if patience is not None:
        metrics = {"loss": "val_loss"}
        callbacks=[TuneReportCallback(metrics, on="validation_end",),EarlyStopping('val_loss', patience=patience)]

    else:
        metrics = {"loss": "loss"}
        callbacks = [TuneReportCallback(metrics)]

    runner = Trainer(callbacks=callbacks,logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."))
    runner.fit(model, datamodule=dm)


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
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

integration_model_class = integration_models[experiment_parameters['core_params']['name']]

search_space_dict = create_search_space_dict(experiment_parameters['tuned_params'],
                                             integration_model_class.get_hyper_parameter_search_dict())
parameters_configuration_dict = {key:tune.grid_search(search_space_dict[key]) for key in search_space_dict.keys()}

trainable = tune.with_parameters(
    train_integration_model,
    model_core_parameters = experiment_parameters['core_params'],
    data_parameters = experiment_parameters['data_params'],
    trainer_parameters = experiment_parameters['trainer_params']
    )

analysis = tune.run(
    trainable,
    metric="loss",
    mode="min",
    config=parameters_configuration_dict,
    name="tune_integration_model",
    local_dir =experiment_parameters['logging_params']['save_dir'])

print(analysis.best_config)


