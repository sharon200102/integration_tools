from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from integration_tools.IntegrativeGAN import IntegrativeGAN
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import yaml
import nni
from integration_tools.utils.parser.parser import load_yaml


def main(fixed_args, tuned_args):
    """""defining some constants"""

    callbacks = []
    monitor = 'val_loss'
    model_logs_folder = 'GAN_Logs'
    model_checkpoints_folder = 'GAN_checkpoints'
    max_epochs = 200
    # Creating the model.
    model = IntegrativeGAN.from_configuration_dictionaries(tuned_args, fixed_args)
    # Creating the callbacks.
    checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=model_checkpoints_folder,save_last=True)
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
    main(fixed_parameters, tuned_parameters)
