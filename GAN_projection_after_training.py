from torchvision.transforms import transforms
import numpy
import pandas as pd
from integration_tools.IntegrativeGAN import IntegrativeGAN
from integration_tools.utils.data.data_classes import DualDataset
from integration_tools.ProjectionModels.stdvae import StandardVAE_pl
import pytorch_lightning as pl
import argparse
from integration_tools.utils.parser.parser import load_yaml
from integration_tools.utils.parser.parser import load_pickle
from integration_tools.utils.transforms.transforms_classes import ToTensor
from integration_tools.generators_collection.base import Generator
from integration_tools.discrimantors_collection.base import Discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrator after hyper parameters tuning')
    parser.add_argument('--GAN_checkpoint_path', '-g',
                        dest="GAN_checkpoint_path",
                        metavar='FILE',
                        help='path to the desired GAN checkpoint')
    parser.add_argument('--sub_models_configs_path', '-s',
                        dest="sub_models_configs_path",
                        metavar='FILE',
                        help='path to the desired GAN checkpoint')

    parser.add_argument('--projector_checkpoint_path', '-p',
                        dest="projector_checkpoint_path",
                        metavar='FILE',
                        help='path to the desired projector checkpoint')

    parser.add_argument('--DualData_path', '-d',
                        dest="DualData_path",
                        metavar='FILE',
                        help='path to the desired data object')
    parser.add_argument('--results_path', '-r',
                        dest="results_path",
                        metavar='FILE',
                        help='The path of where the results should ne saved')

    args = parser.parse_args()
    data = load_pickle(args.DualData_path)
    # This Line of code should be changed when package update will occur
    data = DualDataset.subset(data, list(range(len(data))), transform=transforms.Compose([ToTensor()]))
    trained_projector = StandardVAE_pl.load_from_checkpoint(args.projector_checkpoint_path)
    sub_models_parameters = load_yaml(args.sub_models_configs_path)
    xgenerator = Generator(**sub_models_parameters['xgenerator'])
    ygenerator = Generator(**sub_models_parameters['ygenerator'])
    discriminator = Discriminator(**sub_models_parameters['discriminator'])

    trained_GAN = IntegrativeGAN.load_from_checkpoint(args.GAN_checkpoint_path, xgenerator=xgenerator,
                                                      ygenerator=ygenerator, discriminator=discriminator,
                                                      projection_model=trained_projector,
                                                      data=data)
    integration_results = trained_GAN.project_the_data()

    integration_results_df = pd.DataFrame(integration_results.numpy())

    integration_results_df = integration_results_df.assign(id0=data.id0, id1=data.id1)

    integration_results_df.to_csv(args.results_path)
