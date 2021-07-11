from torchvision.transforms import transforms
import numpy
import pandas as pd
from integration_tools.utils.data.data_classes import DualDataset
import pytorch_lightning as pl
import argparse
from integration_tools.utils.parser.parser import load_yaml
from integration_tools.utils.parser.parser import load_pickle
from integration_tools.utils.transforms.transforms_classes import ToTensor
import integration_tools
import integration_tools.ProjectionModels as Projection_models
import integration_tools.generators_collection as generators_collection
import integration_tools.discriminators_collection as discriminator_collection
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integration after hyper parameters tuning')
    parser.add_argument('--GAN_checkpoint_path', '-g',
                        dest="GAN_checkpoint_path",
                        metavar='FILE',
                        help='path to the desired GAN checkpoint')
    parser.add_argument('--params_path', '-p',
                        dest="models_configs_path",
                        metavar='FILE',
                        help='path to the parameters of the models')

    parser.add_argument('--projector_checkpoint_path', '-pr',
                        dest="projector_checkpoint_path",
                        metavar='FILE',
                        help='path to the desired projector checkpoint')

    parser.add_argument('--results_path', '-r',
                        dest="results_path",
                        metavar='FILE',
                        help='The path of where the results should ne saved')

    args = parser.parse_args()
    parameters = load_yaml(args.models_configs_path)

    data = load_pickle(parameters['exp_params']['data_path'])
    data = DualDataset.subset(data, list(range(len(data))), transform=transforms.Compose([ToTensor()]))

    projection_name = parameters['model_params']['projection_name']
    xgenerator_name = parameters['model_params']['x_generator_name']
    ygenerator_name = parameters['model_params']['y_generator_name']
    discriminator_name = parameters['model_params']['discriminator_name']
    GAN_name = parameters['model_params']['GAN_name']

    trained_projector = Projection_models.projection_models[projection_name].\
        load_from_checkpoint(args.projector_checkpoint_path)

    xgenerator = generators_collection.generator_models[xgenerator_name](**parameters['model_params']['x_generator_params'])

    ygenerator = generators_collection.generator_models[ygenerator_name](**parameters['model_params']['y_generator_params'])

    discriminator = discriminator_collection.discriminator_models[discriminator_name] \
        (**parameters['model_params']['discriminator_params'])

    trained_GAN = integration_tools.GAN_models[GAN_name].load_from_checkpoint(args.GAN_checkpoint_path, xgenerator=xgenerator,
                                                      ygenerator=ygenerator, discriminator=discriminator,
                                                      projection_model=trained_projector,
                                                      data=data, **parameters['model_params']['GAN_params'])

    integration_results = trained_GAN.project_the_data()

    integration_results_df = pd.DataFrame(integration_results.numpy())

    integration_results_df = integration_results_df.assign(id0=data.id0, id1=data.id1)

    integration_results_df.to_csv(args.results_path)

# --GAN_checkpoint_path "/home/sharon200102/Documents/second degree/Research/integration_tools/GAN_checkpoints/0/last.ckpt" --projector_checkpoint_path "/home/sharon200102/Documents/second_degree/Research/integration_tools/StandardVAE_pl_checkpoints/last.ckpt" -s "/home/sharon200102/Documents/second_degree/Research/integration_tools/configs/reproduction_configs/config.yaml" --DualData_path "/home/sharon200102/Documents/second_degree/Research/microbiome_projects/microbiome/Projects/BGU_fatty_liver/data/data_used/entities_datasets/entities" -r "/home/sharon200102/Documents/second degree/Research/microbiome_projects/microbiome/Projects/BGU_fatty_liver/data/exported_data/latent_representation_data/latent_representation.csv"
# --GAN_checkpoint_path results/GDM/batch_discrimination_with_batch_normalization/GAN_checkpoints/0/last.ckpt -s configs/reproduction_configs/GDM/config.yaml -p results/GDM/batch_discrimination_with_batch_normalization/StandardVAE_pl_checkpoints/0/last.ckpt -d "/home/sharon200102/Documents/second_degree/Research/microbiome_projects/microbiome/Projects/GDM/experiment1/data/data_used/entities_datasets" -r "/home/sharon200102/Documents/second_degree/Research/microbiome_projects/microbiome/Projects/GDM/experiment1/data/exported_data/latent_representation_data/GAN/lastent_representation.csv"
#--GAN_checkpoint_path results/GDM/feature_matching_with_batch_normalization/GAN_checkpoints/48/last.ckpt -p configs/reproduction_configs/GDM/feature_matching_with_batch_normalization/params.yaml -pr results/GDM/feature_matching_with_batch_normalization/StandardVAE_pl_checkpoints/48/last.ckpt -r "/home/sharon200102/Documents/second_degree/Research/microbiome_projects/microbiome/Projects/GDM/experiment1/data/exported_data/latent_representation_data/GAN/latent_representation.csv"
#--GAN_checkpoint_path results/Fatty_liver/batch_discrimination/GAN_checkpoints/36/last.ckpt -p configs/reproduction_configs/BGU/batch_discrimination/params.yaml/ -pr results/Fatty_liver/batch_discrimination/StandardVAE_pl_checkpoints/36/last.ckpt -r "/home/sharon200102/Documents/second_degree/Research/microbiome_projects/microbiome/Projects/BGU_fatty_liver/data/exported_data/latent_representation_data/GAN/batch_discrimination/latent_representation"
#-g results/GDM_experiment2_israeli/feature_matching_and_batch_discrimination/GAN_checkpoints/0/last.ckpt -p configs/integration_configs/GAN_configs/GDM_experiment2_israel/feature_matching_and_batch_discrimination/params.yaml -pr results/GDM_experiment2_israeli/feature_matching_and_batch_discrimination/StandardVAE_pl_checkpoints/0/last.ckpt  -r /home/sharon200102/Documents/second_degree/Research/microbiome_projects/microbiome/Projects/GDM/experiment2/data/used_data/integration_projections/israeli_integration_projection.csv