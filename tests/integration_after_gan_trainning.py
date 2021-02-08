from integration_tools.IntegrativeGAN import IntegrativeGAN
from integration_tools.ProjectionModels.stdvae import StandardVAE_pl
from pathlib import Path
import pickle
import torch
import pandas as pd
import numpy as np
import os
data_path = Path('../../microbiome_projects/microbiome/Projects/BGU_fatty_liver/data/data_used/entities_datasets/entities')
result_path = Path('../../microbiome_projects/microbiome/Projects/BGU_fatty_liver/data/exported_data/latent_representation_data')
with open(data_path, 'rb') as data_file:
    dual_entities_dataset = pickle.load(data_file)

vae_parameters_dict = {'layers_structure': [271, 135, 40], 'kld_coefficient': 0.01, 'learning_rate': 0.001}
gan_core_parameters_dict = {'xgenerator_architecture': [203,100], 'ygenerator_architecture': [68],
                            'discriminator_architecture': [20], 'latent_layer_size': 40}
gan_hyper_parameters = {'learning_rate': 0.0016}

gan_parameters_dict = {'config':gan_hyper_parameters,'core_parameters':gan_core_parameters_dict}

vae_checkpoint_path = Path(
    '../../microbiome_projects/microbiome/Projects/BGU_fatty_liver/data/exported_data/latent_representation_data/epoch=26-step=1484.ckpt')
gan_checkpoint_path = Path('../../microbiome_projects/microbiome/Projects/BGU_fatty_liver/data/exported_data/latent_representation_data/integration_models/tune_integration_model/best_configuration/checkpoints/best_model.ckpt')

trained_vae_model = StandardVAE_pl.load_from_checkpoint(checkpoint_path=vae_checkpoint_path, **vae_parameters_dict)
trained_gan_model = IntegrativeGAN.load_from_checkpoint(checkpoint_path=gan_checkpoint_path, **gan_parameters_dict)

dual_entities_dataset.dict_retrieval_flag=0

latent_representation_list = []
entities_id0_list = []
entities_id1_list = []
with torch.no_grad():
    trained_gan_model.eval()
    trained_vae_model.eval()
    # Iterate over all samples.
    for entity in dual_entities_dataset:
        entities_id0_list.append(entity.id0)
        entities_id1_list.append(entity.id1)
        if entity.get_status() == 0:
            field0, field1 = entity.field0, entity.field1
            try:
                latent_representation = trained_vae_model(torch.cat([field0, field1], dim=0))[4]
            except RuntimeError:
                latent_representation = trained_vae_model(torch.cat([field0, field1], dim=1))[4]
        # if the entity consists only of X use xgenerator
        elif entity.get_status() == 1:
            field0 = entity.field0
            latent_representation = trained_gan_model.xgenerator(field0)
        # if the entity consists only of y use ygenerator

        elif entity.get_status() == 2:
            field1 = entity.field1
            latent_representation = trained_gan_model.ygenerator(field1)
        # Save all projections into a dataframe.
        latent_representation_list.append(latent_representation.squeeze().numpy())

    latent_representation_df = pd.DataFrame(data=latent_representation_list)
    latent_representation_df = latent_representation_df.assign(id0=entities_id0_list, id1=entities_id1_list)
    latent_representation_df.to_csv(os.path.join(result_path, 'latent_representation.csv'))
