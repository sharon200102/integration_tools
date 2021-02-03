from integration_tools.utils.data.data_classes import DualDataModule,ConcatDataset
from integration_tools.stdvae import StandardVAE_pl
import pytorch_lightning as pl
import torch
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import numpy as np
import pickle

data_path = Path(
    "/home/sharon200102/Documents/second degree/Research/microbiome_projects/microbiome/Projects/BGU_fatty_liver/data/data_used/entities_datasets/entities")

dm = DualDataModule(data_path)
xy_vae = StandardVAE_pl([271, 135, 50], torch.nn.LeakyReLU(), kld_coefficient=0.01, learning_rate=0.001)
runner = pl.Trainer(callbacks=[EarlyStopping('val_loss', patience=5)])
runner.fit(xy_vae, datamodule=dm)
trained_model = StandardVAE_pl.load_from_checkpoint(
    Path('lightning_logs/version_0/checkpoints/epoch=30-step=1704.ckpt'), layers_structure=[271, 135, 50],
    activation_fn=torch.nn.LeakyReLU(), kld_coefficient=0.01, learning_rate=0.001)

xy_dataset = dm.xy_dataset
latent_representation_list = []

with torch.no_grad():
    trained_model.eval()
    for entity_dict in xy_dataset:
        field0, field1 = entity_dict['FIELD0'], entity_dict['FIELD1']
        try:
            latent_representation = trained_model(torch.cat([field0, field1], dim=0))[4]
        except RuntimeError:
            latent_representation = trained_model(torch.cat([field0, field1], dim=1))[4]
        latent_representation_list.append(latent_representation.squeeze().numpy())

    latent_representation_df = pd.DataFrame(data=latent_representation_list)


projected_dataset = torch.tensor(np.array(latent_representation_df)).float()
projected_dataset = torch.utils.data.TensorDataset(projected_dataset)
concatds=ConcatDataset(xy_dataset,projected_dataset)

data_loader=torch.utils.data.DataLoader(concatds,10)
for batch in data_loader:
    print(batch)

with open(Path('dataset/concatds'),'wb') as file:
    pickle.dump(concatds,file)