import argparse
import pandas as pd
import torch
import yaml
import os
import pickle
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision.transforms import transforms

from integration_tools import data_modules_for_projection_models, projection_models
from integration_tools.utils.data.data_classes import ConcatDataset
from integration_tools.utils.transforms.transforms_classes import ToTensor

parser = argparse.ArgumentParser(description='Generic runner for projection models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        parameters_dict = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

data_path = parameters_dict['program_params']['data_path']
model_name = parameters_dict['program_params']['model_name']
model_constructor = projection_models[model_name]
results_dir = parameters_dict['program_params']['results_dir']
if os.path.isfile(data_path):
    # Load the dual_dataset object which upon it the learning will be performed.
    with open(data_path, 'rb') as data_file:
        dataset = pickle.load(data_file)
# Not valid
else:
    raise FileNotFoundError(data_path)
dataset.dict_retrieval_flag=0
data_params ={}
if 'data_params' in parameters_dict:
    data_params = parameters_dict['data_params']
dm = data_modules_for_projection_models[model_name](dataset,**data_params)

model = model_constructor(**parameters_dict['model_params'])

checkpoint_callback = ModelCheckpoint(dirpath=results_dir)
callbacks = [checkpoint_callback]

if 'patience' in parameters_dict['program_params']:
    callbacks.append(EarlyStopping(parameters_dict['program_params']['monitor'], patience=parameters_dict['program_params']['patience']))

runner = Trainer(callbacks=callbacks,default_root_dir=results_dir)


runner.fit(model=model,datamodule=dm)
trained_model = model_constructor.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path,**parameters_dict['model_params'])
latent_representation_list = []

with torch.no_grad():
    trained_model.eval()
    for element in dataset:
        latent_representation = trained_model(element)[4]
        latent_representation_list.append(latent_representation.squeeze().numpy())

latent_representation_df = pd.DataFrame(data=latent_representation_list)
projected_dataset = torch.tensor(np.array(latent_representation_df)).float()
dataset.transform=transforms.Compose([ToTensor()])
dataset.dict_retrieval_flag=1
cdm = ConcatDataset(dataset,projected_dataset)

with open(os.path.join(results_dir,'xy_dataset_and_ground_truth'),'wb') as output_file:
    pickle.dump(cdm,output_file)


