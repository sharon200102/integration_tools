import itertools
from typing import List, Any
import torch
import pickle
from .stdvae import StandardVAE
from .mulvae import MultipleVAE
from .utils.models.learning_utils import make_train_step, early_stopping
import os
import pandas as pd

# Global variables which translate the names into the corresponding functions and objects
activation_fn_dict = {'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU, 'sigmoid': torch.nn.Sigmoid,
                      'leaky relu': torch.nn.LeakyReLU}
optimizers_dict = {'adam': torch.optim.Adam, 'Adadelta': torch.optim.Adadelta, 'Adagrad': torch.optim.Adagrad}


class Data_inegrator:
    def __init__(self, parse_args):
        """
        The idea of these classes, is to transform the inputs inserted by the user into a data integration tool. The
        Data_inegrator will iterate over all different configurations and exploit the best one to project the data
        into its latent representation.
        :param parse_args: which can be achieved by using the parser module in the parser package.
        """
        with open(parse_args.data_path, 'rb') as data_file:
            self.dual_entities_dataset = pickle.load(data_file)
        self.results_path = parse_args.results_path
        # find the patient which consists of both saliva and stool, and the patient without one of them.
        indexes_of_entities_with_all_fields, indexes_of_entities_with_field0_only, indexes_of_entities_with_field1_only = self.dual_entities_dataset.separate_to_groups()
        self.xy_dataset = torch.utils.data.Subset(self.dual_entities_dataset, indexes_of_entities_with_all_fields)
        self.x_dataset = torch.utils.data.Subset(self.dual_entities_dataset, indexes_of_entities_with_field0_only)
        self.y_dataset = torch.utils.data.Subset(self.dual_entities_dataset, indexes_of_entities_with_field1_only)

        self.x_architecture = parse_args.x_architecture
        self.y_architecture = parse_args.y_architecture
        self.xy_architecture = parse_args.xy_architecture

        # train and validate only according to the patients with both fields.

        train_size = int(parse_args.train_size * len(self.xy_dataset))
        validation_size = len(self.xy_dataset) - train_size

        # split the data into train and validation
        self.xy_train_dataset, self.xy_validation_dataset = torch.utils.data.random_split(self.xy_dataset,
                                                                                          [train_size, validation_size])
        # Creating the dataloaders for the xy_dataset.

        self.xy_train_dataloader = torch.utils.data.DataLoader(self.xy_train_dataset, shuffle=True,
                                                               batch_size=parse_args.batch_size)

        self.xy_validation_dataloader = torch.utils.data.DataLoader(self.xy_validation_dataset,
                                                                    batch_size=len(self.xy_validation_dataset),
                                                                    shuffle=True)

        self.learning_rate_list = parse_args.learning_rate
        self.activation_fn = activation_fn_dict[parse_args.activation_fn]()
        self.optimizer = optimizers_dict[parse_args.optimizer]
        self.latent_layer_size_list = parse_args.latent_representation
        self.klb_coefficient_list = parse_args.klb_coefficient
        self.patience = parse_args.patience

        self.configuration_list = self._create_all_configurations()

    def _create_all_configurations(self) -> List[dict]:
        """
        Create all possible combinations of (learning_rate.latent_layer_size,klb_coefficient) that can be extracted
        from the user's input
        :return: A list of configurations, where each one is represented by a dictionary.
        """
        parameters_list = itertools.product(self.learning_rate_list, self.latent_layer_size_list,
                                            self.klb_coefficient_list)
        configuration_list = [
            {'learning_rate': learning_rate, 'latent_layer_size': latent_layer_size, 'klb_coefficient': klb_coefficient}
            for learning_rate, latent_layer_size, klb_coefficient in parameters_list]
        return configuration_list

    def find_best_configuration(self):
        best_results_of_each_configuration = []
        best_model_loss = 10 ** 10
        for configuration in self.configuration_list:
            learning_rate = configuration['learning_rate']
            latent_layer_size = configuration['latent_layer_size']
            klb_coefficient = configuration['klb_coefficient']

            print(
                'A model with LR : {lr} and latent_size : {latent_size} and klb coefficient : {klb}  Is now runing'.format(
                    lr=learning_rate, latent_size=latent_layer_size,
                    klb=klb_coefficient))

            # Construct the models according to the current configuration
            xy_vae = StandardVAE(self.xy_architecture + [latent_layer_size], self.activation_fn,kld_coefficient=klb_coefficient)
            x_vae = StandardVAE(self.x_architecture + [latent_layer_size], self.activation_fn,kld_coefficient=klb_coefficient)
            y_vae = StandardVAE(self.y_architecture + [latent_layer_size], self.activation_fn,kld_coefficient=klb_coefficient)
            # Create the full VAE based on the standardVAE's above.
            full_vae = MultipleVAE(xy_vae, x_vae, y_vae)
            optimizer = self.optimizer(full_vae.parameters(), lr=learning_rate)
            train_step_function = make_train_step(full_vae, optimizer)

            # Train the model
            average_validation_sample_loss_per_epoch = []
            epoch = 0


            # Train full model.

            stop = False

            # Use early stopping.

            while not stop:

                epoch += 1
                # Do an epoch on the training set.

                for entities_train_batch in self.xy_train_dataloader:
                    field0_batch, field1_batch = entities_train_batch['FIELD0'], entities_train_batch['FIELD1']
                    # If for some reason the batch has more dimensions than expected, squeeze it.
                    if len(field0_batch.shape) > 2:
                        field0_batch, field1_batch = entities_train_batch['FIELD0'].squeeze(), entities_train_batch[
                            'FIELD1'].squeeze()

                    train_step_function(field0_batch, field1_batch)
                # Do validation
                total_validation_loss_per_epoch = 0
                with torch.no_grad():
                    full_vae.eval()
                    for entities_validation_batch in self.xy_validation_dataloader:
                        field0_batch, field1_batch = entities_validation_batch['FIELD0'].squeeze(), \
                                                     entities_validation_batch[
                                                         'FIELD1'].squeeze()
                        forward_dict = full_vae(field0_batch, field1_batch)
                        # Computes reconstruction loss and according to it decide whether to stop.
                        loss_dict = full_vae.loss_function(forward_dict)
                        total_validation_loss_per_epoch += float(loss_dict['xvae_loss']['Reconstruction_Loss'] + \
                                                                 loss_dict['xyvae_loss']['Reconstruction_Loss'] + \
                                                                 loss_dict['yvae_loss']['Reconstruction_Loss'])
                    # Compute the average validation sample loss.
                    average_validation_sample_loss = total_validation_loss_per_epoch / len(self.xy_validation_dataset)
                    # If the loss achieved on the validation is the smallest until so far, save the model.
                    if average_validation_sample_loss < best_model_loss:
                        torch.save(full_vae, os.path.join(self.results_path, 'best_model.pt'))
                        best_model_loss = average_validation_sample_loss
                    # keep tracking of the validation loss in every epoch.
                    average_validation_sample_loss_per_epoch.append(average_validation_sample_loss)
                    stop = early_stopping(average_validation_sample_loss_per_epoch, patience=self.patience,
                                          ascending=False)
            configuration_best_result = min(average_validation_sample_loss_per_epoch)
            print('\nThe model best loss achieved on the validation set is : {}  '.format(configuration_best_result))
            print('\nThe training stopped after {epochs} epochs'.format(epochs=epoch))
            best_results_of_each_configuration.append(configuration_best_result)

        configuration_results_df = pd.DataFrame(self.configuration_list)
        configuration_results_df = configuration_results_df.assign(best_loss=best_results_of_each_configuration)
        configuration_results_df.to_csv(os.path.join(self.results_path, 'models_results.csv'))

    def project_all_data(self):
        latent_representation_list = []
        entities_id0_list = []
        entities_id1_list = []
        self.dual_entities_dataset.dict_retrieval_flag = 0
        full_vae = torch.load(os.path.join(self.results_path, 'best_model.pt'))

        with torch.no_grad():
            full_vae.eval()
            for entity in self.dual_entities_dataset:
                entities_id0_list.append(entity.id0)
                entities_id1_list.append(entity.id1)

                if entity.get_status() == 0:
                    field0, field1 = entity.field0, entity.field1
                    try:
                        latent_representation = full_vae.xyvae(torch.cat([field0, field1], dim=0))[4]
                    except RuntimeError:
                        latent_representation = full_vae.xyvae(torch.cat([field0, field1], dim=1))[4]

                elif entity.get_status() == 1:
                    field0 = entity.field0
                    latent_representation = full_vae.xvae(field0)[4]
                else:
                    field1 = entity.field1
                    latent_representation = full_vae.yvae(field1)[4]
                latent_representation_list.append(latent_representation.squeeze().numpy())
            latent_representation_df = pd.DataFrame(data=latent_representation_list)
            latent_representation_df=latent_representation_df.assign(id0=entities_id0_list,id1=entities_id1_list)
            latent_representation_df.to_csv(os.path.join(self.results_path, 'latent_representation.csv'))
