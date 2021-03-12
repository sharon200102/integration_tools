import pandas as pd
from torch.utils.data import Dataset
from collections.abc import Iterable
import pytorch_lightning as pl
import os
import pickle
import torch

# Introducing some global variables indicating the state of a an entity ,
# i.e consists of all fields, field zero only, field one only.


ALL_FIELDS = 0
FIELD_ZERO_ONLY = 1
FIELD_ONE_ONLY = 2

"""The concept of the whole package is to offer novel methods for integration of two datasets with partial overlap in samples.
More specifically given two related data-sets  denoted by X and Y where each sample i can have either the X vector xi  
the Y vector yi or both
The goal is to develop a machine learning architecture that will provide a latent representation of the integrated data-sets.
Therefore, its necessary to create a data object in which every sample can consist of Xi or Yi or both. """

"""DualDataset provides exactly that, every sample (DualEntity) can consist Field0,Field1 or both. """


class DualEntity():
    def __init__(self, entity_dict):
        self.entity_dict = entity_dict
        self.id0 = entity_dict.get('ID0', 'NO_ID0')
        self.id1 = entity_dict.get('ID1', 'NO_ID1')
        self.field0 = entity_dict.get('FIELD0', None)
        self.field1 = entity_dict.get('FIELD1', None)

    """The get_status function retrieves the fields status of the patient i.e whether all fields exists or otherwise 
    if one field is absent. """

    def get_status(self):
        # Both fields exist.
        if self.field0 is not None and self.field1 is not None:
            return ALL_FIELDS
        # field one is absent
        elif self.field0 is not None:
            return FIELD_ZERO_ONLY
        # field zero is absent
        return FIELD_ONE_ONLY

    def get_fields(self):
        entity_status = self.get_status()
        if entity_status == ALL_FIELDS:
            return pd.concat([self.field0, self.field1])
        elif entity_status == FIELD_ZERO_ONLY:
            return self.field0
        else:
            return self.field1


class DualDataset(Dataset):
    """The core of the object is an iterable of Dual_entities"""

    def __init__(self, dual_entities: Iterable, transform=None):
        """

        :param dual_entities : An iterable of DualEntity, which compose the data set structure
        :param transform: As the dataset class from which DualDataset inherits allows,
         a transform function can be inserted and activated on each sample when accessing it.
         for more info :https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        """
        self.entities = dual_entities
        self.transform = transform
        self.simplified = [entity.get_fields() for entity in self.entities]
        self.id0 =[entity.id0 for entity in self.entities ]
        self.id1 =[entity.id1 for entity in self.entities ]

    @classmethod
    def from_sources(cls, source0: pd.DataFrame, source1: pd.DataFrame = None, matching_info_source0=None,
                     matching_info_source1=None, matching_fn=None, separator: pd.Series = None, **kwargs):
        """
        The core function in the class, this function is the main way to construct the DualDataset object, and its supports
         many different inputs.
         As mentioned before a sample i is composed of xi or yi or both.
         Therefore it's necessary to find x and y that correspond to the same sample i.

        :param source0: A dataframe, if the datasets are represented in different tables than source0 should be the first table.
        otherwise,if the datasets are represented in different tables than source1 should be the first table.
        :param source1: if the datasets are represented in different tables than source1 should be the second table.
        otherwise it should be None.
        :param matching_info_source0: info on each sample x in source0 that allows to identify that x belongs to an entity i
        if the identifier is a number than matching_info_source0 should be a series,
        if the identifier is a series than matching_info_source0 should be a dataframe.
        if matching_info_source0 is None than the identifier of each sample is its source0 corresponding index.
        :param matching_info_source1:
        info on each sample y in source1 that allows to identify that y belongs to an entity i
        if the identifier is a number than matching_info_source1 should be a series,
        if the identifier is a series than matching_info_source1 should be a dataframe.
        if matching_info_source1 is None than the identifier of each sample is its source1 corresponding index.
        Should be None if source1 is None.
        :param matching_fn: A function which will determine whether samples x,y belong to the same entity.
        the function should accept two parameters
        (the matching_info corresponding to x and the matching_info corresponding to y) and to return true or false if
        they belong to the same entity.
        If None, than the matching_fn is mathcing_info_of_x.equals(mathcing_info_of_y)

        :param separator: only relevant if source1 is None, i.e all the samples are composed in source0, the separator
         should be a binary series (two unique values) that indicates to each sample to what dataset does it belongs to.
        :param kwargs:
        :return: DualDataset object.
        """
        dual_entities_list = []
        source0, source1, matching_info_source0, matching_info_source1, matching_fn = cls._adjust_inputs(source0,
                                                                                                         source1,
                                                                                                         matching_info_source0,
                                                                                                         matching_info_source1,
                                                                                                         matching_fn,
                                                                                                         separator)
        matching_couples = cls._find_matches(matching_info_source0, matching_info_source1, matching_fn)
        for index_field0, index_field1 in matching_couples:
            entity_dict = {}
            if index_field0 is not None:
                field0 = source0.iloc[index_field0].copy()
                entity_dict['FIELD0'] = field0
                entity_dict['ID0'] = source0.index[index_field0]
            if index_field1 is not None:
                field1 = source1.iloc[index_field1].copy()
                entity_dict['FIELD1'] = field1
                entity_dict['ID1'] = source1.index[index_field1]
            dual_entities_list.append(DualEntity(entity_dict))
        return cls(dual_entities_list, **kwargs)

    @staticmethod
    def _adjust_inputs(source0: pd.DataFrame, source1: pd.DataFrame = None, matching_info_source0=None,
                       matching_info_source1=None, matching_fn=None, separator: pd.Series = None):
        """ The function executes the actions introduced in from_sources function"""
        if separator is not None:  # If all sources are combined in one table (source0) and all matching info is combined in  matching_info_source0, then split by separator
            if len(separator.unique()) != 2:  # i.e the separator is not binary.
                raise ValueError("Separator must be composed of two unique values only")
            else:
                all_matching_info = matching_info_source0
                all_sources = source0
                # find the unique labels of the separator series.
                zero_sep_value, first_sep_value = sorted(separator.unique())

                # Separate the combined datasets into two subsets according to the separator labels.
                source0 = all_sources[separator == zero_sep_value].copy()
                source1 = all_sources[separator == first_sep_value].copy()
                # Separate the combined matching_info_tables into two subsets according to the separator labels.
                matching_info_source0 = all_matching_info[separator == zero_sep_value].copy()
                matching_info_source1 = all_matching_info[separator == first_sep_value].copy()
        # If no matching info was given use the indices of the sources.
        if matching_info_source0 is None and matching_info_source1 is None:
            matching_info_source0 = source0.index.to_series()
            matching_info_source1 = source1.index.to_series()
        # If matching_fn is None use the equality function according to the type of the matching_info's
        if matching_fn is None:
            if isinstance(matching_info_source0, pd.Series):
                matching_fn = lambda x, y: x == y
            elif isinstance(matching_info_source0, pd.DataFrame):
                matching_fn = lambda x, y: x.eq(y).all()
            else:
                raise ValueError
        return source0, source1, matching_info_source0, matching_info_source1, matching_fn

    @staticmethod
    def _find_matches(matching_info_source0, matching_info_source1, matching_fn):
        """match a sample x to a sample y according to what was previously introduced in from_sources function """
        matching_couples = []
        indices_of_not_matched_source0 = list(range(len(matching_info_source0)))
        indices_of_not_matched_source1 = list(range(len(matching_info_source1)))

        for index_source0 in range(len(matching_info_source0)):
            field_zero_info = matching_info_source0.iloc[index_source0]
            for index_source1 in range(len(matching_info_source1)):
                field_one_info = matching_info_source1.iloc[index_source1]
                if index_source0 in indices_of_not_matched_source0 and index_source1 in indices_of_not_matched_source1:  # If none of them was matched so far
                    if matching_fn(field_zero_info,
                                   field_one_info):  # IF the couple match, insert their indices to the list
                        matching_couples.append((index_source0, index_source1))
                        indices_of_not_matched_source0.remove(index_source0)  # Remove the indices from the not matched.
                        indices_of_not_matched_source1.remove(index_source1)

                        break
                    else:
                        pass
        # don't forget to add the ones that didn't match to anybody.

        for not_matched_index_source0 in indices_of_not_matched_source0:
            matching_couples.append((not_matched_index_source0, None))
        for not_matched_index_source1 in indices_of_not_matched_source1:
            matching_couples.append((None, not_matched_index_source1))

        return matching_couples

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.entities[idx]
        else:
            return self.transform(self.entities[idx])

    @classmethod
    def subset(cls, full_data, indices: list, **kwargs):
        subset_of_entities = [full_data.entities[subset_index] for subset_index in indices]
        return cls(subset_of_entities, **kwargs)

    def separate_to_groups(self):
        indexes_of_entities_with_all_fields = [self.entities.index(entity) for entity in self.entities if
                                               entity.get_status() == ALL_FIELDS]
        indexes_of_entities_with_field1_only = [self.entities.index(entity) for entity in self.entities if
                                                entity.get_status() == FIELD_ZERO_ONLY]
        indexes_of_entities_with_field2_only = [self.entities.index(entity) for entity in self.entities if
                                                entity.get_status() == FIELD_ONE_ONLY]
        return indexes_of_entities_with_all_fields, indexes_of_entities_with_field1_only, indexes_of_entities_with_field2_only


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class TrainAndValidateDataModule(pl.LightningDataModule):
    def __init__(self, data, train_batch_size=10, validation_batch_size=10, train_size=0.8):
        super().__init__()
        self.data = data
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.train_size = train_size

    def setup(self, stage: str = None):
        train_size = int(self.train_size * len(self.data))
        validation_size = len(self.data) - train_size
        # train and validate only according to the patients with both fields.

        self.train_data, self.validation_data = torch.utils.data.random_split(self.data, [train_size, validation_size])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, shuffle=True,
                                           batch_size=self.train_batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_data, batch_size=self.validation_batch_size,
                                           shuffle=False)


class TrainOnlyDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size: int = 10):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data, self.batch_size, shuffle=True)
