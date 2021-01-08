import pandas as pd
from torch.utils.data import Dataset
from collections.abc import Iterable

ALL_FIELDS = 0
FIELD_ZERO_ONLY = 1
FIELD_ONE_ONLY = 2

"""A basic patient class"""
"""This object will compose the patient_dataset object"""


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


class DualDataset(Dataset):

    def __init__(self, dual_entities: Iterable, transform=None):
        self.entities = dual_entities
        self.transform=transform
        self.dict_retrieval_flag = 1

    @classmethod
    def from_sources(cls, source0: pd.DataFrame, source1: pd.DataFrame = None, matching_info_source0=None,
                     matching_info_source1=None, matching_fn=None, separator: pd.Series = None,**kwargs):

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

        if separator is not None:  # If all sources are combined in one table (source0) and all matching info is combined in  matching_info_source0, then split by separator
            if len(separator.unique() != 2):  # i.e the separator is not binary.
                raise ValueError("Separator must be composed of two unique values only")
            else:
                all_matching_info = matching_info_source0
                all_sources = source0

                zero_sep_value, first_sep_value = sorted(separator.unique())

                source0 = all_sources[separator == zero_sep_value].copy()
                source1 = all_sources[separator == first_sep_value].copy()

                matching_info_source0 = all_matching_info[separator == zero_sep_value].copy()
                matching_info_source1 = all_matching_info[separator == first_sep_value].copy()

        if matching_info_source0 is None and matching_info_source1 is None:
            matching_info_source0 = source0.index.to_series()
            matching_info_source1 = source1.index.to_series()

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
            retrieved_entity = self.entities[idx]
        else:
            retrieved_entity = self.transform(self.entities[idx])
        return retrieved_entity.entity_dict if self.dict_retrieval_flag else retrieved_entity

    def separate_to_groups(self):
        indexes_of_entities_with_all_fields = [self.entities.index(entity) for entity in self.entities if
                                               entity.get_status() == ALL_FIELDS]
        indexes_of_entities_with_field1_only = [self.entities.index(entity) for entity in self.entities if
                                                entity.get_status() == FIELD_ZERO_ONLY]
        indexes_of_entities_with_field2_only = [self.entities.index(entity) for entity in self.entities if
                                                entity.get_status() == FIELD_ONE_ONLY]
        return indexes_of_entities_with_all_fields, indexes_of_entities_with_field1_only, indexes_of_entities_with_field2_only
