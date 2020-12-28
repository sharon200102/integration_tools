import pandas as pd
from torch.utils.data import Dataset
from collections.abc import Iterable

ALL_FIELDS = 0
FIELD_ZERO_ONLY = 1
FIELD_ONE_ONLY = 2

"""A basic patient class"""
"""This object will compose the patient_dataset object"""


class patient_samples_single_tp():
    def __init__(self, patient_dict):
        self.patient_dict = patient_dict
        self.id = patient_dict.get('ID', 'NO_ID')
        self.time_point = patient_dict.get('TP', 'NO_TP')
        self.field0 = patient_dict.get('FIELD0', None)
        self.field1 = patient_dict.get('FIELD1', None)

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


class patient_dataset(Dataset):

    def __init__(self, otu_df, mapping_table, patient_id_name, time_name, sample_type_name=None, matching_fn=None,
                 transform=None):
        """
        otu_df - If the two otu types are combined in one table than otu_df parameter should be a single data frame,
        otherwise a list of dataframes.

        mapping_table - Similarly to the otu_df, If the mapping info of the two types
        is combined in one table than mapping_table parameter should be a single data frame, otherwise a list of
        dataframes.

        patient_id_name - if mapping_table is a not a list than the parameter should contain the patient id column name in the mapping_table.
        otherwise a list of names corresponding to the list of mapping_tables.

        time_name - similarly to the previous one, this parameter should contain the name of the time column in the mapping table.
         should be a list if mapping_table is a list.

         The idea is that a specific id and timepoint identify at most one sample of each type.

         sample_type_name= Only relevant if the whole mapping info is combined in one mapping table, this parameter
         should contain the column name which separates between the samples of different types.

        """
        if matching_fn is None:
            matching_fn = lambda x, y: x == y

        self.dict_retrieval_flag = 1
        self.transform = transform
        patients_list = []
        if sample_type_name is not None:
            merged_df = pd.concat(
                [otu_df, mapping_table[patient_id_name], mapping_table[time_name], mapping_table[sample_type_name]],
                axis=1)
            unique_sample_types = list(sorted(merged_df[sample_type_name].unique()))
            for name, group in merged_df.groupby([patient_id_name, time_name]):
                patient_dict = {'ID': name[0], 'TP': [name[1]]}
                for i in range(len(group.index)):
                    current_sample = group.iloc[i]
                    current_microbiome_sample = current_sample.drop([sample_type_name, patient_id_name, time_name])
                    if unique_sample_types.index(current_sample[sample_type_name]) == 0:
                        patient_dict['FIELD0'] = current_microbiome_sample
                    else:
                        patient_dict['FIELD1'] = current_microbiome_sample
                patients_list.append(patient_samples_single_tp(patient_dict))
        else:
            all_unique_patient_ids = set.union(*[set(specific_mapping_table[specific_patient_id_name]) for
                                                 specific_mapping_table, specific_patient_id_name in
                                                 zip(mapping_table, patient_id_name)])

            for patient_id in all_unique_patient_ids:

                field_0_unique_tp, field_1_unique_tp = [
                    specific_mapping_table[specific_mapping_table[specific_patient_id_name] == patient_id][time] for
                    specific_mapping_table, specific_patient_id_name, time in
                    zip(mapping_table, patient_id_name, time_name)]
                matching_tp_list = self._find_matches(field_0_unique_tp, field_1_unique_tp, matching_fn)

                for matching_field_zero_tp, matching_field_one_tp in matching_tp_list:
                    patient_dict = {}
                    for i, (specific_otu, specific_mapping_table, patient_col_name, tp_col_name) in enumerate(
                            zip(otu_df, mapping_table, patient_id_name, time_name)):
                        if i == 0 and matching_field_zero_tp is not None:
                            patient_dict['FIELD0'] = specific_otu[
                                (specific_mapping_table[patient_col_name] == patient_id) & (specific_mapping_table[
                                                                                                tp_col_name] == matching_field_zero_tp)]

                        if i == 1 and matching_field_one_tp is not None:

                            matching_field_one_otu = specific_otu[
                                (specific_mapping_table[patient_col_name] == patient_id) & (specific_mapping_table[
                                                                                                tp_col_name] == matching_field_one_tp)]
                            if len(matching_field_one_otu) > 1:
                                print('hi')

                            patient_dict['FIELD1'] = matching_field_one_otu

                    if bool(patient_dict):  # check if dictionary is not empty, which means that there is no samples
                        # for the given patient_id and tp.
                        patient_dict.update({'ID': patient_id, 'TP': [matching_field_zero_tp, matching_field_one_tp]})
                    patients_list.append(patient_samples_single_tp(patient_dict))

        self.patients = patients_list

    @classmethod
    def no_mapping(cls, source0, source1,**kwargs):
        default_matching_name = 'id'
        sources = [source0, source1]
        sources_mapping = [pd.DataFrame(source.index.values, columns=[default_matching_name], index=source.index)
                           for source in sources]
        return cls(sources, sources_mapping, (default_matching_name, default_matching_name),
                   (default_matching_name, default_matching_name),**kwargs)

    @staticmethod
    def _find_matches(first_field_list, second_field_list, matching_fn):
        matching_couples = []
        second_field_iterator = second_field_list
        first_field_iterator = first_field_list
        for first_field in first_field_list:
            for second_field in second_field_iterator:
                if matching_fn(first_field, second_field):
                    matching_couples.append((first_field, second_field))
                    second_field_iterator = [item for item in second_field_iterator if item != second_field]
                    first_field_iterator = [item for item in first_field_iterator if item != first_field]
                    break
                else:
                    pass

        for first_field in first_field_iterator:
            matching_couples.append((first_field, None))
        for second_field in second_field_iterator:
            matching_couples.append((None, second_field))
        return matching_couples

        # don't forget to add the ones that didn't match to anybody.

    def __len__(self):
        return len(self.patient_groups)

    def __getitem__(self, idx):
        if self.transform is None:
            retrieved_patient = self.patients[idx]
        else:
            retrieved_patient = self.transform(self.patients[idx])
        return retrieved_patient.patient_dict if self.dict_retrieval_flag else retrieved_patient

    def separate_to_groups(self):
        indexes_of_patients_with_all_fields = [self.patients.index(patient) for patient in self.patients if
                                               patient.get_status() == ALL_FIELDS]
        indexes_of_patients_with_field1_only = [self.patients.index(patient) for patient in self.patients if
                                                patient.get_status() == FIELD_ZERO_ONLY]
        indexes_of_patients_with_field2_only = [self.patients.index(patient) for patient in self.patients if
                                                patient.get_status() == FIELD_ONE_ONLY]
        return indexes_of_patients_with_all_fields, indexes_of_patients_with_field1_only, indexes_of_patients_with_field2_only
