import torch
from ...utils.data.data_classes import DualEntity


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample_dict_copy = dict(sample.entity_dict)
        if sample_dict_copy.get('FIELD0', None) is not None:
            sample_dict_copy['FIELD0'] = torch.tensor(sample.entity_dict['FIELD0'].values.astype('float64')).type(torch.FloatTensor)
        if sample_dict_copy.get('FIELD1', None) is not None:
            sample_dict_copy['FIELD1'] = torch.tensor(sample.entity_dict['FIELD1'].values.astype('float64')).type(torch.FloatTensor)
        return DualEntity(sample_dict_copy)

