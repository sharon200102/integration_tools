import torch
from ...utils.data.data_classes import DualEntity


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample_dict_copy = dict(sample.entity_dict)
        if sample_dict_copy.get('FIELD0', None) is not None:
            sample_dict_copy['FIELD0'] = torch.tensor(sample.entity_dict['FIELD0'].values.astype('float64')).type(
                torch.FloatTensor)
        if sample_dict_copy.get('FIELD1', None) is not None:
            sample_dict_copy['FIELD1'] = torch.tensor(sample.entity_dict['FIELD1'].values.astype('float64')).type(
                torch.FloatTensor)
        return DualEntity(sample_dict_copy)




class EntityToTensor(object):
    """Convert an Tensor entity to tensor based on its fields"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, sample: DualEntity):
        if sample.field0 is not None and sample.field1 is not None:
            return torch.cat([sample.field0, sample.field1], **self.kwargs)
        elif sample.field0 is not None:
            return sample.field0
        else:
            return sample.field1


class Todict(object):
    """Convert the entity to its corresponding dict."""

    def __call__(self, sample):
        sample_dict_copy = dict(sample.entity_dict)
        return sample_dict_copy
