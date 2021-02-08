import argparse
import os
import pickle
from integration_tools.utils.data.data_classes import DualDataset
from integration_tools.utils.transforms.transforms_classes import Concatenate, ToTensor
from torchvision import transforms

parser = argparse.ArgumentParser(description='data preparation module')
parser.add_argument('-data_path', '-d',
                    dest="data_path",
                    metavar='FILE',
                    help='path to the DualDataset')

parser.add_argument('-results_dir', '-r',
                    dest="results_dir",
                    metavar='FILE',
                    help='path to the results directory')

args = parser.parse_args()

if os.path.isfile(args.data_path):
    # Load the dual_dataset object which upon it the learning will be performed.
    with open(args.data_path, 'rb') as data_file:
        dual_entities_dataset = pickle.load(data_file)
# Not valid
else:
    raise FileNotFoundError(args.data_path)

indexes_of_entities_with_all_fields, indexes_of_entities_with_field0_only, indexes_of_entities_with_field1_only = \
    dual_entities_dataset.separate_to_groups()

xy_dataset = DualDataset.subset(dual_entities_dataset, indexes_of_entities_with_all_fields,transform =transforms.Compose([ToTensor()]))
x_dataset = DualDataset.subset(dual_entities_dataset, indexes_of_entities_with_field0_only,transform =transforms.Compose([ToTensor()]))
y_dataset = DualDataset.subset(dual_entities_dataset, indexes_of_entities_with_field1_only,transform =transforms.Compose([ToTensor()]))
concatenated_xy_dataset = DualDataset.subset(dual_entities_dataset, indexes_of_entities_with_all_fields,transform=transforms.Compose([ToTensor(),Concatenate()]))


with open(os.path.join(args.results_dir,'xy_dataset'),'wb') as output_file:
    pickle.dump(xy_dataset,output_file)

with open(os.path.join(args.results_dir,'concatenated_xy_dataset'),'wb') as output_file:
    pickle.dump(concatenated_xy_dataset,output_file)

with open(os.path.join(args.results_dir,'x_dataset'),'wb') as output_file:
    pickle.dump(x_dataset,output_file)

with open(os.path.join(args.results_dir,'y_dataset'),'wb') as output_file:
    pickle.dump(xy_dataset,output_file)



