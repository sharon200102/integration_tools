
import os
from typing import List, Any
import pickle
import yaml

def load_yaml(path: str) ->object:
    with open(path, 'r') as file:
        try:
            parameters= yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return parameters


def load_pickle(path: str) -> object:
    """Dispatcher function, to load binary file

     Args:
    ------
       path (str): pickle path

    Return:
    -------
       obj (object): python object

    """
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)

    return obj



def _file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


def _dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def logarithmic_iterator(ls) -> List[Any]:
    # If l is a number and not a list
    if type(ls) == int or type(ls) == float:
        return [ls]
    else:

        if len(ls) == 3:
            start, end, factor = ls
            return _logarithmic_scale(start, end, factor)
        else:
            raise ValueError('Could not interoperate the range inserted')

def linear_iterator(ls) -> List[Any]:
    # If l is a number and not a list

    if type(ls) == int or type(ls) == float:
        return [ls]
    else:
        if len(ls) == 2:
            start, end = ls
            return [start + i for i in range(int(end-start+1))]
        if len(ls) == 3:
            start, end, jump = ls
            return [start+(i*jump) for i in range(int((end-start)/jump)+1)]



"""The function is similar to range function, frange creates a fractional range of numbers"""


def _logarithmic_scale(start: float,end:float,factor:float):
    import math
    max_iterations = int(math.log(end/start,factor))
    return [start*(factor**power) for power in range(max_iterations+1)]

def create_search_space_dict(hyper_parameters_dict:dict,scale_dict:dict) -> dict:
    parameters_space_dict = {}
    for key,value in hyper_parameters_dict.items():
        scale_type = scale_dict[key]
        if  scale_type == 'linear_scale':
            parameters_space_dict[key] = linear_iterator(value)
        elif scale_type == 'log_scale':
            parameters_space_dict[key] = logarithmic_iterator(value)
    return parameters_space_dict
