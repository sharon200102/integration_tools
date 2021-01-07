import argparse
import os
from typing import List, Any


def parse_parameters(parser: argparse.ArgumentParser):
    supported_activation_fn = ['tanh', 'relu', 'sigmoid', 'leaky relu']
    supported_optimizers = ['adam', 'Adadelta', 'Adagrad']
    parser.add_argument("data_path", type=_file_path, help="The path to the data object which contains the ")

    parser.add_argument("results_path", type=_dir_path, help="The path to the folder in which the results should be "
                                                            "placed", default=os.getcwd())

    parser.add_argument('-xy', '--xy_architecture', nargs='+', help='The architecture of the xy vae', type=int)
    parser.add_argument('-x', '--x_architecture', nargs='+', help='The architecture of the x vae', type=int)
    parser.add_argument('-y', '--y_architecture', nargs='+', help='The architecture of the y vae', type=int)
    parser.add_argument('-t', '--train_size', type=float, help='The percentage of the train out of the whole data ',
                        default=0.8)
    parser.add_argument('-b', '--batch_size', type=int, help='The batch size of the train loader ', default=10)
    parser.add_argument('-l', '--learning_rate', type=_float_iterator,
                        help='The learning rate for the vae architecture ',
                        default=[0.01])
    parser.add_argument('-a', '--activation_fn', type=str, help='The activation function that will be applied after '
                                                                'each layer ', default='relu',
                        choices=supported_activation_fn)
    parser.add_argument('-o', '--optimizer', type=str, help='The optimizer that will be used during the training '
                                                            'process', default='adam', choices=supported_optimizers)

    parser.add_argument('-klb', '--klb_coefficient', type=_float_iterator, help='The klb coefficient of the loss '
                                                                               'function for '
                                                                               'each of the vaes', default=[1.0])
    
    parser.add_argument('-p', '--patience', type=int, help='The number of epochs until halting the training process '
                                                           'when there is no improvement on the validation loss',
                        default=5)
    parser.add_argument('-la', '--latent_representation', type=_int_iterator, help='Projection desired dimension',
                        default=[10])

    return parser.parse_args()


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


def _float_iterator(string: str) -> List[Any]:
    try:
        float(string)
        return [float(string)]
    except ValueError:
        split_frange = string.split(',')
        if len(split_frange) == 2:
            start, end = split_frange
            return frang(int(start), int(end))
        if len(split_frange) == 3:
            start, end, jump = split_frange
            return frang(int(start), int(end), int(jump))
        if len(split_frange) == 4:
            start, end, jump, divide = split_frange
            return frang(int(start), int(end), int(jump), int(divide))


def _int_iterator(string: str) -> List[Any]:
    try:
        int(string)
        return [int(string)]
    except ValueError:
        split_frange = string.split(',')
        if len(split_frange) == 2:
            start, end = split_frange
            return list(range(int(start), int(end)))
        if len(split_frange) == 3:
            start, end, jump = split_frange
            return list(range(int(start), int(end), int(jump)))


"""The function is similar to range function, frange creates a fractional range of numbers"""


def frang(start: int, end: int, jump: int = 1, divide: int = 1, transform_to_int=False) -> List[Any]:
    """
    start: the int which begins the  fractional range
    end: the int which ends the range
    jump: the step between the numbers in the range
    divide: the constant which all numbers will be divided according to it.
    """
    if not transform_to_int:
        return list(map(lambda item: item / divide, range(start, end, jump)))
    else:
        return list(map(lambda item: int(item / divide), range(start, end, jump)))
