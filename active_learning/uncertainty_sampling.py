'''
pool-based sampling -> scrnario
uncertainty-based sampling -> query strategy framework
example url: https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py
'''


import torch
from torch import Tensor

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

T_co = TypeVar('T_co', covariant=True)

'''
Every Sampler subclass has to provide an __iter__() method, 
providing a way to iterate over indices of dataset elements, 
and a __len__() method that returns the length of the returned iterators.
'''

def least_confidence(probs, egs):
    # most confidence prediction
    simple_least_conf = torch.max(torch.as_tensor(probs))
    
    # number of data, for torch
    num_labels = probs.numel()

    normalized_least_conf = (1-simple_least_conf) * (num_labels / (num_labels-1))

    return normalized_least_conf.item()