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
class Sampler(Generic[T_co]):
    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError
class UncertaintySampler(Sampler[int]):
    data_source: Sized
    # replacement: bool

    def __init__(self, data_source: Sized,
                 prob_list=[]) -> None:
        self.data_source = data_source
        self.prob_list = prob_list

        # if not isinstance(self.replacement, bool):
        #     raise TypeError("replacement should be a boolean value, but got "
        #                     "replacement={}".format(self.replacement))
        
    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __len__(self) -> int:
        return self.num_samples
    
    def __iter__(self) -> Iterator[int]:
        least_conf = least_confidence(self.prob_list, False)
        print("least_conf: ", least_conf)


def least_confidence(prob_dist):
    # most confidence prediction
    simple_least_conf = torch.max(prob_dist)
    
    # number of data, for torch
    num_labels = prob_dist.numel()

    normalized_least_conf = (1-simple_least_conf) * (num_labels / (num_labels-1))

    return normalized_least_conf.item()