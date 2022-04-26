'''
pool-based sampling -> scrnario
uncertainty-based sampling -> query strategy framework
example url: https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py
'''


import torch
import math
import numpy as np

"""Returns softmax array for array of scores

Converts a set of raw scores from a model (logits) into a 
probability distribution via softmax.
    
The probability distribution will be a set of real numbers
such that each is in the range 0-1.0 and the sum is 1.0.

Assumes input is a pytorch tensor: tensor([1.0, 4.0, 2.0, 3.0])
    
Keyword arguments:
    prediction -- a pytorch tensor of any positive/negative real numbers.
    base -- the base for the exponential (default e)
"""
def softmax(scores, base=math.e):
        exps = (base**scores.to(dtype=torch.float)) # exponential for each value in array
        sum_exps = torch.sum(exps) # sum of all exponentials

        prob_dist = exps / sum_exps # normalize exponentials 
        return prob_dist


def least_confidence(logits):
    # print("logits: ", logits)

    logits_softmax = softmax(logits)
    print("logits after softmax: ", logits_softmax)

    # the index of the max logits
    max_logits_softmax_index = torch.argmin(logits_softmax, dim=1)
    print("the index: ", max_logits_softmax_index)
    print("the length of index: ", len(max_logits_softmax_index))

    max_uncertainty_id = max_logits_softmax_index
    
    return max_uncertainty_id