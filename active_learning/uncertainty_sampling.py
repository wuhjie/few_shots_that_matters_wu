'''
pool-based sampling -> scrnario
uncertainty-based sampling -> query strategy framework
example url: https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py
'''


import torch
import math
import numpy as np

def softmax(scores, base=math.e):
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
        exps = (base**scores.to(dtype=torch.float)) # exponential for each value in array
        sum_exps = torch.sum(exps) # sum of all exponentials

        prob_dist = exps / sum_exps # normalize exponentials 
        return prob_dist

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def least_confidence(logits):
    print("logits: ", logits)

    logits_softmax = stable_softmax(logits)
    print("logits_softmax: ", logits_softmax)
    print("length of softmax: ", len(logits_softmax))

    max_logits_softmax = [max(lo) for lo in logits_softmax]
    print("min_logits_softmax: ", max_logits_softmax)
    
    max_uncertainty = 1-max_logits_softmax
    print("max_uncertainty: ", max_uncertainty)
    # min_logits_softmax_index = [lo.index(min(lo)) for lo in logits_softmax]
    # print("the index: ", min_logits_softmax_index)
 
    # div = min_sentence_logits_index // batch_size
    # mod = min_sentence_logits_index % batch_size
    
    # print("the list is in the ", div, "th batch", "with position ", mod)


