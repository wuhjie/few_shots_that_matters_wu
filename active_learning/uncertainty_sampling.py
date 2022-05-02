'''
pool-based sampling -> scrnario
uncertainty-based sampling -> query strategy framework
example url: https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py
'''


import torch
import math
import numpy as np

def softmax(scores, base=math.e):
    #  exponential for each value in array
        exps = (base**scores.to(dtype=torch.float)) 
    # sum of all exponentials
        sum_exps = torch.sum(exps) 
     # normalize exponentials 
        return exps / sum_exps


def least_confidence(logits):

    logits_softmax = softmax(logits)
    print("logits after softmax: ", logits_softmax)
    print("length of logits: ", len(logits_softmax))

    # the index of the max logitsm, calculate the min in every column
    max_logits_softmax_index = torch.argmin(logits_softmax, dim=1)
    print("the index: ", max_logits_softmax_index)
    print("the length of index: ", len(max_logits_softmax_index))

    return max_logits_softmax_index

def search_in_trn(index_list, loaders):

    print("loaders before: ", loaders.raw_dataset.trn_egs)
    print("list length: ", [len(li) for li in loaders.raw_dataset.trn_egs])

    for j in range(len(loaders.raw_dataset.trn_egs[0])):
        for i in index_list:
            loaders.raw_dataset.trn_egs = [loaders.raw_dataset.trn_egs[0][j][i]]

    print("loaders after: ", loaders.raw_dataset.trn_egs)

    return loaders