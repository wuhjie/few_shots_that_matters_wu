'''
pool-based sampling -> scrnario
uncertainty-based sampling -> query strategy framework
example url: https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py
'''


import torch
import math
import numpy as np

def softmax(scores, base=math.e):
        exps = (base**scores.to(dtype=torch.float)) # exponential for each value in array
        sum_exps = torch.sum(exps) # sum of all exponentials

        prob_dist = exps / sum_exps # normalize exponentials 
        return prob_dist


def least_confidence(logits):

    logits_softmax = softmax(logits)
    print("logits after softmax: ", logits_softmax)

    # the index of the max logitsm, calculate the min in every column
    max_logits_softmax_index = torch.argmin(logits_softmax, dim=1)
    print("the index: ", max_logits_softmax_index)
    print("the length of index: ", len(max_logits_softmax_index))

    return max_logits_softmax_index

def search_in_trn(index_list, trn_list):
    search_in_trn = []

    for i in index_list:
        for trn in trn_list:
            search_in_trn.append(trn[i])

    return search_in_trn