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
        prob_dist = exps / sum_exps
        return prob_dist


def least_confidence(logits):

    logits_softmax = softmax(logits)
    print("logits after softmax: ", logits_softmax)

    # the index of the max logitsm, calculate the min in every column
    max_logits_softmax_index = torch.argmin(logits_softmax, dim=1)
    print("the index: ", max_logits_softmax_index)
    print("the length of index: ", len(max_logits_softmax_index))

    return max_logits_softmax_index

def search_in_trn(index_list, loaders):

    # loaders.uides = index_list
    # loaders.input_idses = loaders.input_idses[index_list]
    # loaders.if_tgtes = loaders.if_tgtes[index_list]
    # loaders.attention_maskes = loaders.attention_maskes[index_list]
    # loaders.tags_ides = loaders.tags_ides[index_list]

    loaders.trn_egs.tagged_sents = loaders.trn_egs.egs[index_list]

    return loaders