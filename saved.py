'''
pool-based sampling -> scrnario
uncertainty-based sampling -> query strategy framework
example url: https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py
'''


import torch
import math
from random import shuffle

class UncertaintySampler:
    def __init__(self, verbose) -> None:
        self.verbose = verbose

    """
    Keyword arguments:
        prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
        sorted -- if the probability distribution is pre-sorted from largest to smallest
    """
    def least_confidence(self, prob_dist, sorted=False):
        # most confidence prediction
        if sorted:
            simple_least_conf = prob_dist.data[0]
        else:
            simple_least_conf = torch.max(prob_dist)
        
        num_labels = prob_dist.numel()

        normalized_least_conf = (1-simple_least_conf) * (num_labels / (num_labels-1))

        return normalized_least_conf.item()

    def softmax(self, scores, base=math.e):
        exps = (base**scores.to(dtype=torch.float))
        sum_exps = torch.sum(exps)

        return exps / sum_exps

    def get_samples(self, model, unlabeled_data, method, feature_method, number=5, limit=10000):
        samples = []

        if limit == -1 and len(unlabeled_data) > 10000 and self.verbose:
            print("get predictions for a large amount of unlabeled data, this might take a while")

        else:
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]

        with torch.no_grad():
            v = 0
            for item in unlabeled_data:
                text = item[1]

                feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)

                prob_dist = torch.exp(log_probs)
                score = method(prob_dist.data[0])

                item[3] = method.__name__
                item[4] = score
                
                samples.append(item)
        
        samples.sort(reverse=True, key=lambda x: x[4])

        return samples[:number:] 