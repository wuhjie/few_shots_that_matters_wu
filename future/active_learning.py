'''
pool-based sampling -> scrnario
uncertainty-based sampling -> query strategy framework
example url: https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html
'''


from tkinter.tix import Y_REGION
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import uncertainty_sampling
import random

#RNG seed for reproductivity
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

N_QUERIES = 10

# tensor to numpy in one batch
def tensor_to_np(egs_item):
    # return np.array([e.tolist() for e in egs_item])
    return np.array(egs_item)

def al_pool(egs):
    all_history, all_predictions = [], []

    for i in range(len(egs)):
        his, pre =  al_with_pool_batched(egs.input_idses[i], egs.tags_ides[i])

        all_history.append(his)
        all_predictions.append(pre)

        # the batch that contains the lowest score
        last_score = [h[-1] for h in all_history]
        lowest_score_index = last_score.index(min(last_score))

    return all_history, all_history, lowest_score_index
    
# for one batch
def al_with_pool_batched(X_raw, tag_raw):
    performance_history = 0
    learner_list = []

    # reshape for ActiveLearner
    X, tag = tensor_to_np(X_raw).reshape(-1, 1), tensor_to_np(tag_raw)
    X_length = X.shape[0]

# 80/20 split
    training_indices = random.sample(range(0, X_length-1), int(X_length*0.8))

    X_train, tag_train = X[training_indices], tag[training_indices] 
    X_pool, tag_pool = np.delete(X, training_indices, axis=0), np.delete(tag, training_indices, axis=0)

# the corex
    knn = KNeighborsClassifier(n_neighbors=3)
    learner = ActiveLearner(
        estimator=knn, 
        X_training=X_train, 
        y_training=tag_train, 
        )

    # query_strategy=uncertainty_sampling

    learner_list.append(learner)

    committee = Committee(learner_list=learner_list)


    unqueried_score = committee.score(X, tag)
    predictions = committee.predict(X)

    performance_history = [unqueried_score]

    for index in range(N_QUERIES):
        query_index, query_instance = committee.query(X_pool)

        X_record, tag_record = X_pool[query_index].reshape(1, -1), tag_pool[query_index].reshape(1, )
        committee.teach(X=X_record, y=tag_record)

        X_pool, tag_pool = np.delete(X_pool, query_index, axis = 0), np.delete(tag_pool, query_index)
        
        model_accuracy = committee.score(X, tag)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index+1, acc=model_accuracy))

        performance_history.append(model_accuracy)

    # TODO: could use plots to visualise the result 
    predictions = committee.predict(X)
    # is_correct = (predictions == tag)

    return performance_history, predictions
    