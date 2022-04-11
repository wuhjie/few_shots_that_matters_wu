'''
pool-based sampling -> scrnario
uncertainty-based sampling -> query strategy framework
example url: https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html
'''


from tkinter.tix import Y_REGION
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import random

#RNG seed for reproductivity
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

N_QUERIES = 10

def extract(lst, n):
    return np.array(list(list(zip(*lst[0]))[n]))

def indices_data_mapping(egs, indices):
    mapped_egs = []
    for i in indices:
        mapped_egs.append(egs[i])
    return mapped_egs

def tensor_to_np(egs_item):
    return np.array([e.tolist() for e in egs_item])

def al_with_pool(egs):
    
    # length of each segmentation is 128
    X, tag = egs.input_idses[0], egs.tags_ides[0] 

    performance_history = 0
    # X, tag = tensor_to_np(egs.input_idses), tensor_to_np(egs.tags_ides)

    # print("X: ", X)
    print("tags: ", tag)

    X_length = X.shape[0]

# 80/20 split
    training_indices = random.sample(range(0, X_length-1), int(X_length*0.8))

    # print("training indices: ", training_indices)

    X_train, tag_train = X[training_indices], tag[training_indices] 
    X_pool, tag_pool = np.delete(X, training_indices, axis=0), np.delete(tag, training_indices, axis=0)

# the corex
    knn = KNeighborsClassifier(n_neighbors=3)
    learner = ActiveLearner(
        estimator=knn, 
        X_training=X_train, 
        y_training=tag_train, 
        query_strategy=uncertainty_sampling
        )

    predictions = learner.predict(X)
    is_correct = (predictions==tag)

    unqueried_score = learner.score(X, tag)

    performance_history = [unqueried_score]

    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(X_pool)

        X_record, tag_record = X_pool[query_index].reshape(1, -1), tag_pool[query_index].reshape(1, )
        learner.teach(X=X_record, y=tag_record)

        X_pool, tag_pool = np.delete(X_pool, query_index, axis = 0), np.delete(tag_pool, query_index)
        
        model_accuracy = learner.score(X, tag)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index+1, acc=model_accuracy))

        # performance_history.append(model_accuracy)

    # TODO: could use plots to visualise the result 
    predictions = learner.predict(X)
    is_correct = (predictions == tag)

    return performance_history, predictions
    