from tkinter.tix import Y_REGION
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee

import random

#RNG seed for reproductivity
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

N_QUERIES = 10

# tensor to numpy in one batch
def tensor_to_np(egs_item):
    return np.array(egs_item)

def al_pool(egs):
    all_history, all_predictions = [], []

    for i in range(len(egs)):
        his, pre =  query_by_committee(egs.input_idses[i], egs.tags_ides[i])

        all_history.append(his)
        all_predictions.append(pre)

        # the batch that contains the lowest score
        last_score = [h[-1] for h in all_history]
        lowest_score_index = last_score.index(min(last_score))

    return all_history, all_history, lowest_score_index
    
# for one batch
def query_by_committee(X_raw, tag_raw):
    performance_history = 0
    learner_list = []

    X, tag = tensor_to_np(X_raw).reshape(-1, 1), tensor_to_np(tag_raw)
    X_length = X.shape[0]

# 80/20 split
    training_indices = random.sample(range(0, X_length-1), int(X_length*0.8))

    X_train, tag_train = X[training_indices], tag[training_indices] 
    X_pool, tag_pool = np.delete(X, training_indices, axis=0), np.delete(tag, training_indices, axis=0)

# creating learners with different learning strategy
    learner_knn = ActiveLearner(
        estimator=KNeighborsClassifier(n_neighbors=3), 
        X_training=X_train, 
        y_training=tag_train, 
    )

    learner_rf = ActiveLearner(
        estimator=RandomForestClassifier(), 
        X_training=X_train, 
        y_training=tag_train, 
    )

    learner_list.append(learner_knn)
    learner_list.append(learner_rf)

    committee = Committee(learner_list=learner_list)

    unqueried_score = committee.score(X, tag)
    predictions = committee.predict(X)
    performance_history = [unqueried_score]

    for index in range(N_QUERIES):
        query_index, query_instance = committee.query(X_pool)

        committee.teach(
            X=X_pool[query_index].reshape(1, -1), 
            y=tag_pool[query_index].reshape(1, )
        )

        performance_history.append(committee.score(X, tag))

        X_pool = np.delete(X_pool, query_index, axis = 0)
        tag_pool = np.delete(tag_pool, query_index)
        
        # print('Accuracy after query {n}: {acc:0.4f}'.format(n=index+1, acc=model_accuracy))

    # TODO: could use plots to visualise the result 
    predictions = committee.predict(X)
    # is_correct = (predictions == tag)

    return performance_history, predictions
    