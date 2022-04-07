'''
pool-based sampling -> scrnario
uncertainty-based sampling -> query strategy framework
example url: https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html
'''


from contextlib import redirect_stderr
from tkinter.tix import Y_REGION
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
from torch.utils.data import SequentialSampler, DataLoader

#RNG seed for reproductivity
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

N_QUERIES = 10

def extract(lst, n):
    return list(list(zip(*lst[0]))[n])

def al_with_pool(trn_data):
    X_raw, tag_raw = extract(trn_data, 0), extract(trn_data, 1)
    X, tag = np.array(X_raw), np.array(tag_raw)
    X_length = X.size

# 80/20 split
    training_indices = np.random.randint(low=0, high=X_length+1, size=int(X_length*0.8))
    X_train, tag_train = X_raw[training_indices], tag[training_indices]
    X_pool, tag_pool = np.delete(X_raw, training_indices, axis=0), np.delete(tag_raw, training_indices, axis=0)

# the core
    knn = KNeighborsClassifier(n_neighbors=3)
    learner = ActiveLearner(
        estimator=knn, 
        X_training=X_train, 
        y_training=tag_train, 
        query_strategy=uncertainty_sampling
        )

    predictions = learner.predict(X_raw)
    is_correct = (predictions==tag_raw)

    unqueried_score = learner.score(X_raw, tag_raw)

    performance_history = [unqueried_score]

    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(X_pool)

        X_record, tag_record = X_pool[query_index].reshape(1, -1), tag_pool[query_index].reshape(1, )
        learner.teach(X=X_record, y=tag_record)

        X_pool, tag_pool = np.delete(X_pool, query_index, axis = 0), np.delete(tag_pool, query_index)
        
        model_accuracy = learner.score(X_raw, tag_raw)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index+1, acc=model_accuracy))

        performance_history.append(model_accuracy)

    # TODO: could use plots to visualise the result 
    predictions = learner.predict(X_raw)
    is_correct = (predictions == tag_raw)

    return performance_history, predictions
    