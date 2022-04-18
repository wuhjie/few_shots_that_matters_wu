import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner

from active_learning.query_by_committee import N_QUERIES


RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)
N_QUERIES = 20

def pool_based_sampling(X_raw, y_raw):
    n_labeled_examples = X_raw.shape[0]
    training_indices = np.random.randint(low=0, high=n_labeled_examples+1, size=3)

    X_train = X_raw[training_indices]
    y_train = y_raw[training_indices]

    X_pool = np.delete(X_raw, training_indices, axis=0)
    y_pool = np.delete(y_raw, training_indices, axis=0)

    knn = KNeighborsClassifier(n_neighbors=3)
    learner = ActiveLearner(estimator=knn, X_training=X_train, y_training=y_train)

    predictions = learner.predict(X_raw)
    is_correct = (predictions == y_raw)

    unqueried_score = learner.score(X_raw, y_raw)

    performance_history = [unqueried_score]

    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(X_pool)
        
        X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)

        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        model_accuracy = learner.score(X_raw, y_raw)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

        performance_history.append(model_accuracy)