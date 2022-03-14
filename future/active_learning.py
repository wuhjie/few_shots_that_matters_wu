# use the method--uncertainty sampling

from modAL.models import ActiveLearner, Committee
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#RNG seed for reproductivity
RAMDOM_STATE_SEED = 1
np.random.seed(RAMDOM_STATE_SEED)

# initialising committee members
n_members = 2
learner_list = list()

class ActiveLearner():
    # Uncertainty sampling query strategy. 
    # Selects the least sure instances for labelling.
    def __init__(self, X_training, y_training):
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            query_strategy=uncertainty_sampling,
            X_training=X_training, 
            y_training = y_training
        )
        
    def leaner_to_list(earner_list, learner):
        learner_list.append(learner)

    def committee_assembling(learner_list):
        return Committee(learner_list=learner_list)

    # the hypothese could be different due to the different form of active learning
    def prediction(committee, X_training):
        predictions = committee.prediction(X_training)