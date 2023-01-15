import pickle
import os
from utils.const import SAVE_AGENT_NAME


def save_learner(learner):
    with open(SAVE_AGENT_NAME, 'wb') as lrn_file:
        pickle.dump(learner, lrn_file)


def load_learner():
    with open(SAVE_AGENT_NAME, 'rb') as lrn_file:
        learner = pickle.load(lrn_file)
        return learner
    # try:
    #     with open(SAVE_AGENT_NAME, 'rb') as lrn_file:
    #         learner = pickle.load(lrn_file)
    #         return learner
    # except:
    #     return None


