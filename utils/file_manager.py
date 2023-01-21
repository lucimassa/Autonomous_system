import pickle
import os
from utils.const import SAVE_AGENT_NAME
from networks.actor_agent import ActorAgent


def restore_results(name=SAVE_AGENT_NAME):
    results = ActorAgent.load_results(name)
    return results


def mix_results(name1, name2, name3):
    results1 = None
    results2 = None
    with open(name1, 'rb') as results_file1:
        with open(name2, 'rb') as results_file2:
            results1 = pickle.load(results_file1)
            results2 = pickle.load(results_file2)
            # results1 = ActorAgent.load_results("", complete_name=name1)
            # results2 = ActorAgent.load_results("", complete_name=name2)
            assert isinstance(results1, list)
            assert isinstance(results2, list)
            results3 = results1 + results2
            ActorAgent._save_scores(results3, name3)


def save_confs(train_step, name=SAVE_AGENT_NAME):
    with open(f"{name}_confs.pkl", 'wb') as results_file:
        pickle.dump(train_step, results_file)


def load_confs(name=SAVE_AGENT_NAME):
    with open(f"{name}_confs.pkl", 'rb') as conf_file:
        return pickle.load(conf_file)


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


