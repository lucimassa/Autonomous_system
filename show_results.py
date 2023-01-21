from utils.const import SAVE_AGENT_NAME
from networks.actor_agent import ActorAgent
from matplotlib import pyplot as plt
from utils.file_manager import restore_results, mix_results


def plot_results(results):
    plt.plot(results)
    plt.show()


if __name__ == '__main__':
    # mix_results(SAVE_AGENT_NAME+"_scores_backup.pkl", SAVE_AGENT_NAME+"_scores.pkl", SAVE_AGENT_NAME+"_scores_tmp.pkl")
    results = restore_results(name=SAVE_AGENT_NAME)
    plot_results(results)
