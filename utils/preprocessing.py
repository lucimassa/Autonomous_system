from typing import List
from utils.const import IMG_STACK_COUNT


def get_state_size(initial_state_size: List):
    """
    get the stacked initial state size
    """
    state_size = initial_state_size.copy()
    state_size[-1] = state_size[-1] * IMG_STACK_COUNT
    return state_size

