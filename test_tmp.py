import numpy as np


img_stack_count = 4



def stack_states(states):
    stacks = np.full((img_stack_count, *states.shape), states[0])
    stacks[0, :] = states
    for i in range(1, img_stack_count):
        stacks[i, i:] = states[:-i]
    return np.concatenate(stacks, axis=-1)


a = np.array([[1, 1],[2, 2],[3, 3],[4, 4],[5, 5]])

b = stack_states(a)
print(b)
