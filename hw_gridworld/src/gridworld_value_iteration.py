# Implements the Value Iteration algorithm, as described in Sutton & Barto 2nd Ed, p.83.
from hw_gridworld.src.gridworld_gym_env import ACTION_SPACE, standard_grid, print_values, print_policy
import numpy as np
import pdb

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


def calc_optimal_policy(env, Vstar):
    # find the policy that leads to optimal value function
    policy = {}
    for s in env.actions.keys():
        best_a = None
        best_value = float('-inf')
        # loop through all possible actions to find the best current action
        for a in ACTION_SPACE:
            v = 0
            for s2 in env.all_states():
                # reward is a function of (s, a, s'), 0 if not specified
                r = rewards.get((s, a, s2), 0)
                v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * Vstar[s2])

            # best_a is the action associated with best_value
            if v > best_value:
                best_value = v
                best_a = a
        policy[s] = best_a
    return policy


if __name__ == '__main__':
    env = standard_grid()
    transition_probs, rewards = env.get_transition_probs_and_rewards()

    # print rewards
    print("rewards:")
    print_values(env.rewards, env)

    # initialize V(s)
    V = {}
    states = env.all_states()
    for _s_ in states:
        V[_s_] = 0

    # repeat until convergence
    iter = 0
    biggest_value_update = SMALL_ENOUGH*1.1

    transition_states = {}
    for key in transition_probs:
        transition_states[(key[0], key[1])] = key[2]
    while biggest_value_update > SMALL_ENOUGH:
        biggest_value_update = 0
        """For each state, calculate
                V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
        ENTER CODE HERE
        """
        for _s_ in states:
            maxx = 0
            for act in ACTION_SPACE:
                s_prime = transition_states.get((_s_, act), -1)
                if s_prime != -1:
                    prob = transition_probs.get((_s_, act, s_prime), 0)
                    rew = rewards.get((_s_, act, s_prime), 0)
                    maxx = max(maxx, prob*(rew + GAMMA * V[s_prime]))
            biggest_value_update = max(np.abs(maxx - V[_s_]), biggest_value_update)
            V[_s_] = maxx
        iter += 1

    # use V* to calculate the optimal policy
    policy = calc_optimal_policy(env, V)

    print("values:")
    print_values(V, env)
    print("policy:")
    print_policy(policy, env)
