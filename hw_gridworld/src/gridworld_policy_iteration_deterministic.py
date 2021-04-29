# Implements the Policy Iteration algorithm, as described in Sutton & Barto 2nd Ed, p.80.
import numpy as np
from hw_gridworld.src.gridworld_gym_env import standard_grid, ACTION_SPACE, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


def evaluate_deterministic_policy(grid, policy):
    # initialize V(s) = 0
    V = {}
    for s in grid.all_states():
        V[s] = 0

    # repeat until convergence
    it = 0
    biggest_value_update = float('inf')
    while biggest_value_update > SMALL_ENOUGH:
        biggest_value_update = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0  # we will accumulate the answer
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        # action probability is deterministic
                        action_prob = 1 if policy.get(s) == a else 0

                        # reward is a function of (s, a, s'), 0 if not specified
                        r = rewards.get((s, a, s2), 0)
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

                # after done getting the new value, update the value table
                V[s] = new_v
                biggest_value_update = max(biggest_value_update, np.abs(old_v - V[s]))
        it += 1
    return V


if __name__ == '__main__':
    env = standard_grid()
    transition_probs, rewards = env.get_transition_probs_and_rewards()

    # print rewards
    print("rewards:")
    print_values(env.rewards, env)

    # state -> action
    # we'll randomly choose an action and update as we learn
    policy = {}
    for s in env.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)

    # initial policy
    print("initial policy:")
    print_policy(policy, env)

    # repeat until convergence - will break out when policy does not change
    is_policy_converged = False
    while not is_policy_converged:
        # policy evaluation step
        V = evaluate_deterministic_policy(env, policy)

        # policy improvement step
        is_policy_converged = True
        for s in env.actions.keys():
            old_a = policy[s]
            new_a = None
            best_value = float('-inf')

            # loop through all possible actions to find the best current action
            for a in ACTION_SPACE:
                v = 0
                for s2 in env.all_states():
                    # reward is a function of (s, a, s'), 0 if not specified
                    r = rewards.get((s, a, s2), 0)
                    v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

                if v > best_value:
                    best_value = v
                    new_a = a

            # new_a now represents the best action in this state
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False

    # once we're done, print the final policy and values
    print("values:")
    print_values(V, env)
    print("policy:")
    print_policy(policy, env)
