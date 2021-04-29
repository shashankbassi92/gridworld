import numpy as np
import gym
import gym.spaces
from gym.envs.registration import EnvSpec

ACTION_SPACE = ('U', 'D', 'L', 'R')


class Grid(gym.Env):  # Environment
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("GridWorld-v0")

    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.reset()
        self.action_space = gym.spaces.Discrete(len(ACTION_SPACE))
        self.observation_space = gym.spaces.Discrete(n=rows*cols)
        self.seed()

    def reset(self):
        self.i = self.start[0]
        self.j = self.start[1]
        return self.i, self.j

    def one_hot_encode_state(self, s):
        state_index = np.ravel_multi_index((s[0], s[1]), (self.rows, self.cols))
        res = np.zeros(self.observation_space.n, dtype=np.float32)
        res[state_index] = 1.0
        return res

    def set(self, rewards, actions):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return self.i, self.j

    def is_terminal(self, s):
        # returns whether s is a terminal state
        return s not in self.actions

    def get_next_state(self, s, a):
        # this answers: where would I end up if I perform action 'a' in state 's'?
        # useful for value methods where there is no succession of states over time.
        self.set_state(s)
        return self.step(a)

    def step(self, a):
        # takes action a from whatever is the current environment state.
        a_str = self.get_action_str(a)

        # if this action moves you somewhere else, then it will be in this dictionary
        if a_str in self.actions[(self.i, self.j)]:
            if a_str == 'U':
                self.i -= 1
            elif a_str == 'D':
                self.i += 1
            elif a_str == 'R':
                self.j += 1
            elif a_str == 'L':
                self.j -= 1

        reward = self.rewards.get((self.i, self.j), 0)
        done = self.game_over()
        s = (self.i, self.j)
        info = {}
        return s, reward, done, info

    @staticmethod
    def get_action_str(a):
        # converts a numeric action into a string, e.g. 0 -> 'U'
        return ACTION_SPACE[a] if isinstance(a, int) else a

    @staticmethod
    def get_action_numeric(a_str):
        # converts a direction to a numeric action, e.g. 'U' -> 0
        return ACTION_SPACE.index(a_str)

    def game_over(self):
        # returns true if game is over, else false
        # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())

    def get_transition_probs_and_rewards(self):
        ### define transition probabilities and grid ###
        # the key is (s, a, s'), the value is the probability
        # that is, transition_probs[(s, a, s')] = p(s' | s, a)
        # any key NOT present will considered to be impossible (i.e. probability 0)
        transition_probs = {}

        # to reduce the dimensionality of the dictionary, we'll use deterministic
        # rewards, r(s, a, s')
        # note: you could make it simpler by using r(s') since the reward doesn't
        # actually depend on (s, a)
        rewards = {}

        for i in range(self.rows):
            for j in range(self.cols):
                s = (i, j)
                if not self.is_terminal(s):
                    for a in ACTION_SPACE:
                        s2, _, _, _ = self.get_next_state(s, a)
                        transition_probs[(s, a, s2)] = 1
                        if s2 in self.rewards:
                            rewards[(s, a, s2)] = self.rewards[s2]

        return transition_probs, rewards

    def render(self, mode='human', close=False):
        pass


def standard_grid():
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # x means you can't go there
    # s means start position
    # number means reward at that state
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g


def negative_grid(step_cost=-0.1):
    # in this game we want to try to minimize the number of moves
    # so we will penalize every move
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    })
    return g


def print_values(V, g):
    # prints the state value function to console
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, g):
    # prints the policy function to console
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print("")
