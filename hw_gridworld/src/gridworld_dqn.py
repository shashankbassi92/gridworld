# Implements a Double DQN algorithm to solve GridWorld
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


from hw_gridworld.src.gridworld_gym_env import standard_grid
from hw_gridworld.src.gridworld_gym_env import print_values, print_policy

MEAN_REWARD_BOUND = .65

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9

REPLAY_SIZE = 1000
LEARNING_RATE = 1e-2
SYNC_TARGET_FRAMES = 100
REPLAY_START_SIZE = 1000

EPSILON_DECAY_LAST_FRAME = 1000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


# holds data for a single step of the episode
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            a_str, _ = get_a_star_and_qmax_at_s(self.env, self.env.current_state(), net, device)
            action = env.get_action_numeric(a_str)

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        new_state_one_hot = self.env.one_hot_encode_state(new_state)
        curr_state_one_hot = self.env.one_hot_encode_state(self.state)
        self.total_reward += reward

        exp = Experience(curr_state_one_hot, action, reward, is_done, new_state_one_hot)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        """Instantiate self.net as a neural network with input size equal to obs_size, 1 hidden layer of size equal to 
        hidden_size with ReLU activation function, and an output linear layer with output size equal to n_actions
        ENTER CODE HERE
        """
        self.layer1 = torch.nn.Linear(obs_size, hidden_size)
        self.activ1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_size, n_actions)
        self.net = lambda x: self.layer2(self.activ1(self.layer1(x)))

    def forward(self, x):
        return self.net(x)


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # get Q values corresponding to the actions taken from the initial states of the transitions
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1).type(torch.int64)).squeeze(-1)
    with torch.no_grad():
        # use target net to get max[a]{ Q(s',a) } at the next states s'
        next_state_values = tgt_net(next_states_v).max(1)[0]
        # manually set Q values to 0 for any transitions that ended in the terminal state
        next_state_values[done_mask] = 0.0
        # prevent NN from optimizing calculation of next state values since we only want to optimize calculation of
        # current state values
        next_state_values = next_state_values.detach()

    """Do a Bellman update and return the MSELoss between the expected and net predicted Q(s,a)
    ENTER CODE HERE
    """
    # import pdb; pdb.set_trace()
    next_state_values = rewards_v + GAMMA*next_state_values
    loss = sum((state_action_values - next_state_values)**2)
    return loss

@torch.no_grad()
def get_a_star_and_qmax_at_s(env, s, net, device):
    s_enc = np.array(env.one_hot_encode_state(s), copy=False)
    state_v = torch.tensor(s_enc).to(device)
    q_vals = net(state_v)
    a_star = env.get_action_str(q_vals.argmax().item())
    q_max = q_vals.max().item()
    return a_star, q_max


if __name__ == "__main__":
    device = torch.device("cpu")

    env = standard_grid()
    env_name = env.spec.id
    obs_size = env.observation_space.n
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions).to(device)
    tgt_net = Net(obs_size, HIDDEN_SIZE, n_actions).to(device)
    writer = SummaryWriter(comment=env_name)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    all_V0 = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_time = ts
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            _, V0 = get_a_star_and_qmax_at_s(env, env.start, net, device)
            all_V0.append(V0)
            time_from_start = time.time() - start_time
            speed = (frame_idx - ts_frame) / (time.time() - ts) if (time.time() - ts) != 0 else 0
            ts_frame = frame_idx
            ts = time.time()
            mean_V0 = np.mean(all_V0[-100:])
            print("%d: done %d games, reward %.3f, eps %.2f, speed %.2f f/s, time from start %.2f" % (
                frame_idx, len(all_V0), mean_V0, epsilon, speed, time_from_start
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("mean_V0_100_ep", mean_V0, frame_idx)
            writer.add_scalar("V0", reward, frame_idx)

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if best_m_reward is None or best_m_reward < mean_V0:
            if best_m_reward is not None:
                print("Best reward updated %.3f -> %.3f" % (best_m_reward, mean_V0))
            best_m_reward = mean_V0
        if mean_V0 > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

        writer.add_scalar("loss", loss_t, frame_idx)
    writer.close()

    # determine the policy from Q*
    # find V* from Q*
    policy = {}
    V = {}
    for s in env.actions.keys():
        a_star, q_max = get_a_star_and_qmax_at_s(env, s, net, device)
        policy[s] = a_star
        V[s] = q_max

    print("values:")
    print_values(V, env)
    print("policy:")
    print_policy(policy, env)
