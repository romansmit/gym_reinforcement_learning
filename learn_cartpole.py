import gym
import numpy as np
from time import sleep
from tools.memory import StepMemory
from tqdm import tqdm
from tools.qnetwork import QNet
from random import random, randrange


class Agent:
    def __init__(self, mem_size, state_dim, action_dim,
                 hidden_layers, training_interval, expl_time, eps_decay_time, eps_base):

        self.memory = StepMemory(mem_size=mem_size, state_dim=state_dim, action_dim=action_dim,
                                 res_sampling=True, discrete_action=True)
        self.training_interval = training_interval
        self.steps_learned = 0
        self.episodes_learned = 0

        self.eps = 1.
        self.expl_time = expl_time
        self.eps_decay_time = eps_decay_time
        self.eps_base = eps_base
        self.eps_reduce = (1 - eps_base) / eps_decay_time

        self.qnet = QNet(action_dim=ACTION_DIM, state_dim=STATE_DIM, hidden_layers=hidden_layers)
        self.action_dim = action_dim

    def __call__(self, obs, greedy):
        if (not greedy) and random() < self.eps:
            return randrange(self.action_dim)
        else:
            q = self.qnet(obs)
            act = np.argmax(q).item()
            return act

    def adjust_eps(self):
        if self.episodes_learned <= self.expl_time:
            pass
        elif self.episodes_learned <= self.expl_time + self.eps_decay_time:
            self.eps -= self.eps_reduce


    def memorize(self, obs1, act, rew, obs2, don):
        self.memory.commit(obs1, act, rew, obs2, don)
        self.steps_learned += 1
        if don:
            self.episodes_learned += 1
            self.adjust_eps()
        if self.steps_learned % self.training_interval == 0:
            self.train()

    def train(self):
        batch_size = self.training_interval
        batch = self.memory.get_batch(batch_size)
        self.qnet.train(batch)

    def write_tb_stats(self, tb_writer, i_episode):
        self.qnet.write_tb_stats(tb_writer, i_episode)


def train_agent(agent, n_episodes, verbose=False, tb_writer=None):
    env = gym.make('CartPole-v0')
    scores = []
    scores_temp = []
    action_ratios = []
    for i_ep in tqdm(range(n_episodes)):
        score, actions = play_episode(env, agent, render=False, training=True, verbose=False)
        scores.append(score)
        action_ratios.append(actions.mean())
        scores_temp.append(score)
        if tb_writer and i_ep % 100 == 0:
            agent.write_tb_stats(tb_writer, i_ep)
            avg_score = np.array(scores_temp).mean()
            action_ratio = np.array(action_ratios).mean()
            scores_temp = []
            action_ratios = []
            tb_writer.add_scalar(f'game/avg_score', avg_score, i_ep)
            tb_writer.add_scalar(f'game/act0', action_ratio, i_ep)
            tb_writer.add_scalar(f'game/act1', 1-action_ratio, i_ep)
    env.close()
    if verbose:
        print('Scores: ', scores)

def play_episode(env, agent, render=True, training=False, verbose=False):
    don = False
    t = 0
    actions = []
    obs = env.reset()
    while not don:
        t += 1
        if render:
            env.render()
            sleep(.04)
        act = agent(obs, greedy=not training)
        actions.append(act)
        last_obs = obs
        obs, rew, don, inf = env.step(act)
        if training:
            # if the game terminates because T_MAX is reached, don't count it as terminal state
            term = don if t < T_MAX else False
            agent.memorize(last_obs, act, rew, obs, term)

    if verbose == True:
        print("Episode finished after {} timesteps".format(t))
    if render:
        env.render()
        sleep(1)

    return t, np.array(actions)


if __name__ == '__main__':

    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    START_TIME = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    STATE_DIM = 4
    ACTION_DIM = 2
    T_MAX = 200

    MEM_SIZE = 10000
    TRAINING_INTERVAL = 16
    HIDDEN_LAYERS = [200, 200]

    EXPLORATION_TIME = 100
    EPSILON_DECAY_TIME = 200
    EPSILON_BASE = .05

    tb_writer = SummaryWriter(log_dir='runs/cartpole/{}'.format(START_TIME))

    agent = Agent(mem_size=MEM_SIZE, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                  hidden_layers=HIDDEN_LAYERS, training_interval=TRAINING_INTERVAL,
                  expl_time=EXPLORATION_TIME, eps_decay_time=EPSILON_DECAY_TIME, eps_base=EPSILON_BASE)

    train_agent(agent, n_episodes=1000, verbose=True, tb_writer=tb_writer)

    env = gym.make('CartPole-v0')
    play_episode(env, agent, render=True, training=False, verbose=True)
    env.close()



