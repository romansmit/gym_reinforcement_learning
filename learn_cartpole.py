import gym
import numpy as np
from time import sleep
from tools.memory import StepMemory
from tqdm import tqdm
from tools.qnetwork import QNet
from tools.multiqnetwork import MultiQNet
from random import random, randrange
from math import sin


class Agent:
    def __init__(self, mem_size, state_dim, action_dim,
                 hidden_layers, training_interval, expl_time, eps_decay_time, eps_base, polyak,
                 n_q_copies):

        self.memory = StepMemory(mem_size=mem_size, state_dim=state_dim, action_dim=action_dim,
                                 res_sampling=True, discrete_action=True)
        self.training_interval = training_interval
        self.steps_learned = 0
        self.episodes_learned = 0

        self.n_q_copies = n_q_copies

        self.batches_trained = 0

        self.eps = 1.
        self.expl_time = expl_time
        self.eps_decay_time = eps_decay_time
        self.eps_base = eps_base
        self.eps_reduce = (1 - eps_base) / eps_decay_time

        self.qnet = MultiQNet(n_copies=n_q_copies, action_dim=ACTION_DIM, state_dim=STATE_DIM,
                              hidden_layers=hidden_layers, discount=.97, lr=.01, wgt_decay=.001, polyak=polyak,
                              cuda=False)

        self.action_dim = action_dim

    def __call__(self, obs, greedy):
        q_copies = self.qnet(obs)
        q = q_copies.max(axis=0)
        if not greedy:
            greed = .5 + .5 * sin(self.episodes_learned)
            q = greed * q + (1 - greed) * q_copies.std(axis=0)
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
        batch_size = 8 * self.training_interval
        batch = self.memory.get_batch(batch_size)
        self.qnet.train(batch)
        self.batches_trained += 1

    def write_tb_stats(self, tb_writer, i_episode):
        self.qnet.write_tb_stats(tb_writer, i_episode)
        tb_writer.add_scalar(f'Agent/epsilon', self.eps, i_episode)


def train_agent(agent, n_episodes, verbose=False, tb_writer=None):
    env = gym.make('CartPole-v0')
    scores = []
    greedy_scores = []
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
            tb_writer.add_scalar(f'Game/avg_score', avg_score, i_ep)
            tb_writer.add_scalar(f'Game/act0', action_ratio, i_ep)
            tb_writer.add_scalar(f'Game/act1', 1-action_ratio, i_ep)
        if i_ep + 1 % 1000 == 0:
            greedy_sc = 0.
            greedy_action_ratio = 0
            n_eval = 10
            for i in range(n_eval):
                score, actions = play_episode(env, agent, render=False, training=False, verbose=False)
                greedy_sc += score
                greedy_action_ratio += actions.mean()
            greedy_sc /= n_eval
            greedy_action_ratio /= n_eval
            greedy_scores.append(greedy_sc)
            if tb_writer:
                tb_writer.add_scalar(f'Game/greedy_score', greedy_sc, i_ep)
                tb_writer.add_scalar(f'Game/greedy_act0', greedy_action_ratio, i_ep)
                tb_writer.add_scalar(f'Game/greedy_act1', 1-greedy_action_ratio, i_ep)

    env.close()
    if verbose:
        print('Evaluation scores: ', greedy_scores)

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

    EXPLORATION_TIME = 50
    EPSILON_DECAY_TIME = 150
    EPSILON_BASE = .04
    POLYAK = .8

    N_Q_COPIES = 3

    tb_writer = SummaryWriter(log_dir='runs/cartpole/{}'.format(START_TIME))

    agent = Agent(mem_size=MEM_SIZE, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                  hidden_layers=HIDDEN_LAYERS, training_interval=TRAINING_INTERVAL,
                  expl_time=EXPLORATION_TIME, eps_decay_time=EPSILON_DECAY_TIME, eps_base=EPSILON_BASE, polyak=POLYAK,
                  n_q_copies=N_Q_COPIES)

    train_agent(agent, n_episodes=10000, verbose=True, tb_writer=tb_writer)

    env = gym.make('CartPole-v0')
    play_episode(env, agent, render=True, training=False, verbose=True)
    env.close()



