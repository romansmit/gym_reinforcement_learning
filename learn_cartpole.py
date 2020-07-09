import gym
import numpy as np
from time import sleep
from tools.memory import StepMemory
from tqdm import tqdm
from tools.qnetwork import QNet

STATE_DIM = 4
ACTION_DIM = 2
T_MAX = 200

MEM_SIZE = 10000
TRAINING_INTERVAL = 16

class Agent:
    def __init__(self, mem_size=MEM_SIZE, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 training_interval=TRAINING_INTERVAL):

        self.memory = StepMemory(mem_size=mem_size, state_dim=state_dim, action_dim=action_dim,
                                 res_sampling=True, discrete_action=True)
        self.training_interval = training_interval
        self.steps_learned = 0

        self.qnet = QNet(action_dim=ACTION_DIM, state_dim=STATE_DIM, hidden_layers=[200, 200])

    def __call__(self, obs, greedy=True):
        q = self.qnet(obs)
        act = np.argmax(q).item()
        return act

    def memorize(self, obs1, act, rew, obs2, don):
        self.memory.commit(obs1, act, rew, obs2, don)
        self.steps_learned += 1
        if self.steps_learned % self.training_interval == 0:
            self.train()

    def train(self):
        batch_size = self.training_interval
        batch = self.memory.get_batch(batch_size)
        self.qnet.train(batch)


def train_agent(agent, n_episodes, verbose=False):
    env = gym.make('CartPole-v0')
    scores = []
    for i_ep in tqdm(range(n_episodes)):
        score = play_episode(env, agent, render=False, training=True, verbose=False)
        scores.append(score)
    env.close()
    if verbose:
        print('Scores: ', scores)

def play_episode(env, agent, render=True, training=False, verbose=False):
    don = False
    t = 0
    obs = env.reset()
    while not don:
        t += 1
        if render:
            env.render()
            sleep(.04)
        act = agent(obs)
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

    return t


if __name__ == '__main__':
    agent = Agent()
    train_agent(agent, n_episodes=1000, verbose=True)


