import gym
import numpy as np
from time import sleep

STATE_DIM = 4
ACTION_DIM = 2

class Agent:
    def __init__(self):
        pass

    def __call__(self, obs, greedy=True):
        return np.random.choice(ACTION_DIM)

    def memorize_and_train(self, obs, rew, don):
        pass


def play_episode(env, agent, render=True, training=False):
    don = False
    t = 0
    obs = env.reset()
    while not don:
        t += 1
        if render:
            env.render()
            sleep(.04)
        act = agent(obs)
        obs, rew, don, inf = env.step(act)
        if training:
            agent.memorize_and_train(obs, rew, don)

    print("Episode finished after {} timesteps".format(t))
    if render:
        env.render()
        sleep(1)



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent()

    play_episode(env=env, agent=agent, render=True)

    env.close()
