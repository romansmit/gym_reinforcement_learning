import numpy as np


class StepMemory:
    """
    Memory for single (obs1, act, rew, obs2, don) time steps in a discrete action space setting
    """
    def __init__(self, mem_size, state_dim, action_dim, res_sampling=False, discrete_action=True):
        """
        Create StepMemory object
        :param mem_size: number of time steps that can me saved
        :param state_dim: dimension of the states
        :param action_dim: dimension of the action space
        :param res_sampling: whether reservoir sampling is used to save new memories
        :param discrete_action: whether the action space is discrete
        """
        self.mem_size = mem_size
        self.res_sampling = res_sampling
        self.discrete_action = discrete_action

        self.running_ind = 0
        self.mem_filled = 0

        self.reservoir_w = 1.0
        self.reservoir_next_i = 0

        self.obs1s = np.zeros((mem_size, state_dim), dtype='float')
        self.obs2s = np.zeros((mem_size, state_dim), dtype='float')
        if discrete_action:
            self.acts = np.zeros(mem_size, dtype='int')
        else:
            self.acts = np.zeros((mem_size, action_dim), dtype='float')
        self.rews = np.zeros(mem_size, dtype='float')
        self.dons = np.zeros(mem_size, dtype='bool')

    def get_batch(self, batch_size):
        """
        returns a batch of time steps randomly sampled from memory
        :param batch_size: number of time steps in the batch
        :return: dictionary object containing numpy arrays for each key (obs1, act, rew, obs2, don)
        """
        inds = np.random.choice(self.mem_filled, batch_size)
        batch = {'obs1': self.obs1s[inds],
                 'act': self.acts[inds],
                 'rew': self.rews[inds],
                 'obs2': self.obs2s[inds],
                 'don': self.dons[inds]
                 }
        return batch

    def commit(self, obs1, act, rew, obs2, don):
        """
        Commit a time step to memory
        :param obs1: numpy array representing the initial state
        :param act: integer (resp. numpy array) representing the action from the discrete (resp. continuous action space
        :param rew: float number representing the reward
        :param obs2: numpy array representing the resulting state
        :param don: boolean indicating whether the resulting state is terminal
        :return: None
        """
        if self.res_sampling:
            if self.mem_filled < self.mem_size:
                ind = self.mem_filled
                self.write(obs1, act, rew, obs2, don, ind)
                self.mem_filled += 1
                if self.mem_filled >= self.mem_size:
                    self.compute_next_reservoir_params()
            else:
                self.running_ind += 1
                if self.running_ind >= self.reservoir_next_i:
                    ind = np.random.choice(self.mem_size)
                    self.write(obs1, act, rew, obs2, don, ind)
                    self.running_ind = 0
                    self.compute_next_reservoir_params()
        else:
            ind = self.running_ind
            self.write(obs1, act, rew, obs2, don, ind)
            if self.mem_filled < self.mem_size:
                self.mem_filled += 1
            self.running_ind = (ind + 1) % self.mem_size

    def write(self, obs1, act, rew, obs2, don, ind):
        self.obs1s[ind] = obs1
        self.obs2s[ind] = obs2
        self.acts[ind] = act
        self.rews[ind] = rew
        self.dons[ind] = don

    def compute_next_reservoir_params(self):
        self.reservoir_w *= np.random.rand()**(1 / self.mem_size)
        self.reservoir_next_i = np.floor(np.log(np.random.rand())/np.log(1-self.reservoir_w)) + 1


if __name__ == '__main__':

    # test StepMemoryDiscreteAction
    state_dim = 3
    action_dim = 2
    mem_size = 10
    res_sampling = True
    discrete_action = False

    memory = StepMemory(mem_size=mem_size, state_dim=state_dim, action_dim=action_dim,
                        res_sampling=res_sampling, discrete_action=discrete_action)
    for i in range(3 * mem_size):
        obs1 = np.ones(state_dim) * i
        obs2 = np.ones(state_dim) * i * .1
        if discrete_action:
            act = np.random.choice(action_dim)
        else:
            act = np.ones(action_dim) * .1 * i
        rew = i
        don = np.random.choice(2)
        memory.commit(obs1, act, rew, obs2, don)

    print('reservoir sampling: ', res_sampling)
    print('discrete action space: ', discrete_action)
    print('rewards: ', memory.rews)

    print('random batch: \n', memory.get_batch(3))

    # test reservoir sampling average
    print('\ntest reservoir sampling:')
    avg_rews = []
    for j in range(100):
        memory = StepMemory(mem_size=mem_size, state_dim=state_dim, action_dim=action_dim,
                        res_sampling=True, discrete_action=discrete_action)
        for i in range(100):
            obs1 = np.ones(state_dim) * i
            obs2 = np.ones(state_dim) * i * .1
            if discrete_action:
                act = np.random.choice(action_dim)
            else:
                act = np.ones(action_dim) * .1 * i
            rew = i
            don = np.random.choice(2)
            memory.commit(obs1, act, rew, obs2, don)
        avg_rews.append(memory.rews.mean())
    avg_avg_rew = np.array(avg_rews).mean()
    print('average average reward: ', avg_avg_rew)
