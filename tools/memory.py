import numpy as np

class StepMemory:
    """
    Memory for single (obs, act, rew, don) time steps in a discrete action space setting
    """
    def __init__(self, mem_size, state_dim, action_dim, res_sampling=False, discrete_action=True):
        self.mem_size = mem_size
        self.res_sampling = res_sampling
        self.discrete_action = discrete_action

        self.running_ind = 0
        self.mem_filled = 0

        self.reservoir_w = 1.0
        self.reservoir_next_i = 0

        self.obss = np.zeros((mem_size, state_dim), dtype='float')
        if discrete_action:
            self.acts = np.zeros(mem_size, dtype='int')
        else:
            self.acts = np.zeros((mem_size, action_dim), dtype='float')
        self.rews = np.zeros(mem_size, dtype='float')
        self.dons = np.zeros(mem_size, dtype='bool')

    def commit(self, obs, act, rew, don):
        if self.res_sampling:
            if self.mem_filled < self.mem_size:
                ind = self.mem_filled
                self.write(obs, act, rew, don, ind)
                self.mem_filled += 1
                if self.mem_filled >= self.mem_size:
                    self.compute_next_reservoir_params()
            else:
                self.running_ind += 1
                if self.running_ind >= self.reservoir_next_i:
                    ind = np.random.choice(self.mem_size)
                    self.write(obs, act, rew, don, ind)
                    self.running_ind = 0
                    self.compute_next_reservoir_params()

        else:
            ind = self.running_ind
            self.write(obs, act, rew, don, ind)
            if self.mem_filled < self.mem_size:
                self.mem_filled += 1
            self.running_ind = (ind + 1) % self.mem_size

    def write(self, obs, act, rew, don, ind):
        self.obss[ind] = obs
        self.acts[ind] = act
        self.rews[ind] = rew
        self.dons[ind] = don

    def compute_next_reservoir_params(self):
        self.reservoir_w *= np.random.rand()**(1 / self.mem_size)
        self.reservoir_next_i = np.floor(np.log(np.random.rand())/np.log(1-self.reservoir_w)) + 1





if __name__ == '__main__':

    # test StepMemoryDiscreteAction
    state_dim = 5
    action_dim = 3
    mem_size = 10
    res_sampling = True
    discrete_action = False

    memory = StepMemory(mem_size=mem_size, state_dim=state_dim, action_dim=action_dim,
                        res_sampling=res_sampling, discrete_action=discrete_action)
    for i in range(3 * mem_size):
        obs = np.ones(state_dim) * i
        if discrete_action:
            act = np.random.choice(action_dim)
        else:
            act = np.ones(action_dim) * .1 * i
        rew = i
        don = np.random.choice(2)
        memory.commit(obs, act, rew, don)

    print('reservoir sampling: ', res_sampling)
    print('discrete action space: ', discrete_action)
    print('rewards: ', memory.rews)

    # test reservoir sampling average
    print('\ntest reservoir sampling:')
    avg_rews = []
    for j in range(100):
        memory = StepMemory(mem_size=mem_size, state_dim=state_dim, action_dim=action_dim,
                        res_sampling=True, discrete_action=discrete_action)
        for i in range(100):
            obs = np.ones(state_dim) * i
            if discrete_action:
                act = np.random.choice(action_dim)
            else:
                act = np.ones(action_dim) * .1 * i
            rew = i
            don = np.random.choice(2)
            memory.commit(obs, act, rew, don)
        avg_rews.append(memory.rews.mean())
    avg_avg_rew = np.array(avg_rews).mean()
    print('average average reward: ', avg_avg_rew)
