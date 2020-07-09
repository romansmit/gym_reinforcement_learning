from torch import nn, optim
import torch

class QNet:
    def __init__(self, action_dim, state_dim, hidden_layers, discount=.99, lr=.01):

        self.dtype = torch.float32

        self.action_matrix = torch.eye(action_dim, dtype=self.dtype)

        self.discount = discount

        layers = []
        prev_dim = state_dim
        for next_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.LeakyReLU())
            prev_dim = next_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.net = nn.Sequential(*layers)
        self.optim = optim.SGD(self.net.parameters(), lr=lr, weight_decay=.01, momentum=.0)
        self.loss_function = nn.functional.smooth_l1_loss

    def train(self, batch):
        act_mask = torch.stack([self.action_matrix[a] for a in batch['act']])
        obs1, obs2, rew, don = [torch.tensor(batch[x], dtype=self.dtype) for x in ('obs1', 'obs2', 'rew', 'don')]
        q_direct = torch.sum(self.net(obs1) * act_mask, dim=1)
        with torch.no_grad():
            q_discount = rew + torch.max(self.net(obs2), dim=1)[0] * self.discount
            q_discount = q_discount * don
        loss = self.loss_function(q_direct, q_discount)
        self.net.zero_grad()
        loss.backward()
        self.optim.step()




    def __call__(self, obs):
        obs = torch.tensor(obs, dtype=self.dtype)
        with torch.no_grad():
            q = self.net(obs)
        return q.numpy()