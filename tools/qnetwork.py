from torch import nn, optim
import torch

class QNet:
    def __init__(self, action_dim, state_dim, hidden_layers, discount, lr, wgt_decay,
                 lagged=False):

        self.dtype = torch.float32

        self.action_matrix = torch.eye(action_dim, dtype=self.dtype)

        self.discount = discount

        self.net = self.make_net(action_dim, state_dim, hidden_layers)

        self.lagged = lagged
        if lagged:
            self.lagged_net = self.make_net(action_dim, state_dim, hidden_layers)
            self.lag_state = self.net.state_dict()
            self.lagged_net.load_state_dict(self.lag_state)
        self.params = self.net.parameters()
        self.optim = optim.SGD(self.params, lr=lr, weight_decay=wgt_decay, momentum=.01)
        self.loss_function = nn.functional.smooth_l1_loss

    def make_net(self, action_dim, state_dim, hidden_layers):
        layers = []
        prev_dim = state_dim
        for next_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.LeakyReLU())
            prev_dim = next_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        return nn.Sequential(*layers)

    def transfer_lag(self):
        self.lagged_net.load_state_dict(self.lag_state)
        self.lag_state = self.net.state_dict()

    def train(self, batch):
        act_mask = torch.stack([self.action_matrix[a] for a in batch['act']])
        obs1, obs2, rew, don = [torch.tensor(batch[x], dtype=self.dtype) for x in ('obs1', 'obs2', 'rew', 'don')]
        q_direct = torch.sum(self.net(obs1) * act_mask, dim=1)
        with torch.no_grad():
            net = self.lagged_net if self.lagged else self.net
            q_discount = rew + torch.max(net(obs2), dim=1)[0] * self.discount
            q_discount = q_discount * don
        loss = self.loss_function(q_direct, q_discount)
        self.net.zero_grad()
        loss.backward()
        self.optim.step()

    def write_tb_stats(self, tb_writer, i_episode):
        avg_wght = torch.tensor([param.mean() for param in self.net.parameters()]).mean().item()
        avg_abs_wght = torch.tensor([param.abs().mean() for param in self.net.parameters()]).mean().item()
        min_abs_wght = torch.tensor([param.abs().min() for param in self.net.parameters()]).min().item()
        max_abs_wght = torch.tensor([param.abs().max() for param in self.net.parameters()]).max().item()
        tb_writer.add_scalar(f'QNet/average_weight', avg_wght, i_episode)
        tb_writer.add_scalar(f'QNet/average_absolute_weight', avg_abs_wght, i_episode)
        tb_writer.add_scalar(f'QNet/min_absolute_weight', min_abs_wght, i_episode)
        tb_writer.add_scalar(f'QNet/max_absolute_weight', max_abs_wght, i_episode)


    def __call__(self, obs):
        obs = torch.tensor(obs, dtype=self.dtype)
        with torch.no_grad():
            q = self.net(obs).numpy()
        return q