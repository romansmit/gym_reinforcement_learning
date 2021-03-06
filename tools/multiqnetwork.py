from torch import nn, optim
import torch

class MultiQNet:
    def __init__(self, n_copies, action_dim, state_dim, hidden_layers, discount, lr, wgt_decay,
                 polyak=False, cuda=True):

        self.n_copies = n_copies

        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.dtype = torch.half
            self.cuda = True
            print('Cuda is used by biased DDQ')
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.float32
            self.cuda = False

        self.action_matrix = torch.eye(action_dim, dtype=self.dtype, device=self.device)

        self.discount = discount

        self.net = self.make_multi_net(n_copies, action_dim, state_dim, hidden_layers)

        self.polyak = polyak
        if polyak:
            self.lagged_net = self.make_multi_net(n_copies, action_dim, state_dim, hidden_layers)
            self.transfer_parameters(polyak=0)
        self.params = self.net.parameters()
        self.optim = optim.SGD(self.params, lr=lr, weight_decay=wgt_decay, momentum=.01)
        self.loss_function = nn.functional.smooth_l1_loss

    def make_single_net(self, action_dim, state_dim, hidden_layers):
        layers = []
        prev_dim = state_dim
        for next_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.LeakyReLU())
            prev_dim = next_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        return nn.Sequential(*layers)

    def transfer_parameters(self, polyak):
        state_dict = self.lagged_net.state_dict()
        update_state = self.net.state_dict()
        for key, old_param in state_dict.items():
            state_dict[key] = polyak * old_param + (1-polyak) * update_state[key]
        self.lagged_net.load_state_dict(state_dict)

    def make_multi_net(self, n_copies, action_dim, state_dim, hidden_layers):
        module_list = nn.ModuleList([self.make_single_net(action_dim, state_dim, hidden_layers)
                                     for _ in range(n_copies)])
        multi_net = MultiModule(module_list)
        if self.dtype == torch.half:
            multi_net.half()
        multi_net.to(self.device)
        return multi_net

    def train(self, batch):
        act_mask = torch.stack([self.action_matrix[a] for a in batch['act']])
        obs1, obs2, rew, don = [torch.tensor(batch[x], dtype=self.dtype, device=self.device)
                                for x in ('obs1', 'obs2', 'rew', 'don')
                                ]
        q_direct = torch.sum(self.net(obs1) * act_mask, dim=2)
        with torch.no_grad():
            net = self.lagged_net if self.polyak else self.net
            q_discount = rew + torch.max(net(obs2), dim=2)[0] * self.discount
            q_discount = q_discount * don
        loss = self.loss_function(q_direct, q_discount)
        self.net.zero_grad()
        loss.backward()
        self.optim.step()
        self.transfer_parameters(self.polyak)

    def write_tb_stats(self, tb_writer, i_episode):
        avg_wght = torch.tensor([param.mean().detach().to(device='cpu', dtype=torch.float32)
                                 for param in self.net.parameters()]).mean().item()
        avg_abs_wght = torch.tensor([param.abs().mean().detach().to(device='cpu', dtype=torch.float32)
                                     for param in self.net.parameters()]).mean().item()
        min_abs_wght = torch.tensor([param.abs().min().detach().to(device='cpu', dtype=torch.float32)
                                     for param in self.net.parameters()]).min().item()
        max_abs_wght = torch.tensor([param.abs().max().detach().to(device='cpu', dtype=torch.float32)
                                     for param in self.net.parameters()]).max().item()
        tb_writer.add_scalar(f'QNet/average_weight', avg_wght, i_episode)
        tb_writer.add_scalar(f'QNet/average_absolute_weight', avg_abs_wght, i_episode)
        tb_writer.add_scalar(f'QNet/min_absolute_weight', min_abs_wght, i_episode)
        tb_writer.add_scalar(f'QNet/max_absolute_weight', max_abs_wght, i_episode)


    def __call__(self, obs):
        obs = torch.tensor(obs, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            q = self.net(obs).cpu().numpy()
        return q


class MultiModule(nn.Module):
    def __init__(self, module_list):
        super(MultiModule, self).__init__()
        self.module_list = module_list

    def forward(self, x):
        x = torch.stack([m(x) for m in self.module_list])
        return x


