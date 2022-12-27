import numpy as np
import torch
import torch.nn as nn


def opt_cuda(t, device):
    if torch.cuda.is_available():
        cuda = "cuda:" + str(device)
        return t.cuda(cuda)
    else:
        return t


def np_to_tensor(n, device):
    return opt_cuda(torch.from_numpy(n).type(torch.float), device)


class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        n_neurons = 64
        self.fc = nn.Sequential(
            nn.Linear(state_dim, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh())
        self.mu = nn.Linear(n_neurons, action_dim)
        self.log_std = nn.Linear(n_neurons, action_dim)

    @staticmethod
    def get_prob(dist, action):
        log_prob = dist.log_prob(action).sum(1, keepdim=True)
        real_log_prob = log_prob - 2 * (np.log(2) - action - nn.Softplus()(-2 * action)).sum(1, keepdim=True)
        return real_log_prob

    def forward(self, s, mean=False, action=None):
        x = self.fc(s)
        mu = self.mu(x)
        if mean:
            return torch.tanh(mu)
        else:
            std = torch.clamp(self.log_std(x), -4, 0).exp()
            dist = torch.distributions.Normal(mu, std)
            a = dist.rsample()
            real_a = torch.tanh(a)
            if action is not None:
                action = torch.arctanh(torch.clamp(action, -0.999, 0.999))
                return self.get_prob(dist, a), self.get_prob(dist, action)
            else:
                return real_a, self.get_prob(dist, a)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        n_neurons = 64
        self.fc = nn.Sequential(
            nn.Linear(state_dim, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, 1))

    def forward(self, s):
        q = self.fc(s)
        return q


class TrajectoryBuffer:
    def __init__(self, state_dim, action_dim, size):
        self.sta1_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.sta2_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.prob_buf = np.zeros([size, 1], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        self.trct_buf = np.zeros([size, 1], dtype=np.float32)
        self.targ_buf = np.zeros([size, 1], dtype=np.float32)
        self.advt_buf = np.zeros([size, 1], dtype=np.float32)
        self.size = size
        self.ptr = 0

    def store(self, sta, next_sta, act, log_p, rew, done, truncated):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.acts_buf[self.ptr] = act
        self.prob_buf[self.ptr] = log_p
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.trct_buf[self.ptr] = truncated
        self.ptr = self.ptr + 1

    def calc_adv(self, gamma, gae_lambda, v1, v2):
        delta = gamma * v2 * (1 - self.done_buf + self.trct_buf) + self.rews_buf - v1
        advt = 0
        for i in range(self.size - 1, -1, -1):
            advt = delta[i] + gae_lambda * gamma * advt * (1 - self.done_buf[i])
            self.advt_buf[i] = advt
        self.targ_buf = self.advt_buf + v1
        self.advt_buf = (self.advt_buf - self.advt_buf.mean()) / (self.advt_buf.std() + 1e-8)

    def sample_batch(self, idxs):
        return dict(sta1=self.sta1_buf[idxs],
                    acts=self.acts_buf[idxs],
                    prob=self.prob_buf[idxs],
                    targ=self.targ_buf[idxs],
                    advt=self.advt_buf[idxs])


class PPOAgent:
    def __init__(self, state_dim, action_dim, rollout_length, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rollout_length = rollout_length
        self.device = device
        self.actor = opt_cuda(Actor(self.state_dim, self.action_dim), self.device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = opt_cuda(Critic(self.state_dim), self.device)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.buffer = TrajectoryBuffer(self.state_dim, self.action_dim, self.rollout_length)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.epsilon = 0.2
        self.entropy_weight = 0.01
        self.batch_size = 64
        self.n_epoch = 10
        self.gradient_clip = 0.5

    def act(self, state, mean=False):
        with torch.no_grad():
            state = np_to_tensor(state, self.device).reshape(1, -1)
            if mean:
                action = self.actor(state, mean=True)
                return action.cpu().squeeze().numpy()
            else:
                action, log_p = self.actor(state)
                return action.cpu().squeeze().numpy(), log_p.cpu().squeeze().numpy()

    def remember(self, state, next_state, action, log_p, reward, done, truncated):
        self.buffer.store(state.reshape(1, -1), next_state.reshape(1, -1), action, log_p, reward, done, truncated)

    def forget(self):
        self.buffer.ptr = 0

    def update_adv(self):
        s1 = np_to_tensor(self.buffer.sta1_buf, self.device)
        s2 = np_to_tensor(self.buffer.sta2_buf, self.device)
        with torch.no_grad():
            v1 = self.critic(s1).cpu().numpy().reshape(-1, 1)
            v2 = self.critic(s2).cpu().numpy().reshape(-1, 1)
        self.buffer.calc_adv(self.gamma, self.gae_lambda, v1, v2)

    def train(self):
        self.update_adv()
        for k in range(self.n_epoch):
            idxs = np.arange(self.rollout_length)
            np.random.shuffle(idxs)
            for i in range(self.rollout_length // self.batch_size):
                batch = self.buffer.sample_batch(idxs[self.batch_size * i:self.batch_size * (i + 1)])
                si = np_to_tensor(batch['sta1'], self.device)
                ai = np_to_tensor(batch['acts'], self.device)
                pi = np_to_tensor(batch['prob'], self.device)
                ti = np_to_tensor(batch['targ'], self.device)
                advi = np_to_tensor(batch['advt'], self.device)

                self.optim_actor.zero_grad()
                log_p, log_p_new = self.actor(si, action=ai)
                ratio = torch.exp(log_p_new - pi)
                clip_loss = - torch.min(ratio * advi,
                                        torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advi
                                        ).mean() + self.entropy_weight * log_p.mean()
                clip_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
                self.optim_actor.step()

                self.optim_critic.zero_grad()
                lc = ((ti - self.critic(si)) ** 2).mean()
                lc.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
                self.optim_critic.step()
        self.forget()
