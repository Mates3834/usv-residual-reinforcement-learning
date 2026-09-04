from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.asarray, zip(*batch))

        return (
            torch.tensor(s, dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.float32, device=device),
            torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(ns, dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(-1),
        )


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        h = self.body(state)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic = torch.tanh(mu)
        return action, log_prob, deterministic


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))


class ResidualSAC:
    """
    Soft Actor-Critic agent producing residual control corrections.

    The policy output is not used as the complete plant command.
    It is scaled and added to a separate nominal controller.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        residual_scale=(0.25, 0.25),
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.residual_scale = torch.tensor(
            residual_scale,
            dtype=torch.float32,
            device=self.device,
        )

        self.actor = GaussianActor(state_dim, action_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)

    def residual_action(self, state, deterministic=False):
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            sampled, _, mean = self.actor.sample(state_t)
            action = mean if deterministic else sampled
            residual = action * self.residual_scale

        return residual.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=128):
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size,
            self.device,
        )

        with torch.no_grad():
            next_raw, next_logp, _ = self.actor.sample(next_states)
            next_actions = next_raw * self.residual_scale

            q1_t = self.q1_target(next_states, next_actions)
            q2_t = self.q2_target(next_states, next_actions)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp

            target = rewards + self.gamma * (1.0 - dones) * q_t

        q1_loss = F.mse_loss(self.q1(states, actions), target)
        q2_loss = F.mse_loss(self.q2(states, actions), target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        raw_actions, logp, _ = self.actor.sample(states)
        residual_actions = raw_actions * self.residual_scale

        q_pi = torch.min(
            self.q1(states, residual_actions),
            self.q2(states, residual_actions),
        )
        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, source, target):
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)
