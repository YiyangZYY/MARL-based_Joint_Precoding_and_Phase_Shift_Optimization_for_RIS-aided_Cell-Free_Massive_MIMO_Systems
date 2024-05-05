import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, L, M, R, N, K, power_limit, device, max_action=1):
        super(Actor, self).__init__()
        hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()

        # # print(state_dim)
        # # print(hidden_dim)
        self.device = device

        self.L = L
        self.M = M
        self.R = 1
        self.N = N
        self.K = K
        self.power_limit = power_limit

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        # print(action_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.max_action = max_action

    def compute_power(self, actor):
        # Normalize the power
        # current_power = torch.Tensor(self.L, 1).to(self.device)

        W_real = actor[:, :(self.M) * self.K].cpu().data.numpy()
        # # print(actor.shape)
        # # print(W_real.shape)
        # # print(W_real.shape[0])
        W_imag = actor[:, self.M * self.K:2 * (self.M * self.K)].cpu().data.numpy()
        # print(W_imag.shape)
        W = W_real.reshape(W_real.shape[0], self.K, self.M) + 1j * W_imag.reshape(W_imag.shape[0], self.K, (self.M))

        WW_H = W[0] @ W[0].conjugate().T
        if np.real(np.trace(WW_H)) > self.power_limit:
            W = W * (np.real(np.trace(WW_H)) / self.power_limit)
            W_real = W * np.sqrt(2)
            W_imag = W * np.sqrt(2)
        WW_H = W[0] @ W[0].conjugate().T
        # print(WW_H==0)
        # # print(torch.from_numpy(np.array(np.real(np.trace(WW_H)))).reshape(-1, 1))
        current_power = torch.from_numpy(np.array(np.real(np.trace(WW_H)))).reshape(-1, 1).to(
            self.device)
        W_real = torch.from_numpy(np.array(W_real)).reshape(-1, 1).to(
            self.device)
        W_imag = torch.from_numpy(np.array(W_imag)).reshape(-1, 1).to(
            self.device)
        # # print(current_power_t.shape)
        # # print(current_power_t)
        # # print(current_power.shape)
        # # print(current_power)
        return current_power, W_real, W_imag

    def compute_phase(self, actor):
        # Normalize the phase matrix
        Phi_real = actor[:, -2 * self.R * self.N:-self.R * self.N].detach()
        Phi_imag = actor[:, -self.R * self.N:].detach()
        # print(Phi_imag.shape)
        return torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), torch.sum(torch.abs(Phi_imag),
                                                                                            dim=1).reshape(-1,
                                                                                                           1) * np.sqrt(
            2)

    def forward(self, state):
        # # print(state.shape)
        # # print(self.l1.weight.shape)
        # print(state.shape)
        actor = torch.tanh(self.l1(state.float()))
        # print(actor.shape)
        # Apply batch normalization to the each hidden layer's input
        actor = self.bn1(actor)
        actor = torch.tanh(self.l2(actor))
        # print(actor.shape)
        actor = self.bn2(actor)
        actor = torch.tanh(self.l3(actor))
        # print(actor.shape)
        # print(actor.detach().shape)
        # Normalize the transmission power and phase matrix
        # # print(self.power_limit)

        # current_power_real = torch.Tensor(self.L, 2 * self.M * self.K).to(self.device)
        # # print(current_power_real.shape)
        # # print(self.power_limit)

        # # print((self.compute_power(actor.detach())).expand(-1, 2 * (self.M) * self.K) / self.power_limit)
        current_power_real, W_real, W_imag = self.compute_power(actor.detach())
        # current_power_real = self.compute_power(actor.detach()).expand(-1, 2 * (self.L*self.M) **2) / np.sqrt(self.power_limit)
        # # print(current_power_real.shape)
        W_real = W_real.expand(-1, (self.M * self.K))
        W_imag = W_imag.expand(-1, (self.M * self.K))
        Phi_real_normal, Phi_imag_normal = self.compute_phase(actor.detach())

        Phi_real_normal = Phi_real_normal.expand(-1, self.R * self.N)
        Phi_imag_normal = Phi_imag_normal.expand(-1, self.R * self.N)
        # # print(current_power_real.shape)
        # # print(current_power_real[l].shape)
        # # print(Phi_real_normal.shape)
        # # print(Phi_imag_normal.shape)
        # # print([current_power_real[l], Phi_real_normal, Phi_imag_normal])
        division_term = torch.cat([W_real[0], W_imag[0], Phi_real_normal[0], Phi_imag_normal[0]], dim=0)
        # # print(division_term.shape)

        #         # print(division_term.shape)s

        #         # print((self.max_action * actor).shape)
        # print(actor.shape)
        print(W_imag.shape)
        print(W_imag[0].shape)
        print(Phi_real_normal[0].shape)
        print(division_term.shape)
        print(actor.shape)

        print((actor / division_term).shape)
        return self.max_action * actor / division_term


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden_dim = 1 if (state_dim + action_dim) == 0 else 2 ** ((state_dim + action_dim) - 1).bit_length()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, state, action):
        q = torch.tanh(self.l1(state.float()))

        q = self.bn1(q)
        # # print(type(q))
        # # print(type(action))
        q = torch.tanh(self.l2(torch.cat([q, action], 1)))

        q = self.l3(q)

        return q


class MADDPG(object):
    def __init__(self, state_dim, action_dim, L, M, R, N, K, power_limit, max_action, actor_lr, critic_lr, actor_decay,
                 critic_decay, device, discount=0.99, tau=0.001):
        self.device = device
        self.L = L
        self.M = M
        self.K = K
        self.N = N
        power_limit_W = 10 ** (power_limit / 10)
        self.actor_l = Actor(state_dim, action_dim, L, M, R, N, K, power_limit_W, max_action=max_action,
                             device=device).to(
            self.device)
        # # print(self.actor_l)
        # # print(state_dim)
        self.actor = [self.actor_l for _ in range(L)]
        # self.critic_l = Critic(state_dim , action_dim ).to(self.device)
        self.critic_l = Critic(state_dim * L, action_dim * L).to(self.device)
        self.critic = [self.critic_l for _ in range(L)]
        self.actor_tl = Actor(state_dim, action_dim, L, M, R, N, K, power_limit_W, max_action=max_action,
                              device=device).to(self.device)
        self.actor_target = [self.actor_tl for _ in range(L)]
        # self.critic_tl = Critic(state_dim , action_dim ).to(self.device)
        self.critic_tl = Critic(state_dim * L, action_dim * L).to(self.device)
        self.critic_target = [self.critic_tl for _ in range(L)]

        self.actor_optimizer_l = torch.optim.Adam(self.actor_l.parameters(), lr=actor_lr, weight_decay=actor_decay)
        self.actor_optimizer = [self.actor_optimizer_l for _ in range(L)]

        self.critic_optimizer_l = torch.optim.Adam(self.critic_l.parameters(), lr=critic_lr, weight_decay=critic_decay)
        self.critic_optimizer = [self.critic_optimizer_l for _ in range(L)]
        # self.actor = torch.zeros(L)
        # self.critic = torch.zeros(L)
        # self.actor_target = torch.zeros(L)
        # self.critic_target = torch.zeros(L)
        # self.actor_optimizer = torch.zeros(L)
        # self.critic_optimizer = torch.zeros(L)
        # Initialize actor networks and optimizer
        for l in range(L):
            self.actor[l] = Actor(state_dim, action_dim, L, M, R, N, K, power_limit_W, max_action=max_action,
                                  device=device).to(
                self.device)
            self.actor_target[l] = copy.deepcopy(self.actor[l])
            self.actor_optimizer[l] = torch.optim.Adam(self.actor[l].parameters(), lr=actor_lr,
                                                       weight_decay=actor_decay)

            # Initialize critic networks and optimizer
            self.critic[l] = Critic(state_dim * L, action_dim * L).to(self.device)
            self.critic_target[l] = copy.deepcopy(self.critic[l])
            self.critic_optimizer[l] = torch.optim.Adam(self.critic[l].parameters(), lr=critic_lr,
                                                        weight_decay=critic_decay)

        # Initialize the discount and target update rated
        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        # # # print the shape of layer1
        # # print(self.actor.l1.weight.shape)
        action = np.zeros((self.L, 2 * self.M * self.K + 2 * self.N))
        for l in range(self.L):
            self.actor[l].eval()
            state = torch.tensor(state.reshape(self.L, -1)).to(self.device)
            # action 赋值
            action[l] = self.actor[l](state[l].reshape(1, -1)).cpu().data.numpy().flatten().reshape(1, -1)

        return action

    def update_parameters(self, replay_buffer, batch_size=16):
        for l in range(self.L):
            self.actor[l].train()

            # Sample from the experience replay buffer
            state, action, next_state, reward = replay_buffer.sample(batch_size)
            # # print(next_state.shape)
            # # print(((next_state[:,l,:].reshape(batch_size, -1).reshape(batch_size, 1, -1))).shape)
            # next_action_old = self.actor_target[l](next_state.reshape(self.L * batch_size, -1)).reshape(batch_size,
            #                                                                                             self.L, -1)
            # # print(self.actor_target[l]((next_state[:,l,:].reshape(batch_size, -1).reshape(batch_size, 1, -1))))
            #
            # # Compute the target Q-value
            next_action_old = np.zeros((batch_size, self.L, 2 * self.M * self.K + 2 * self.N),dtype=complex)
            for l in range(self.L):
                next_action_old[:, l, :] = self.actor_target[l](
                    next_state[:, l, :].reshape(batch_size, -1)).detach().cpu().numpy()
            next_action = torch.tensor(next_action_old).to(self.device)
            for l in range(self.L):
                next_action[:, l, :] = self.actor_target[l](next_state[:, l, :])

            # print(next_action.shape)
            target_Q = self.critic_target[l](next_state.reshape(batch_size, -1),
                                             next_action.to(torch.float).reshape(batch_size, -1))

            target_Q = reward[:, l] + (self.discount * target_Q).detach()

            # Get the current Q-value estimate
            current_Q = self.critic[l](state.reshape(batch_size, -1), action.reshape(batch_size, -1))
            # Compute the critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer[l].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[l].step()

            # Compute the actor loss
            new_action_old = self.actor[l](state.reshape(self.L * batch_size, -1)).reshape(batch_size, self.L, -1)
            new_action = new_action_old
            for l in range(self.L):
                new_action[:, l, :] = self.actor[l](state[:, l, :])
            # print(new_action.shape)
            actor_loss = -self.critic[l](state.reshape(batch_size, -1),
                                         new_action.to(torch.float).reshape(batch_size, -1)).mean()

            # Optimize the actor
            self.actor_optimizer[l].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[l].step()

            # Soft update the target networks
            for param, target_param in zip(self.critic[l].parameters(), self.critic_target[l].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor[l].parameters(), self.actor_target[l].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Save the model parameters
    # def save(self, file_name):
    #     torch.save(self.critic.state_dict(), file_name + "_critic")
    #     torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")
    #
    #     torch.save(self.actor.state_dict(), file_name + "_actor")
    #     torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")
    #
    # # Load the model parameters
    # def load(self, file_name):
    #     self.critic.load_state_dict(torch.load(file_name + "_critic"))
    #     self.critic_optimizer.load_state_dict(torch.load(file_name + "_critic_optimizer"))
    #     self.critic_target = copy.deepcopy(self.critic)
    #
    #     self.actor.load_state_dict(torch.load(file_name + "_actor"))
    #     self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer"))
    #     self.actor_target = copy.deepcopy(self.actor)
