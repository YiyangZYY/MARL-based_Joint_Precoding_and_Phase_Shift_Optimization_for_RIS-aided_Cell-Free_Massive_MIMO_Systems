import numpy as np
import torch


class ExperienceReplayBuffer(object):
    def __init__(self, L, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.L = L

        self.state = np.zeros((max_size, L, state_dim))
        self.action = np.zeros((max_size, L, action_dim))
        self.next_state = np.zeros((max_size, L, state_dim))
        self.reward = np.zeros((max_size, L, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward):
        # print(state.shape)
        # print(self.state.shape)
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward


        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # def add(self,l, state, action, next_state, reward, done):
    #     self.state[self.ptr] = state[l]
    #     self.action[self.ptr] = action[l]
    #     self.next_state[self.ptr] = next_state[l]
    #     self.reward[self.ptr] = reward[l]
    #     self.not_done[self.ptr] = 1.0 - done[l]
    #
    #     self.ptr = (self.ptr + 1) % self.max_size
    #     self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)
        # print(self.state.shape)
        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),

        )

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def updatemembership(obs_agent, obs_fuzzy):
    sigma = 1 / len(obs_fuzzy)
    membership = np.zeros((len(obs_agent), len(obs_fuzzy)))
    for i in range(len(obs_agent)):
        for j in range(len(obs_fuzzy)):
            temp = 1
            for k in range(len(obs_fuzzy[0])):
                # temp *= stats.norm.pdf(obs_adv[i][k],obs_fuzzy[j][k],sigma)
                # 1/action_space*num_fuzzy
                temp = temp * np.exp(-sigma * (abs(obs_agent[i][k] - obs_fuzzy[j][k])))
            membership[i, j] = temp
    return membership


def getagentaction(fuzzy_action, membership):
    temp = [[] for _ in range(int(membership.shape[0]))]
    for i in range(int(membership.shape[0])):
        # print(int(membership.shape[0]))
        for j in range(len(fuzzy_action)):
            # print(len(fuzzy_action))
            # print(membership.shape)
            # print(membership[i, j])
            # print(fuzzy_action[j])
            temp[i].append(membership[i, j] * fuzzy_action[j])
    # 解模糊
    agent_action = []
    all_action = fuzzy_action[0]
    for i in range(1, len(fuzzy_action)):
        all_action += fuzzy_action[i]
    for i in range(int(membership.shape[0])):
        action = temp[i][0]
        temp_u = membership[i][0]
        for j in range(1, len(fuzzy_action)):
            action += temp[i][j]
            temp_u += membership[i][j]
        if temp_u <= 1e-1:
            temp_u = 1
        agent_action.append(action / temp_u)
    return agent_action


def getfuzzyreward(rew_n, membership):
    temp = [0 for _ in range(int(membership.shape[1]))]
    for i in range(int(membership.shape[1])):
        membership_softmax = softmax(membership.T[i])
        for j in range(int(membership.shape[0])):
            temp[i] += (membership_softmax[j] * rew_n[j])
    return temp


def getfuzzyobs(new_obs_n, membership):
    new_obs_n = np.array(new_obs_n)
    fuzzy_obs = [None for _ in range(int(membership.shape[1]))]
    for i in range(len(fuzzy_obs)):
        temp = softmax(membership.T[i])
        fuzzy_obs[i] = temp[0] * new_obs_n[0]
        for j in range(1, membership.shape[0]):
            fuzzy_obs[i] += (temp[j] * new_obs_n[j])
    return fuzzy_obs
