import numpy as np


class Env_Generate(object):
    def __init__(self,
                 num_APs,
                 num_AP_antennas,
                 num_RIS_elements,
                 num_users,
                 area_size):
        self.L = num_APs
        self.M = num_AP_antennas
        self.R = 1  # the number of RISs
        self.N = num_RIS_elements
        self.K = num_users

        self.area_size = area_size
        self.AP_position = None
        self.RIS_position = None
        self.user_position = None
        self.AP_height = 15
        self.RIS_height = 5
        self.user_height = 2
        self.disAP2RIS = None
        self.disAP2user = None
        self.disRIS2user = None

        self.H = np.zeros((self.L, self.K, self.M, 1), dtype=complex)
        self.F = np.zeros((self.R, self.K, self.N, 1), dtype=complex)
        self.G = np.zeros((self.L, self.R, self.N, self.M), dtype=complex)
        # the dim of W is L,M,K
        self.W = np.zeros((self.L, self.K, self.M), dtype=complex)+ 1j * np.zeros((self.L, self.K, self.M), dtype=complex)
        # print(self.W.shape)
        # the dim of phi is R*N,R*N
        self.Phi = np.eye(self.R * self.N, dtype=complex) + 1j*np.eye(self.R * self.N, dtype=complex)

    def _position_generate_(self):
        # RIS is at the center of the area, the area is a square, with the length of area_size
        self.RIS_position = np.array([self.area_size / 2, self.area_size / 2])
        # AP seperate the area into L same parts, and the num of lines and rows should be m and n, where m is the
        # nearest int of sqrt(L) and n = L/m
        m = int(np.sqrt(self.L))
        n = self.L // m
        # APs are the centers of the L parts
        self.AP_position = np.zeros((self.L, 2))
        for i in range(m):
            for j in range(n):
                self.AP_position[i * n + j, :] = np.array(
                    [(i + 0.5) * self.area_size / m, (j + 0.5) * self.area_size / n])

        # users are randomly distributed in the area
        # self.user_position = np.random.uniform(0, self.area_size, (self.K, 2))
        self.user_position = np.array([[25.5, 59.5], [70.5, 20.5], [40, 30.5], [80.5, 87],[10, 11],[95, 65]])

        self.disAP2user = np.zeros((self.L, self.K), dtype=complex)
        self.disRIS2user = np.zeros((self.R, self.K), dtype=complex)
        self.disAP2RIS = np.zeros((self.L, self.R), dtype=complex)

        for i in range(self.L):
            for j in range(self.K):
                self.disAP2user[i, j] = np.sqrt(np.sum((self.AP_position[i, :] - self.user_position[j, :]) ** 2) + (
                        self.AP_height - self.user_height) ** 2)

        for r in range(self.R):
            for k in range(self.K):
                self.disRIS2user[r, k] = np.sqrt(
                    np.sum((self.RIS_position[r] - self.user_position[k, :]) ** 2) + (
                                self.RIS_height - self.user_height) ** 2)

        for i in range(self.L):
            self.disAP2RIS[i] = np.sqrt(
                np.sum((self.RIS_position - self.AP_position[i, :]) ** 2) + (self.RIS_height - self.AP_height) ** 2)

    def _channel_H_generate(self, dis):
        tmp_H = np.random.rayleigh(1, (self.M, 1)) + 1j * np.zeros((self.M, 1), dtype=complex)
        for m in range(self.M):
            # print(tmp_H[m,0])
            tmp_H[m, 0] = tmp_H[m, 0] * np.exp(1j * 2 * np.pi * np.random.rand(), dtype=complex)
            # print(np.exp(np.pi*1j,dtype = complex))
            # print(tmp_H[m,0])
        disAP2UE = np.sqrt(1e-3 * (dis ** (-3.5)))
        channel_H = disAP2UE * tmp_H
        return channel_H

    def _channel_F_generate(self, dis):
        tmp_F = np.random.rayleigh(1, (self.N, 1)) + 1j * np.zeros((self.N, 1), dtype=complex)
        for n in range(self.N):
            tmp_F[n, 0] = tmp_F[n, 0] * np.exp(1j * 2 * np.pi * np.random.rand(), dtype=complex)

        disRIS2UE = np.sqrt(1e-3 * (dis ** (-2.8)))
        channel_F = disRIS2UE * tmp_F
        return channel_F

    def _channel_G_generate(self, dis):
        tmp_G = np.ones((self.N, self.M), dtype=complex) + 1j * np.zeros((self.N, 1), dtype=complex)
        disAP2RIS = np.sqrt(2 * 1e-3 * (dis ** (-2.2)))
        channel_G = disAP2RIS * tmp_G
        return channel_G

    def _channel_generate_(self):
        # the dimension of H is (L*M, K),obey rayleigh distribution,use np.random.rayleigh
        # beta_H = np.zeros((self.L, self.K))

        # print(beta_H.shape)

        for l in range(self.L):
            for k in range(self.K):
                self.H[l, k, :, :] = self._channel_H_generate(self.disAP2user[l, k])

        for r in range(self.R):
            for k in range(self.K):
                self.F[r, k, :, :] = self._channel_F_generate(self.disRIS2user[r, k])

        # self.F = beta_F * np.random.rayleigh(1, (self.N, self.K)) * np.exp(1j * 2 * np.pi * np.random.rand())
        # print(self.F.shape)
        # the dimension of G is (N,L*M)
        for l in range(self.L):
            for r in range(self.R):
                self.G[l, r, :, :] = self._channel_G_generate(self.disAP2RIS[l, r])
        # self.G = beta_G * np.ones((self.N, self.M), dtype=complex)


