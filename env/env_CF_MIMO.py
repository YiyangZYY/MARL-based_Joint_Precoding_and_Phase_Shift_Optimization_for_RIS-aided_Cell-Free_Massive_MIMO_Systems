import numpy as np

from .env import Env_Generate


def dB2power(dB):
    power = 10 ** (np.float64(dB) / 10)
    return power


class CF_MIMO(object):
    def __init__(self,
                 num_APs,
                 num_AP_antennas,
                 num_RIS_elements,
                 num_users,
                 area_size,
                 AWGN_var,
                 power_limit,
                 channel_est_error=False,
                 channel_noise_var=1e-11):
        self.L = num_APs
        self.M = num_AP_antennas
        self.R = 1  # the number of RISs
        self.N = num_RIS_elements
        self.K = num_users

        self.Env = Env_Generate(num_APs, num_AP_antennas, num_RIS_elements, num_users, area_size)

        self.awgn_var = AWGN_var

        self.power_limit = power_limit

        self.channel_est_error = channel_est_error
        self.channel_noise_var = channel_noise_var

        self.Env._position_generate_()
        self.Env._channel_generate_()

        self.H = self.Env.H
        self.F = self.Env.F
        self.G = self.Env.G
        self.W = self.Env.W
        self.Phi = self.Env.Phi

        self.G_l = np.zeros((self.L, self.R * self.N, self.M), dtype=complex)
        self.F_l = np.zeros((self.K, self.R * self.N, 1), dtype=complex)
        self.h_hat_l = np.zeros((self.L, self.K, self.M, 1), dtype=complex)
        self.h_hat = np.zeros((self.K, self.L * self.M, 1), dtype=complex)

        self.w_k = np.zeros((self.K, self.L * self.M), dtype=complex)

        power_size = 2 * self.K

        channel_size = 2 * (self.N * self.M * self.R + self.N * self.K * self.R + self.M * self.K)

        self.action_dim = 2 * self.M * self.K + 2 * self.R * self.N
        self.state_dim = channel_size + self.action_dim

        self.state = np.zeros((self.L, self.state_dim))

        self.done = None

        self.episode_t = None

    def _compute_h_hat_(self):
        for k in range(self.K):
            for r in range(self.R):
                self.F_l[k, r * self.N:(r + 1) * self.N, :] = self.F[r, k, :, :]
        for l in range(self.L):
            for r in range(self.R):
                self.G_l[l, r * self.N:(r + 1) * self.N, :] = self.G[l, r, :, :]
        for l in range(self.L):
            for k in range(self.K):
                tmp0 = self.H[l, k, :, :].reshape(self.M, 1)
                tmp1 = self.G_l[l, :, :].reshape(self.R * self.N, self.M)
                tmp2 = self.F_l[k, :, :].reshape(self.R * self.N, 1)
                tmp3 = tmp1.conj().T @ self.Phi @ tmp2
                self.h_hat_l[l, k, :, :] = tmp0 + tmp3

        for k in range(self.K):
            for l in range(self.L):
                self.h_hat[k, l * self.M:(l + 1) * self.M, :] = self.h_hat_l[l, k, :, :]

    def _compute_w_k_(self):
        for k in range(self.K):
            for l in range(self.L):
                self.w_k[k, l * self.M:(l + 1) * self.M] = self.W[l, k, :]

    def reset(self):
        self.episode_t = 0
        self.Env._position_generate_()
        self.Env._channel_generate_()

        for l in range(self.L):
            init_action_W = np.hstack((np.real(self.W[l].reshape(1, -1)), np.imag(self.W[l].reshape(1, -1))))

            init_action_Phi = np.hstack(
                (np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))

            # print(init_action_W.shape)
            # print(init_action_Phi.shape)
            init_action = np.hstack((init_action_W, init_action_Phi))

            W_real = init_action[:, :(self.M) * self.K]
            # print(W_real.shape)
            W_imag = init_action[:, self.M * self.K:2 * (self.M * self.K)]

            Phi_real = init_action[:, -2 * self.R * self.N:-self.R * self.N]
            Phi_imag = init_action[:, -self.R * self.N:]

            self.Phi = np.eye(self.R * self.N, dtype=complex) * (Phi_real + 1j * Phi_imag)
            self.W[l] = W_real.reshape(self.K, self.M) + 1j * W_imag.reshape(self.K, self.M)
            h_hat_l = self.h_hat_l[l]

            # power_real = np.real(np.diag(self.W[l] @ self.W[l].conjugate().T)).reshape(1, -1)
            #
            # power_limit = dB2power(self.power_limit) * np.ones((1, self.K)).reshape(1, -1)

            H_real, H_imag = np.real(self.H[l]).reshape(1, -1), np.imag(self.H[l]).reshape(1, -1)
            F_real, F_imag = np.real(self.F_l).reshape(1, -1), np.imag(self.F_l).reshape(1, -1)
            G_real, G_imag = np.real(self.G_l[l]).reshape(1, -1), np.imag(self.G_l[l]).reshape(1, -1)
            # H_hat_real,H_hat_imag = np.real(self.h_hat_l[l]).reshape(1,-1),np.imag(self.h_hat_l[l]).reshape(1,-1)
            # print(H_imag.shape)
            # print(F_imag.shape)
            # print(G_imag.shape)
            # print(power_real.shape)
            # print(power_limit.shape)
            self.state[l] = np.hstack(
                (init_action, H_real, H_imag, F_real, F_imag, G_real, G_imag))
            # self.state[l] = np.hstack(
            #     (init_action, power_real, power_limit, H_hat_real,H_hat_imag))
            # print("1")
        return self.state

   
    def _compute_reward_(self):
        pass

    def step(self, action):
        pass


    def close(self):
        pass
