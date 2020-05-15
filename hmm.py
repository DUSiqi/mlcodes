from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

        self.alpha = None
        self.beta = None
        self.seq_prob = None
        self.post_prob = None
        self.likelihood = None
        self.path = None

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        ini_index = self.obs_dict[Osequence[0]]
        # bso1 = self.B[:,ini_index]
        alpha[:,0] = self.pi * self.B[:,ini_index]

        for t in range(1,L):
            index_ot = self.obs_dict[Osequence[t]]
            for s in range(S):
                alpha[s][t] = self.B[s][index_ot] * np.sum(alpha[:,t-1] * self.A[:,s])
        ###################################################

        self.alpha = alpha
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        beta[:,L-1] = 1

        for t in range(L-2,-1,-1):
            otand1 = self.obs_dict[Osequence[t+1]]
            for s in range(S):
                beta[s][t] = np.sum(self.A[s] * self.B[:,otand1] * beta[:,t+1])
        ###################################################

        self.beta = beta
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        T = len(Osequence)-1
        prob = np.sum(self.alpha[:,T])
        ###################################################

        self.seq_prob = prob
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        prob = ( self.alpha * self.beta ) / self.seq_prob
        ###################################################

        self.post_prob = prob
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        for s in range(S):
            for s1 in range(S):
                for t in range(L-1):
                    otand1 = self.obs_dict[Osequence[t+1]]
                    prob[s][s1][t] = self.alpha[s][t] * self.A[s][s1] * self.B[s1][otand1] * self.beta[s1][t+1]
        prob = prob / self.seq_prob
        ###################################################

        self.likelihood = prob
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        DELTA = np.zeros([S,L])

        ini_index = self.obs_dict[Osequence[0]]
        delta[:, 0] = self.pi * self.B[:, ini_index]

        for t in range(1,L):
            ot = self.obs_dict[Osequence[t]]
            for s in range(S):
                delta[s][t] = self.B[s][ot] * max(self.A[:,s] * delta[:,t-1])
                DELTA[s][t] = np.argmax(self.A[:,s] * delta[:,t-1])

        OPT = np.zeros(L)
        OPT[L-1] = np.argmax(delta[:,L-1])

        for t in range(L-1,0,-1):
            OPT[t-1] = DELTA[int(OPT[t])][t]

        reverse_state_dict = {v:k for k,v in self.state_dict.items()}
        for i in OPT:
            path.append(reverse_state_dict[i])

        ###################################################
        self.path = path
        return path
