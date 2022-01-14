import numpy as np
from pykalman import KalmanFilter
from pykalman.standard import _filter, _smooth, _smooth_pair, _em

class BatchKalmanFilter(KalmanFilter):
    def __init__(self, transition_matrices=None, observation_matrices=None,
            transition_covariance=None, observation_covariance=None,
            transition_offsets=None, observation_offsets=None,
            initial_state_mean=None, initial_state_covariance=None,
            random_state=None,
            em_vars=['transition_covariance', 'observation_covariance',
                     'initial_state_mean', 'initial_state_covariance'],
            n_dim_state=None, n_dim_obs=None):
        super().__init__(transition_matrices, observation_matrices,
            transition_covariance, observation_covariance,
            transition_offsets, observation_offsets,
            initial_state_mean, initial_state_covariance,
            random_state,
            em_vars,
            n_dim_state, n_dim_obs)
    
    def batch_em(self, X, n_iter=1, em_vars=None):
        """ Apply the EM algorithm in batch

        Args:
            X (np.array): [batch_size, T, obs_dim]
        """
        Z = X
        
        # initialize parameters
        (self.transition_matrices, self.transition_offsets,
         self.transition_covariance, self.observation_matrices,
         self.observation_offsets, self.observation_covariance,
         self.initial_state_mean, self.initial_state_covariance) = (
            self._initialize_parameters()
        )
         
        # Create dictionary of variables not to perform EM on
        if em_vars is None:
            em_vars = self.em_vars

        if em_vars == 'all':
            given = {}
        else:
            given = {
                'transition_matrices': self.transition_matrices,
                'observation_matrices': self.observation_matrices,
                'transition_offsets': self.transition_offsets,
                'observation_offsets': self.observation_offsets,
                'transition_covariance': self.transition_covariance,
                'observation_covariance': self.observation_covariance,
                'initial_state_mean': self.initial_state_mean,
                'initial_state_covariance': self.initial_state_covariance
            }
            em_vars = set(em_vars)
            for k in list(given.keys()):
                if k in em_vars:
                    given.pop(k)
                    
        # Actual EM iterations
        for i in range(n_iter):
            transition_matrices = []
            observation_matrices = []
            transition_offsets = []
            observation_offsets = []
            transition_covariance = []
            observation_covariance = []
            initial_state_mean = []
            initial_state_covariance = []
            for j in range(len(X)):
                (predicted_state_means, predicted_state_covariances,
                kalman_gains, filtered_state_means,
                filtered_state_covariances) = (
                    _filter(
                        self.transition_matrices, self.observation_matrices,
                        self.transition_covariance, self.observation_covariance,
                        self.transition_offsets, self.observation_offsets,
                        self.initial_state_mean, self.initial_state_covariance,
                        Z[j]
                    )
                )
                
                (smoothed_state_means, smoothed_state_covariances,
                kalman_smoothing_gains) = (
                    _smooth(
                        self.transition_matrices, filtered_state_means,
                        filtered_state_covariances, predicted_state_means,
                        predicted_state_covariances
                    )
                )
                
                sigma_pair_smooth = _smooth_pair(
                    smoothed_state_covariances,
                    kalman_smoothing_gains
                )
                
                out = _em(Z[j], self.transition_offsets, self.observation_offsets,
                        smoothed_state_means, smoothed_state_covariances,
                        sigma_pair_smooth, given=given
                    )
                
                transition_matrices.append(out[0])
                observation_matrices.append(out[1])
                transition_offsets.append(out[2])
                observation_offsets.append(out[3])
                transition_covariance.append(out[4])
                observation_covariance.append(out[5])
                initial_state_mean.append(out[6])
                initial_state_covariance.append(out[7])
                
            self.transition_matrices = np.stack(transition_matrices).mean(0)
            self.observation_matrices = np.stack(observation_matrices).mean(0)
            self.transition_offsets = np.stack(transition_offsets).mean(0)
            self.observation_offsets = np.stack(observation_offsets).mean(0)
            self.transition_covariance = np.stack(transition_covariance).mean(0)
            self.observation_covariance = np.stack(observation_covariance).mean(0)
            self.initial_state_mean = np.stack(initial_state_mean).mean(0)
            self.initial_state_covariance = np.stack(initial_state_covariance).mean(0)
            
        return self
    
    def batch_loglikelihood(self, X):
        ll = 0
        for i, x in enumerate(X):
            ll += self.loglikelihood(x)
            
        ll /= (i + 1)
        return ll
