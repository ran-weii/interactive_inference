import torch
import torch.nn as nn
from src.distributions.mixture_models import ConditionalGaussian
from src.distributions.transition_models import DiscreteMC, LogisticMC
from src.distributions.utils import softmax

class DiscreteGaussianHMM(nn.Module):
    """Input-output hidden markov model with:
        discrete transitions, discrete actions, gaussian observations
    """
    def __init__(self, state_dim, act_dim, obs_dim, rank=0, cov="full"):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): discrete action dimension
            obs_dim (int): observation dimension
            cov (str): observation covariance type. chioces=["full", "diag], default="full"
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.eps = 1e-6

        self.obs_model = ConditionalGaussian(
            obs_dim, state_dim, cov=cov, batch_norm=True
        )
        self.transition_model = DiscreteMC(state_dim, act_dim, rank)
    
    def get_initial_state(self):
        return self.transition_model.get_initial_state()

    def get_transition_matrix(self, a):
        """
        Args:
            a (torch.tensor): action vector. size=[batch_size, act_dim]

        Returns:
            transition_matrix (torch.tensor): action conditioned transition matrix.
                size=[batch_size, state_dim, state_dim]
        """
        a_ = a.unsqueeze(-1).unsqueeze(-1)
        transition_matrix = self.transition_model.get_transition_matrix()
        transition_matrix = torch.sum(transition_matrix * a_, dim=-3)
        return transition_matrix
    
    def alpha(self, b, x, a=None):
        """ Compute forward message for a single time step

        Args:
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            x (torch.tensor): observation vector. size=[batch_size, obs_dim]
            a (torch.tensor, None): action vector. size=[batch_size, act_dim]. 
                None for initial state

        Returns:
            b_t (torch.tensor): posterior belief. size=[batch_size, state_dim]
        """
        batch_size = x.shape[0]
        if a is None:
            logp_z = self.get_initial_state()
            logp_z = logp_z * torch.ones(batch_size, 1)
        else:
            transition = self.get_transition_matrix(a)
            logp_z = torch.sum(transition * b.unsqueeze(-1) + self.eps, dim=-2).log()
        
        logp_x = self.obs_model.log_prob(x)
        b_t = torch.softmax(logp_x + logp_z, dim=-1)
        return b_t

    def beta(self, logb, x_next, a):
        """ Compute log backward message for a single time step

        Args:
            logb (torch.tensor): log backward message. size=[batch_size, state_dim]
            x_next (torch.tensor): next observation vector. size=[batch_size, obs_dim]
            a (torch.tensor): action vector. size=[batch_size, act_dim]

        Returns:
            logb_t (torch.tensor): log backward message. size=[batch_size, state_dim]
        """
        batch_size = logb.shape[0]
        if a is None:
            logp_z = self.transition_model.get_initial_state().unsqueeze(-2)
            logp_z = logp_z * torch.ones(batch_size, self.state_dim, 1)
        else:
            transition = self.get_transition_matrix(a)
            logp_z = torch.log(transition + self.eps)
        
        logp_x = self.obs_model.log_prob(x_next)
        logb_t = torch.logsumexp(
            logp_x.unsqueeze(-2) + logp_z + logb.unsqueeze(-2), dim=-1
        )
        return logb_t

    def _forward(self, x, a):
        """ Forward algorithm

        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, act_dim]

        Returns:
            alpha (torch.tensor): forward messages. size=[T, batch_size, state_dim]
        """
        batch_size = x.shape[1]
        T = x.shape[0]
        
        z0 = torch.ones(batch_size, self.state_dim)

        alpha = [torch.empty(0)] * (T + 1)
        alpha[0] = z0
        for t in range(T):
            x_t = x[t]
            a_t = None if t == 0 else a[t-1]
            alpha[t+1] = self.alpha(alpha[t], x_t, a_t)

        alpha = torch.stack(alpha)[1:]
        return alpha

    def _backward(self, x, a, mask):
        """ Backward algorithm

        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, act_dim]
            mask (torch.tensor): mask sequence. size=[T, batch_size]

        Returns:
            beta (torch.tensor): log backward messages. size=[T, batch_size, state_dim]
        """
        batch_size = x.shape[1]
        T = x.shape[0]

        log_beta = [torch.empty(0)] * T
        log_beta[-1] = torch.zeros(batch_size, self.state_dim)
        for t in range(T - 1, 0, -1):
            x_t = x[t]
            a_t = None if t == 0 else a[t-1]
            log_beta[t-1] = self.beta(log_beta[t], x_t, a_t)

            # mask sequence tail
            log_beta[t-1] *= mask[t].view(-1, 1) == 1
        
        log_beta = torch.stack(log_beta)
        return log_beta
    
    def Q_z(self, alpha, log_beta):
        """ Compute state marginals
        
        Args:
            alpha (torch.tensor): forward messages. size=[T, batch_size, state_dim]
            log_beta (torch.tensor): log backward messages. size=[T, batch_size, state_dim]
        
        Returns:
            q_z (torch.tensor): posterior marginal sequence. size=[T, batch_size, state_dim]
        """
        log_alpha = torch.log(alpha + self.eps)
        q_z = torch.softmax(log_alpha + log_beta, dim=-1)
        return q_z

    def Q_zz(self, alpha, log_beta, x, a):
        """ Compute transition marginals

        Args:
            alpha (torch.tensor): forward messages. size=[T, batch_size, state_dim]
            log_beta (torch.tensor): log backward messages. size=[T, batch_size, state_dim]
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, ctl_dim]

        Returns:
            q_zz (torch.tensor): posterior joint sequence. size=[T, batch_size, state_dim, state_dim]
        """
        T = x.shape[0]
        
        logp_x = [torch.empty(0)] * T
        logp_trans = [torch.empty(0)] * T
        for t in range(1, T):
            logp_x[t] = self.obs_model.log_prob(x[t])
            logp_trans[t] = self.get_transition_matrix(a[t-1])
        logp_x = torch.stack(logp_x[1:]).unsqueeze(-2)
        logp_trans = torch.stack(logp_trans[1:]) 
        
        log_alpha_ = torch.log(alpha + self.eps)[1:].unsqueeze(-1)
        log_beta_ = log_beta[:-1].unsqueeze(-2)
        logits = log_alpha_ + logp_trans + logp_x + log_beta_
        q_zz = softmax(logits, dims=[-1, -2])
        return q_zz
    
    def predict(self, x, a, prior=True, inference=True, num_samples=10):
        """
        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, ctl_dim]
            prior (bool, optional): whether to use prior predictive. 
                If false use posterior predictive. Default=True
            inference (bool, optional): inference mode return likelihood. 
                If false return samples. Default=True
            num_samples (int, optional): number of samples to return. Default=10

        Returns:
            logp (torch.tensor): predictive log likelihood. size=[T, batch_size]
            x_pred (torch.tensor): prediction. size=[T, batch_size, obs_dim]
            x_sample (torch.tensor): sampled sequence. size=[num_samples, T, batch_size, obs_dim]
            alpha (torch.tensor): forward messages. size=[T, batch_size, state_dim]
        """
        alpha = self._forward(x, a)
        
        batch_size = x.shape[1]
        if prior:
            z0 = self.transition_model.get_initial_state()
            z0 = z0 * torch.ones(batch_size, 1)
            z = torch.cat([z0.unsqueeze(0), alpha[:-1]], dim=0)
        else:
            z = alpha

        if not inference:
            logp = self.obs_model.mixture_log_prob(z, x)
            return logp
        else:
            # model average prediction
            x_pred = self.obs_model.bayesian_average(z)
            x_sample = self.obs_model.ancestral_sample(z, num_samples)
            return x_pred, x_sample, alpha


class ContinuousGaussianHMM(nn.Module):
    """Input-output hidden markov model with:
        discrete transitions, continuous actions, gaussian observations
    """
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, 
        rank=0, obs_cov="full", ctl_cov="full"):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): discrete action dimension
            obs_dim (int): observation dimension
            ctl_dim (int): control dimension
            obs_cov (str): observation covariance type. choices=["full", "diag], default="full"
            ctl_cov (str): control covariance type. choices=["full", "diag], default="full"
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.eps = 1e-6
        
        self.obs_model = ConditionalGaussian(
            obs_dim, state_dim, cov=obs_cov, batch_norm=True
        )
        self.transition_model = DiscreteMC(state_dim, act_dim, rank)
        self.ctl_model = ConditionalGaussian(
            ctl_dim, act_dim, cov=ctl_cov, batch_norm=True
        )
        self.act_prior = nn.Parameter(torch.randn(state_dim, act_dim))
        nn.init.xavier_normal_(self.act_prior, gain=1.)
    
    def get_initial_state(self):
        return self.transition_model.get_initial_state()

    def get_transition_matrix(self, a):
        """
        Args:
            a (torch.tensor): action vector. size=[batch_size, act_dim]

        Returns:
            transition_matrix (torch.tensor): action conditioned transition matrix.
                size=[batch_size, state_dim, state_dim]
        """
        a_ = a.unsqueeze(-1).unsqueeze(-1)
        transition_matrix = self.transition_model.get_transition_matrix()
        transition_matrix = torch.sum(transition_matrix * a_, dim=-3)
        return transition_matrix
    
    def alpha(self, b, x, u=None, a=None):
        """ Compute forward message for a single time step

        Args:
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            x (torch.tensor): observation vector. size=[batch_size, obs_dim]
            u (torch.tensor, None, optional): control vector. size=[batch_size, ctl_dim]. 
                None for initial state. Default=None
            a (torch.tensor, None, optional): action prior, to be supplied by a planner.
                size=[batch_size, act_dim]. Default=None

        Returns:
            b_t (torch.tensor): state posterior. size=[batch_size, state_dim]
            a_t (torch.tensor): action posterior, return None if u is None. 
                size=[batch_size, act_dim]
        """
        batch_size = x.shape[0]
        if u is None:
            logp_z = self.get_initial_state()
            logp_z = logp_z * torch.ones(batch_size, 1)
            a_t = None
        else:
            if a is None:
                a_logits = torch.sum(b.unsqueeze(-1) * self.act_prior.unsqueeze(0), dim=-2)
                a = torch.softmax(a_logits, dim=-1)
            a_t = self.ctl_model.infer(a, u)
            transition = self.get_transition_matrix(a_t)
            logp_z = torch.sum(transition * b.unsqueeze(-1) + self.eps, dim=-2).log()
        
        logp_x = self.obs_model.log_prob(x)
        b_t = torch.softmax(logp_x + logp_z, dim=-1)
        return b_t, a_t
    
    def _forward(self, x, u):
        """ Forward algorithm

        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]

        Returns:
            alpha_b (torch.tensor): state forward messages. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action forward messages. size=[T-1, batch_size, state_dim]
        """
        batch_size = x.shape[1]
        T = x.shape[0]
        
        z0 = torch.ones(batch_size, self.state_dim)

        alpha_b = [torch.empty(0)] * (T + 1)
        alpha_b[0] = z0
        alpha_a = [torch.empty(0)] * (T)
        for t in range(T):
            x_t = x[t]
            u_t = None if t == 0 else u[t-1]
            alpha_b[t+1], alpha_a[t] = self.alpha(alpha_b[t], x_t, u_t)

        alpha_b = torch.stack(alpha_b)[1:]
        alpha_a = torch.stack(alpha_a[1:])
        return alpha_b, alpha_a
    
    """ TODO: update beta with action likelihood """
    def beta(self, logb, x_next, a):
        """ Compute log backward message for a single time step

        Args:
            logb (torch.tensor): log backward message. size=[batch_size, state_dim]
            x_next (torch.tensor): next observation vector. size=[batch_size, obs_dim]
            a (torch.tensor): action vector. size=[batch_size, act_dim]

        Returns:
            logb_t (torch.tensor): log backward message. size=[batch_size, state_dim]
        """
        batch_size = logb.shape[0]
        if a is None:
            logp_z = self.transition_model.get_initial_state().unsqueeze(-2)
            logp_z = logp_z * torch.ones(batch_size, self.state_dim, 1)
        else:
            transition = self.get_transition_matrix(a)
            logp_z = torch.log(transition + self.eps)
        
        logp_x = self.obs_model.log_prob(x_next)
        logb_t = torch.logsumexp(
            logp_x.unsqueeze(-2) + logp_z + logb.unsqueeze(-2), dim=-1
        )
        return logb_t

    def predict(self, x, u, prior=True, inference=True, sample_method="ace", num_samples=1):
        """ Predict observations and controls

        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, ctl_dim]
            prior (bool, optional): whether to use prior predictive. 
                If false use posterior predictive. Default=True
            inference (bool, optional): inference mode return likelihood. 
                If false return samples. Default=True
            sample_method (str, optional): sampling method. choices=["ace", "bma"], Default="ace"
            num_samples (int, optional): number of samples to return. Default=1

        Returns:
            logp_o (torch.tensor): observation likelihood. size=[T, batch_size]
            logp_u (torch.tensor): control likelihood. size=[T-1, batch_size]
            x_sample (torch.tensor): sampled observation sequence. size=[num_samples, T, batch_size, obs_dim]
            u_sample (torch.tensor): sampled control sequence. size=[num_samples, T-1, batch_size, ctl_dim]
            alpha_b (torch.tensor): state forward messages. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action forward messages. size=[T-1, batch_size, act_dim]
        """
        alpha_b, alpha_a = self._forward(x, u)
        
        batch_size = x.shape[1]
        if prior:
            z0 = self.transition_model.get_initial_state()
            z0 = z0 * torch.ones(batch_size, 1)
            z = torch.cat([z0.unsqueeze(0), alpha_b[:-1]], dim=0)
        else:
            z = alpha_b
        
        if not inference:
            logp_o = self.obs_model.mixture_log_prob(z, x)
            logp_u = self.ctl_model.mixture_log_prob(alpha_a, u[:-1])
            return logp_o, logp_u
        else:
            # model average prediction
            if sample_method == "ace":
                x_sample = self.obs_model.ancestral_sample(z, num_samples)
                u_sample = self.ctl_model.ancestral_sample(alpha_a, num_samples)
            else:
                x_sample = self.obs_model.bayesian_average(z)
                u_sample = self.ctl_model.bayesian_average(alpha_a)
            
            return x_sample, u_sample, alpha_b, alpha_a


class LogisticGaussianHMM(nn.Module):
    """ Input-output hidden markov model with:
        logistic transitions, continuous actions, gaussian observations
    """
    def __init__(self, state_dim, act_dim, obs_dim, cov="full"):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): continuous action dimension
            obs_dim (int): observation dimension
            cov (str): observation covariance type ["full", "diag], default="full"
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.eps = 1e-6

        self.obs_model = ConditionalGaussian(
            obs_dim, state_dim, cov=cov, batch_norm=True
        )
        self.transition_model = LogisticMC(state_dim, act_dim)
    
    def get_transition_matrix(self, a, params=None):
        return self.transition_model.get_transition_matrix(a, params)
        
    def alpha(self, b, x, a=None):
        """ Compute forward message for a single time step

        Args:
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            x (torch.tensor): observation vector. size=[batch_size, obs_dim]
            a (torch.tensor, None): action vector. size=[batch_size, act_dim]
                None for initial state

        Returns:
            b_t (torch.tensor): posterior belief. size=[batch_size, state_dim]
        """
        batch_size = x.shape[0]
        if a is None:
            logp_z = self.transition_model.get_initial_state()
            logp_z = logp_z * torch.ones(batch_size, 1)
        else:
            transition = self.get_transition_matrix(a)
            logp_z = torch.sum(transition * b.unsqueeze(-1) + self.eps, dim=-2).log()
        
        logp_x = self.obs_model.log_prob(x)
        b_t = torch.softmax(logp_x + logp_z, dim=-1)
        return b_t

    def beta(self, logb, x_next, a):
        """ Compute backward message for a single time step

        Args:
            logb (torch.tensor): log backward message. size=[batch_size, state_dim]
            x_next (torch.tensor): next observation vector. size=[batch_size, obs_dim]
            a (torch.tensor): action vector. size=[batch_size, act_dim]

        Returns:
            logb_t (torch.tensor): backward message. size=[batch_size, state_dim]
        """
        batch_size = logb.shape[0]
        if a is None:
            logp_z = self.transition_model.get_initial_state().unsqueeze(-2)
            logp_z = logp_z * torch.ones(batch_size, self.state_dim, 1)
        else:
            transition = self.get_transition_matrix(a)
            logp_z = torch.log(transition + self.eps)
        
        logp_x = self.obs_model.log_prob(x_next)
        logb_t = torch.logsumexp(
            logp_x.unsqueeze(-2) + logp_z + logb.unsqueeze(-2), dim=-1
        )
        return logb_t

    def _forward(self, x, a):
        """ Forward algorithm

        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, act_dim]

        Returns:
            alpha (torch.tensor): forward messages. size=[T, batch_size, state_dim]
        """
        batch_size = x.shape[1]
        T = x.shape[0]
        
        z0 = torch.ones(batch_size, self.state_dim)

        alpha = [torch.empty(0)] * (T + 1)
        alpha[0] = z0
        for t in range(T):
            x_t = x[t]
            a_t = None if t == 0 else a[t-1]
            alpha[t+1] = self.alpha(alpha[t], x_t, a_t)

        alpha = torch.stack(alpha)[1:]
        return alpha

    def _backward(self, x, a, mask):
        """ Backward algorithm

        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, act_dim]
            mask (torch.tensor): mask sequence. size=[T, batch_size]

        Returns:
            beta (torch.tensor): log backward messages. size=[T, batch_size, state_dim]
        """
        batch_size = x.shape[1]
        T = x.shape[0]

        log_beta = [torch.empty(0)] * T
        log_beta[-1] = torch.zeros(batch_size, self.state_dim)
        for t in range(T - 1, 0, -1):
            x_t = x[t]
            a_t = None if t == 0 else a[t-1]
            log_beta[t-1] = self.beta(log_beta[t], x_t, a_t)

            # mask sequence tail
            log_beta[t-1] *= mask[t].view(-1, 1) == 1
        
        log_beta = torch.stack(log_beta)
        return log_beta
    
    def Q_z(self, alpha, log_beta):
        """ Compute state marginals
        
        Args:
            alpha (torch.tensor): forward messages. size=[T, batch_size, state_dim]
            log_beta (torch.tensor): log backward messages. size=[T, batch_size, state_dim]
        
        Returns:
            q_z (torch.tensor): posterior marginal sequence. size=[T, batch_size, state_dim]
        """
        log_alpha = torch.log(alpha + self.eps)
        q_z = torch.softmax(log_alpha + log_beta, dim=-1)
        return q_z

    def Q_zz(self, alpha, log_beta, x, a):
        """ Compute transition marginals

        Args:
            alpha (torch.tensor): forward messages. size=[T, batch_size, state_dim]
            log_beta (torch.tensor): log backward messages. size=[T, batch_size, state_dim]
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, ctl_dim]

        Returns:
            q_zz (torch.tensor): posterior joint sequence. size=[T, batch_size, state_dim, state_dim]
        """
        T = x.shape[0]
        
        logp_x = [torch.empty(0)] * T
        logp_trans = [torch.empty(0)] * T
        for t in range(1, T):
            logp_x[t] = self.obs_model.log_prob(x[t])
            logp_trans[t] = self.get_transition_matrix(a[t-1])
        logp_x = torch.stack(logp_x[1:]).unsqueeze(-2)
        logp_trans = torch.stack(logp_trans[1:]) 
        
        log_alpha_ = torch.log(alpha + self.eps)[1:].unsqueeze(-1)
        log_beta_ = log_beta[:-1].unsqueeze(-2)
        logits = log_alpha_ + logp_trans + logp_x + log_beta_
        q_zz = softmax(logits, dims=[-1, -2])
        return q_zz
    
    def predict(self, x, a, prior=True, inference=True, num_samples=10):
        """
        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, ctl_dim]
            prior (bool, optional): whether to use prior predictive. 
                If false use posterior predictive. Default=True
            inference (bool, optional): inference mode return likelihood. 
                If false return samples. Default=True
            num_samples (int, optional): number of samples to return. Default=10

        Returns:
            logp (torch.tensor): predictive log likelihood. size=[T, batch_size]
            x_pred (torch.tensor): prediction. size=[T, batch_size, obs_dim]
            x_sample (torch.tensor): sampled sequence. size=[num_samples, T, batch_size, obs_dim]
            alpha (torch.tensor): forward messages. size=[T, batch_size, state_dim]
        """
        alpha = self._forward(x, a)
        
        batch_size = x.shape[1]
        if prior:
            z0 = self.transition_model.get_initial_state()
            z0 = z0 * torch.ones(batch_size, 1)
            z = torch.cat([z0.unsqueeze(0), alpha[:-1]], dim=0)
        else:
            z = alpha

        if not inference:
            logp = self.obs_model.mixture_log_prob(z, x)
            return logp
        else:
            # model average prediction
            x_pred = self.obs_model.bayesian_average(z)
            x_sample = self.obs_model.ancestral_sample(z, num_samples)
            return x_pred, x_sample, alpha


    def take_gradient_step(self, x, a, mask, optimizer, m_steps=1, train=True):
        """
        Args:
            x (torch.tensor): observation sequence [T, batch_size, obs_dim]
            a (torch.tensor): action sequence [T, batch_size, ctl_dim]
            mask (torch.tensor): mask sequence [T, batch_size]
            optimizer (torch.optim): optimizer
            m_steps (int, optional): number of m steps. Default=1
            train (bool, optional): whether in train mode. Default=True

        Returns:
            stats (dict): loss stats
        """
        if train:
            self.train()
        else:
            self.eval()
            m_steps = 1
        
        batch_size = x.shape[1]
        T = x.shape[0]

        # e step
        with torch.no_grad():
            alpha = self._forward(x, a)
            log_beta = self._backward(x, a, mask)
            q_z = self.Q_z(alpha, log_beta)
            q_zz = self.Q_zz(alpha, log_beta, x, a)

        # m step
        for i in range(m_steps):
            # get likelihood
            logp_x = [torch.empty(0)] * (T)
            logp_trans = [torch.empty(0)] * (T - 1)
            for t in range(T):
                logp_x[t] = self.obs_model.log_prob(x[t])
                if t < (T - 1):
                    logp_trans[t] = self.get_transition_matrix(a[t]).log()
            logp_x = torch.stack(logp_x)
            logp_trans = torch.stack(logp_trans)
            logp_z0 = self.transition_model.get_initial_state().log()
            
            # compute loss
            loss_x = torch.sum(q_z * logp_x, dim=-1) * mask
            loss_trans = torch.sum(q_zz * logp_trans, dim=-1).sum(-1) * mask[:-1]
            loss_z0 = torch.sum(q_z[0] * logp_z0, dim=-1)

            loss_x = loss_x.sum(0) / mask.sum(0)
            loss_trans = loss_trans.sum(0) / mask[:-1].sum(0)

            loss_total = -torch.mean(loss_x + loss_trans + loss_z0)
            
            if train:
                loss_total.backward()
                optimizer.step()
                optimizer.zero_grad()

        stats = {
            "loss": loss_total.data.item(),
            "loss_x": loss_x.data.mean().item(),
            "loss_trans": loss_trans.data.mean().item(),
            "loss_z0": loss_z0.data.mean().item()
        }
        return stats