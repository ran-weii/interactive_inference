import torch
import torch.nn as nn
from src.distributions.models import ConditionalDistribution
from src.distributions.transition_models import LogisticMC
from src.distributions.utils import softmax

class LogisticGaussianHMM(nn.Module):
    """ Input-output hidden markov model with:
        logistic transitions, continuous actions, gaussian observations
    """
    def __init__(self, state_dim, act_dim, obs_dim, cov="full"):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): action dimension
            obs_dim (int): observation dimension
            cov (str): observation covariance type ["full", "diag], default="full"
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.eps = 1e-6

        self.obs_model = ConditionalDistribution(
            obs_dim, state_dim, cov=cov, batch_norm=True
        )
        self.transition_model = LogisticMC(state_dim, act_dim)
    
    def get_transition_matrix(self, a, params=None):
        return self.transition_model.get_transition_matrix(a, params)
        
    def alpha(self, b, x, a=None, params=None):
        """ Forward message
        Args:
            b (torch.tensor): prior belief [batch_size, state_dim]
            x (torch.tensor): observation vector [batch_size, obs_dim]
            a (torch.tensor, None): action vector [batch_size, act_dim]. 
                None for initial state
            params (optional): to be filled

        Returns:
            b_t (torch.tensor): posterior belief [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        if a is None:
            logp_z = self.transition_model.get_initial_state()
            logp_z = logp_z * torch.ones(batch_size, 1)
        else:
            transition = self.get_transition_matrix(a, params)
            logp_z = torch.sum(transition * b.unsqueeze(-1) + self.eps, dim=-2).log()
        
        logp_x = self.obs_model.log_prob(x)
        b_t = torch.softmax(logp_x + logp_z, dim=-1)
        return b_t

    def beta(self, logb, x_next, a, params=None):
        """ Backward message
        Args:
            logb (torch.tensor): log backward message [batch_size, state_dim]
            x_next (torch.tensor): next observation vector [batch_size, obs_dim]
            a (torch.tensor): action vector [batch_size, act_dim]
            params (optional): to be filled

        Returns:
            logb_t (torch.tensor): backward message [batch_size, state_dim]
        """
        batch_size = logb.shape[0]
        if a is None:
            logp_z = self.transition_model.get_initial_state().unsqueeze(-2)
            logp_z = logp_z * torch.ones(batch_size, self.state_dim, 1)
        else:
            transition = self.get_transition_matrix(a, params)
            logp_z = torch.log(transition + self.eps)
        
        logp_x = self.obs_model.log_prob(x_next)
        logb_t = torch.logsumexp(
            logp_x.unsqueeze(-2) + logp_z + logb.unsqueeze(-2), dim=-1
        )
        return logb_t

    def _forward(self, x, a, params=None):
        """ Forward algorithm
        Args:
            x (torch.tensor): observation sequence [T, batch_size, obs_dim]
            a (torch.tensor): action sequence [T, batch_size, ctl_dim]
            params (optional): 

        Returns:
            alpha (torch.tensor): forward messages [T, batch_size, state_dim]
        """
        batch_size = x.shape[1]
        T = x.shape[0]
        
        z0 = torch.ones(batch_size, self.state_dim)

        alpha = [torch.empty(0)] * (T + 1)
        alpha[0] = z0
        for t in range(T):
            x_t = x[t]
            a_t = None if t == 0 else a[t-1]
            alpha[t + 1] = self.alpha(alpha[t], x_t, a_t, params)

        alpha = torch.stack(alpha)[1:]
        return alpha

    def _backward(self, x, a, mask, params=None):
        """ Backward algorithm
        Args:
            x (torch.tensor): observation sequence [T, batch_size, obs_dim]
            a (torch.tensor): action sequence [T, batch_size, ctl_dim]
            mask (torch.tensor): mask sequence [T, batch_size]

        Returns:
            beta (torch.tensor): log backward messages [T, batch_size, state_dim]
        """
        batch_size = x.shape[1]
        T = x.shape[0]

        log_beta = [torch.empty(0)] * T
        log_beta[-1] = torch.zeros(batch_size, self.state_dim)
        for t in range(T - 1, 0, -1):
            x_t = x[t]
            a_t = None if t == 0 else a[t - 1]
            log_beta[t - 1] = self.beta(log_beta[t], x_t, a_t, params)

            # mask sequence tail
            log_beta[t - 1] *= mask[t].view(-1, 1) == 1
        
        log_beta = torch.stack(log_beta)
        return log_beta

    def Q_z(self, alpha, log_beta):
        """ State marginals
        Args:
            alpha (torch.tensor): forward messages [T, batch_size, state_dim]
            log_beta (torch.tensor): backward messages [T, batch_size, state_dim]
        
        Returns:
            q_z (torch.tensor): posterior marginal sequence [T, batch_size, state_dim]
        """
        log_alpha = torch.log(alpha + self.eps)
        q_z = torch.softmax(log_alpha + log_beta, dim=-1)
        return q_z

    def Q_zz(self, alpha, log_beta, x, a):
        """ Transition marginals
        Args:
            alpha (torch.tensor): forward messages [T, batch_size, state_dim]
            log_beta (torch.tensor): backward messages [T, batch_size, state_dim]
            x (torch.tensor): observation sequence [T, batch_size, obs_dim]
            a (torch.tensor): action sequence [T, batch_size, ctl_dim]

        Returns:
            q_zz (torch.tensor): posterior joint sequence [T, batch_size, state_dim, state_dim]
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
            x (torch.tensor): observation sequence [T, batch_size, obs_dim]
            a (torch.tensor): action sequence [T, batch_size, ctl_dim]
            prior (bool, optional): whether to use prior predictive. 
                If false use posterior predictive. Default=True
            inference (bool, optional): inference mode return likelihood. 
                If false return samples. Default=True
            num_samples (int, optional): number of samples to return. Default=10

        Returns:
            logp (torch.tensor): predictive log likelihood [T, batch_size]
            x_pred (torch.tensor): prediction [T, batch_size, obs_dim]
            x_sample (torch.tensor): sampled sequence [num_samples, T, batch_size, obs_dim]
            alpha (torch.tensor): forward messages [T, batch_size, state_dim]
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