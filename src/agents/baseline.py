import torch
import torch.nn as nn

class ExpertNetwork(nn.Module):
    def __init__(self, act_dim, obs_dim, ctl_dim, nb=False, prod=False):
        """
        Args:
            act_dim (int): action dimension
            obs_dim (int): observation dimension
            ctl_dim (int): control dimension
            nb (bool, optional): naive bayes observation model. Defaults to False.
            prod (bool, optional): product of experts observation model. Defaults to False.
        """
        super().__init__()
        self.prod = prod
        self.nb = nb
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        
        self.lin = nn.Linear(obs_dim, act_dim)
        self.mu = nn.Parameter(torch.randn(act_dim, ctl_dim), requires_grad=True)
        self.lv = nn.Parameter(torch.randn(act_dim, ctl_dim), requires_grad=True)
        
        nn.init.xavier_normal_(self.mu, gain=1.)
        nn.init.xavier_normal_(self.lv, gain=1.)
        
        if self.nb:
            self.b0 = nn.Parameter(torch.randn(1, act_dim), requires_grad=True)
            self.mu_o = nn.Parameter(torch.randn(act_dim, obs_dim), requires_grad=True)
            self.lv_o = nn.Parameter(torch.randn(act_dim, obs_dim), requires_grad=True)
            
            nn.init.xavier_normal_(self.b0, gain=0.3)
            nn.init.xavier_normal_(self.mu_o, gain=1.)
            nn.init.xavier_normal_(self.lv_o, gain=1.)
            
    def forward(self, o, u, inference=False):
        """
        Args:
            o (torch.tensor): observation sequence [T, batch_size, obs_dim]
            u (torch.tensor): control sequence [T, batch_size, ctl_dim]
            inference (bool, optional): whether in inference model. Defaults to False
            
        Returns:
            logp_pi (torch.tensor): predicted control likelihood [T, batch_size]
            logp_obs (torch.tensor): predicted observation likelihood [T, batch_size]
        """
        # recognition
        if self.nb:
            log_b0 = torch.softmax(self.b0, dim=-1).log()
            logp_o = torch.distributions.Normal(
                self.mu_o.unsqueeze(0), self.lv_o.exp().unsqueeze(0)
            ).log_prob(o.unsqueeze(-2)).sum(dim=-1)
            p_a = torch.softmax(log_b0 + logp_o, dim=-1)
            
            logp_obs = torch.logsumexp(torch.log(p_a + 1e-6) * logp_o, dim=-1)
        else:
            p_a = torch.softmax(self.lin(o), dim=-1)
            logp_obs = torch.zeros_like(o[:, :, 0])
        
        # control
        if self.prod:
            mu = p_a.matmul(self.mu.unsqueeze(0))
            lv = p_a.matmul(self.lv.unsqueeze(0))
            logp_pi = torch.distributions.Normal(mu, lv.exp()).log_prob(u).sum(dim=-1)
        else:
            logp_a = torch.distributions.Normal(self.mu, self.lv.exp()).log_prob(u).sum(dim=-1)
            logp_pi = torch.logsumexp(torch.log(p_a + 1e-6) * logp_a, dim=-1)
        
        if not inference:
            return logp_pi, logp_obs
        else:
            return p_a
    
    def choose_action(self, o, u):
        """ Choose action via Bayesian model averaging

        Args:
            o (torch.tensor): observation sequence [T, batch_size, obs_dim]
            u (torch.tensor): control sequence [T, batch_size, ctl_dim]

        Returns:
            u: predicted control [T, batch_size, ctl_dim]
        """
        p_a = self.forward(o, u, inference=True)
        
        # bayesian model averaging
        u = p_a.matmul(self.mu.unsqueeze(0))
        return u