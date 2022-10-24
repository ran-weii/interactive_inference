import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from src.agents.models import MLP
from src.distributions.utils import kl_divergence

class ImitationLearning(nn.Module):
    def __init__(self, agent, obs_penalty=0, 
        lr=1e-3, decay=0, grad_clip=None):
        super().__init__()
        self.obs_penalty = obs_penalty
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip
        self.agent = agent
        self.optimizers = [torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )]
        self.loss_keys = ["logp_pi_mean", "logp_obs_mean"]
        
    def train_epoch(self, loader):
        self.train()
        
        epoch_stats = []
        num_samples = 0
        for i, batch in enumerate(loader):
            o, u, mask = batch
            out = self.agent(o, u)
            loss, stats = self.loss(out, mask)
            
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_stats.append(stats)
            num_samples += o.shape[1]
        
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def test_epoch(self, loader):
        self.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                out = self.agent(o, u)
                loss, stats = self.loss(out, mask)
                epoch_stats.append(stats)
            
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def loss(self, agent_out, mask):
        [logp_pi, logp_obs] = agent_out
        
        loss_pi = -torch.sum(mask * logp_pi)
        loss_obs = -torch.sum(mask * logp_obs)
        loss = loss_pi + self.obs_penalty * loss_obs
        
        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = float("nan")
        logp_pi_np = (nan_mask * logp_pi).data.numpy()
        logp_obs_np = (nan_mask * logp_obs).data.numpy()
        stats_dict = {
            "loss": loss.data.numpy(),
            "loss_pi": loss_pi.data.numpy(),
            "loss_obs": loss_obs.data.numpy(),
            "logp_pi_mean": np.nanmean(logp_pi_np),
            "logp_pi_std": np.nanstd(logp_pi_np),
            "logp_pi_min": np.nanmin(logp_pi_np),
            "logp_pi_max": np.nanmax(logp_pi_np),
            "logp_obs_mean": np.nanmean(logp_obs_np),
            "logp_obs_std": np.nanstd(logp_obs_np),
            "logp_obs_min": np.nanmin(logp_obs_np),
            "logp_obs_max": np.nanmax(logp_obs_np),
        }
        return loss, stats_dict
    

class MLEIRL(nn.Module):
    def __init__(
        self, agent, obs_penalty=0, plan_penalty=0, 
        lr=1e-3, decay=0, grad_clip=None
        ):
        super().__init__()
        self.obs_penalty = obs_penalty
        self.plan_penalty = plan_penalty
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip
        
        self.agent = agent
        self.optimizers = [torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )]

        self.grad_history = []
        self.loss_keys = ["logp_pi_mean", "logp_obs_mean", "loss_plan"]
        
    def train_epoch(self, loader):
        self.train()
        
        """ debug gradients """
        epoch_grads = []
        epoch_stats = []
        num_samples = 0
        for i, batch in enumerate(loader):
            o, u, mask = batch
            out = self.agent(o, u)
            loss, stats = self.loss(out, mask)
            
            loss.backward()
            
            """ debug gradients """
            epoch_grads.append(self.get_grad_stats())

            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_stats.append(stats)
            num_samples += u.shape[1]
        
        self.grad_history.append(epoch_grads[-1])
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    """ TODO: add offline metrics to test epoch? """
    def test_epoch(self, loader):
        self.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                out = self.agent(o, u)
                loss, stats = self.loss(out, mask)
                epoch_stats.append(stats)
            
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def planner_loss(self, b):
        batch_size = b.shape[1]
        state_dim = b.shape[-1]
        b = b[1:].view(-1, state_dim).data

        # sample beliefs
        idx = torch.multinomial(
            torch.ones(len(b)), num_samples=batch_size * 2, replacement=False
        )
        b_batch = b[idx]
        b_sample = torch.distributions.Dirichlet(
            1.5 * torch.ones(state_dim)
        ).sample((batch_size * 2,))
        b_batch = torch.cat([b_batch, b_sample], dim=0)
        
        loss_planner = self.agent.planner.loss(b_batch)
        return loss_planner

    def loss(self, agent_out, mask):
        [logp_pi, logp_obs, b, a] = agent_out
        plan_error = self.planner_loss(b)
        
        loss_pi = -torch.mean(torch.sum(mask * logp_pi, dim=0))
        loss_obs = -torch.mean(torch.sum(mask * logp_obs, dim=0))
        loss_plan = torch.mean(plan_error)
        
        loss = loss_pi + self.obs_penalty * loss_obs + self.plan_penalty * loss_plan
        
        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = float("nan")
        logp_pi_np = (nan_mask * logp_pi).data.numpy()
        logp_obs_np = (nan_mask * logp_obs).data.numpy()
        stats_dict = {
            "loss": loss.data.numpy(),
            "loss_pi": loss_pi.data.numpy(),
            "loss_obs": loss_obs.data.numpy(),
            "loss_plan": loss_plan.data.numpy(),
            "logp_pi_mean": np.nanmean(logp_pi_np),
            "logp_pi_std": np.nanstd(logp_pi_np),
            "logp_pi_min": np.nanmin(logp_pi_np),
            "logp_pi_max": np.nanmax(logp_pi_np),
            "logp_obs_mean": np.nanmean(logp_obs_np),
            "logp_obs_std": np.nanstd(logp_obs_np),
            "logp_obs_min": np.nanmin(logp_obs_np),
            "logp_obs_max": np.nanmax(logp_obs_np),
            "loss_plan_mean": plan_error.data.mean().numpy(),
            "loss_plan_std": plan_error.data.std().numpy(),
            "loss_plan_min": plan_error.data.min().numpy(),
            "loss_plan_max": plan_error.data.max().numpy(),
        }
        return loss, stats_dict

    def get_grad_stats(self):
        grad_dict = []
        for name, p in self.agent.named_parameters():
            if p.grad is not None:
                grad = p.grad.data
                grad_dict.append({
                    "p": name,
                    "mean": grad.mean().numpy(),
                    "std": grad.std().numpy(),
                    "min": grad.min().numpy(),
                    "max": grad.max().numpy(),
                    "norm": grad.norm().numpy()
                })
        return grad_dict 


class BayesianIRL(MLEIRL):
    def __init__(self, state_dim, act_dim, obs_dim, ctl_dim, H, 
        obs_dist="mvn", obs_cov="full", ctl_dist="mvn", ctl_cov="full",
        obs_penalty=0, lr=1e-3, decay=0, grad_clip=None):
        super().__init__(state_dim, act_dim, obs_dim, ctl_dim, H, 
        obs_dist, obs_cov, ctl_dist, ctl_cov,
        obs_penalty, lr, decay, grad_clip)
        A_size = sum(self.agent.obs_model.parameter_size)
        B_size = self.agent.hmm.parameter_size[0]
        C_size = self.agent.state_dim
        D_size = self.agent.hmm.parameter_size[1]
        F_size = sum(self.agent.ctl_model.parameter_size)
        tau_size = 1
        self.parameter_keys = ["A", "B", "C", "D", "F", "tau"]
        self.parameter_size = [A_size, B_size, C_size, D_size, F_size, tau_size]
        
        self.init_parameters()
        self.optimizers = [torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=decay
        )]
        
    def init_parameters(self):
        [A_size, B_size, C_size, D_size, F_size, tau_size] = self.parameter_size
        self.A_mu = nn.Parameter(torch.randn(A_size), requires_grad=True)
        self.A_lv = nn.Parameter(torch.randn(A_size), requires_grad=True)
        self.B_mu = nn.Parameter(torch.randn(B_size), requires_grad=True)
        self.B_lv = nn.Parameter(torch.randn(B_size), requires_grad=True)
        self.C_mu = nn.Parameter(torch.randn(C_size), requires_grad=True)
        self.C_lv = nn.Parameter(torch.randn(C_size), requires_grad=True)
        self.D_mu = nn.Parameter(torch.randn(D_size), requires_grad=True)
        self.D_lv = nn.Parameter(torch.randn(D_size), requires_grad=True)
        self.F_mu = nn.Parameter(torch.randn(F_size), requires_grad=True)
        self.F_lv = nn.Parameter(torch.randn(F_size), requires_grad=True)
        self.tau_mu = nn.Parameter(torch.randn(tau_size), requires_grad=True)
        self.tau_lv = nn.Parameter(torch.randn(tau_size), requires_grad=True)
        
        mu_init_std = -np.log(0.3)
        lv_init_mean = np.log(0.3)
        lv_init_std = 0.1
        nn.init.uniform_(self.A_mu, a=-mu_init_std, b=mu_init_std)
        nn.init.uniform_(self.B_mu, a=-mu_init_std, b=mu_init_std)
        nn.init.uniform_(self.C_mu, a=-mu_init_std, b=mu_init_std)
        nn.init.uniform_(self.D_mu, a=-mu_init_std, b=mu_init_std)
        nn.init.uniform_(self.F_mu, a=-mu_init_std, b=mu_init_std)
        nn.init.uniform_(self.tau_mu, a=-mu_init_std, b=mu_init_std)
        
        nn.init.normal_(self.A_lv, mean=lv_init_mean, std=lv_init_std)
        nn.init.normal_(self.B_lv, mean=lv_init_mean, std=lv_init_std)
        nn.init.normal_(self.C_lv, mean=lv_init_mean, std=lv_init_std)
        nn.init.normal_(self.D_lv, mean=lv_init_mean, std=lv_init_std)
        nn.init.normal_(self.F_lv, mean=lv_init_mean, std=lv_init_std)
        nn.init.normal_(self.tau_lv, mean=lv_init_mean, std=lv_init_std)
    
    def load_map_parameters(self):
        """ Load MAP parameters to agent """
        A_mu, A_lv, A_tl, A_sk = self.agent.obs_model.transform_parameters(self.A_mu.unsqueeze(0))
        self.agent.obs_model.mu.data = A_mu.data.view(self.agent.obs_model.mu.shape)
        self.agent.obs_model.lv.data = A_lv.data.view(self.agent.obs_model.lv.shape)
        self.agent.obs_model.tl.data = A_tl.data.view(self.agent.obs_model.tl.shape)
        self.agent.obs_model.sk.data = A_sk.data.view(self.agent.obs_model.sk.shape)
        
        self.agent.hmm.B.data = self.B_mu.data.view(self.agent.hmm.B.shape)
        self.agent.C.data = self.C_mu.data.view(self.agent.C.shape)
        self.agent.hmm.D.data = self.D_mu.data.view(self.agent.hmm.D.shape)
        
        F_mu, F_lv, F_tl, F_sk = self.agent.ctl_model.transform_parameters(self.F_mu.unsqueeze(0))
        self.agent.ctl_model.mu.data = F_mu.data.view(self.agent.ctl_model.mu.shape)
        self.agent.ctl_model.lv.data = F_lv.data.view(self.agent.ctl_model.lv.shape)
        self.agent.ctl_model.tl.data = F_tl.data.view(self.agent.ctl_model.tl.shape)
        self.agent.ctl_model.sk.data = F_sk.data.view(self.agent.ctl_model.sk.shape)
        
        self.agent.tau.data = self.tau_mu.data.view(self.agent.tau.shape)
    
    def train_epoch(self, loader):
        self.train()
        
        epoch_stats = []
        num_samples = 0
        for i, batch in enumerate(loader):
            o, u, mask = batch
            theta = self.encode(o.shape[1])
            out = self.agent(o, u, theta=theta)
            loss, stats = self.loss(out, mask)
            
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_stats.append(stats)
            num_samples += o.shape[1]
        
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def test_epoch(self, loader):
        self.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                theta = self.encode(o.shape[1])
                out = self.agent(o, u, theta=theta)
                loss, stats = self.loss(out, mask)
                epoch_stats.append(stats)
            
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def encode(self, batch_size):
        """ Draw samples from posterior distribution """
        mu = torch.cat(
            [self.A_mu, self.B_mu, self.C_mu, self.D_mu, self.F_mu, self.tau_mu], dim=-1
        )
        lv = torch.cat(
            [self.A_lv, self.B_lv, self.C_lv, self.D_lv, self.F_lv, self.tau_lv], dim=-1
        )
        cov = torch.diag_embed(lv.clip(np.log(1e-6), np.log(1e6)).exp())
        
        theta = torch.distributions.MultivariateNormal(mu, cov).rsample((batch_size,))
        theta = torch.split(theta, self.parameter_size, dim=-1)
        theta_dict = {k: theta[i] for (i, k) in enumerate(self.parameter_keys)}
        return theta_dict
    
    def entropy(self):
        mu = torch.cat(
            [self.A_mu, self.B_mu, self.C_mu, self.D_mu, self.F_mu, self.tau_mu], dim=-1
        )
        lv = torch.cat(
            [self.A_lv, self.B_lv, self.C_lv, self.D_lv, self.F_lv, self.tau_lv], dim=-1
        )
        cov = torch.diag_embed(lv.clip(np.log(1e-6), np.log(1e6)).exp())
        ent = torch.distributions.MultivariateNormal(mu, cov).entropy()
        return ent
    
    def loss(self, agent_out, mask):
        # entropy loss
        loss_ent = -self.entropy()
        
        # likelihood loss
        [logp_pi, logp_obs] = agent_out
        
        loss_pi = -torch.sum(mask * logp_pi)
        loss_obs = -torch.sum(mask * logp_obs)
        loss = loss_pi + self.obs_penalty * loss_obs + loss_ent
        
        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = float("nan")
        logp_pi_np = (nan_mask * logp_pi).data.numpy()
        logp_obs_np = (nan_mask * logp_obs).data.numpy()
        stats_dict = {
            "loss": loss.data.numpy(),
            "loss_pi": loss_pi.data.numpy(),
            "loss_obs": loss_obs.data.numpy(),
            "loss_ent": loss_ent.data.numpy(),
            "logp_pi_mean": np.nanmean(logp_pi_np),
            "logp_pi_std": np.nanstd(logp_pi_np),
            "logp_pi_min": np.nanmin(logp_pi_np),
            "logp_pi_max": np.nanmax(logp_pi_np),
            "logp_obs_mean": np.nanmean(logp_obs_np),
            "logp_obs_std": np.nanstd(logp_obs_np),
            "logp_obs_min": np.nanmin(logp_obs_np),
            "logp_obs_max": np.nanmax(logp_obs_np),
        }
        return loss, stats_dict
    

class ReverseKL(nn.Module):
    """ Reverse KL minimization using Expected Information Maximization """
    def __init__(
        self, agent, mle_penalty=0.1, obs_penalty=0, plan_penalty=0, 
        lr=1e-3, decay=0, grad_clip=None, 
        e_step=20, m_step=1, batch_size=512, num_samples=10
        ):
        super().__init__()
        self.mle_penalty = mle_penalty
        self.obs_penalty = obs_penalty
        self.plan_penalty = plan_penalty
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip
        self.e_step = e_step
        self.m_step = m_step
        self.batch_size = batch_size
        self.num_samples = num_samples 
        
        # init agent
        self.agent = agent
        self.agent_optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )

        # init classifier
        clf_input_dim = self.agent.ctl_dim + self.agent.state_dim + self.agent.obs_dim
        clf_output_dim = 1
        clf_hidden_dim = 64
        clf_num_hidden = 2
        self.clf = MLP(
            clf_input_dim, clf_output_dim, clf_hidden_dim, clf_num_hidden, "silu"
        )
        self.clf_optimizer = torch.optim.Adam(
            self.clf.parameters(), lr=lr, weight_decay=decay
        )

        self.grad_history = []
        self.loss_keys = ["logp_pi_mean", "logp_obs_mean", "loss_clf", "loss_a", "loss_cmp"]
        
    def train_epoch(self, loader):
        self.train()
        
        epoch_stats = []
        num_samples = 0
        for i, batch in enumerate(loader):
            o, u, mask = batch
            loss, stats = self.take_gradient_step(o, u, mask, train=True)
            
            epoch_stats.append(stats)
            num_samples += u.shape[1]
        
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def test_epoch(self, loader):
        self.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                loss, stats = self.take_gradient_step(o, u, mask, train=False)
                epoch_stats.append(stats)
        
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def planner_loss(self, b):
        batch_size = b.shape[1]
        state_dim = b.shape[-1]
        b = b[1:].view(-1, state_dim).data

        # sample beliefs
        idx = torch.multinomial(
            torch.ones(len(b)), num_samples=batch_size * 2, replacement=False
        )
        b_batch = b[idx]
        b_sample = torch.distributions.Dirichlet(
            1.5 * torch.ones(state_dim)
        ).sample((batch_size * 2,))
        b_batch = torch.cat([b_batch, b_sample], dim=0)
        
        loss_planner = self.agent.planner.loss(b_batch)
        return loss_planner

    def sample_batch(self, inputs, mask, start_dim, end_dim, num_samples):
        """ Flatten the temportal dimension for all input tensors and subsample """
        mask_flat = mask.flatten(start_dim, end_dim)
        mask_flat, sort_id = torch.sort(mask_flat, descending=True)
        
        permute_id = torch.randperm(int(mask_flat.sum()))
        sort_id[mask_flat == 1] = sort_id[mask_flat == 1][permute_id]
        mask_flat[mask_flat == 1] = mask_flat[mask_flat == 1][permute_id]
        num_samples = min(mask_flat.shape[0], num_samples)
        
        outputs = []
        for x in inputs:
            x_flat = x.flatten(start_dim, end_dim)
            x_flat = x_flat[sort_id]
            x_flat = x_flat[:num_samples]
            outputs.append(x_flat)
        mask_flat = mask_flat[:num_samples]
        return outputs, mask_flat

    def take_gradient_step(self, o, u, mask, train=True):
        e_step = self.e_step if train else 1
        m_step = self.m_step if train else 1

        with torch.no_grad():
            [_, _, b_old, a_old] = self.agent(o, u)

        # train density ration estimator
        for i in range(e_step):
            with torch.no_grad():
                fake_samples = self.agent.ctl_model.ancestral_sample(a_old[:-1], 1).squeeze(0)
            
            # create classification dataset
            """ add observations to clf input """
            [fake_samples, u_samples, b_old_samples, o_samples], mask_samples = self.sample_batch(
                [fake_samples, u, b_old[:-1], o], mask, 0, 1, self.batch_size
            )
            fake_inputs = torch.cat([fake_samples, b_old_samples, o_samples], dim=-1)
            real_inputs = torch.cat([u_samples, b_old_samples, o_samples], dim=-1)
            clf_inputs = torch.cat([real_inputs, fake_inputs], dim=-2)
            labels = torch.cat(
                [torch.zeros(real_inputs.shape[:-1]), torch.ones(fake_inputs.shape[:-1])], dim=-1
            ).unsqueeze(-1)
            masks = torch.cat([mask_samples, mask_samples], dim=-1).unsqueeze(-1)

            clf_pred = torch.sigmoid(self.clf(clf_inputs))
            clf_loss = F.binary_cross_entropy(clf_pred, labels, weight=masks)
            
            if train:
                clf_loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.clf_optimizer.step()
                self.clf_optimizer.zero_grad()
                self.agent_optimizer.zero_grad()
            # print("clf_loss", clf_loss.data.item())

        # train mixing weights
        for i in range(m_step):
            [logp_pi, logp_obs, b, a] = self.agent(o, u)
            """ add observations to clf input """
            [b_samples, a_samples, a_old_samples, o_samples], mask_samples = self.sample_batch(
                [b[:-1], a[:-1], a_old[:-1], o], mask, 0, 1, self.batch_size
            )
            
            with torch.no_grad():
                fake_samples = self.agent.ctl_model.sample((self.num_samples,))
                fake_samples = fake_samples.repeat_interleave(b_samples.shape[0], 1)
            
            b_inputs = b_samples.unsqueeze(0).unsqueeze(-2)
            b_inputs = b_inputs.repeat_interleave(self.num_samples, 0)
            b_inputs = b_inputs.repeat_interleave(self.agent.act_dim, -2)

            o_inputs = o_samples.unsqueeze(0).unsqueeze(-2)
            o_inputs = o_inputs.repeat_interleave(self.num_samples, 0)
            o_inputs = o_inputs.repeat_interleave(self.agent.act_dim, -2)

            fake_inputs = torch.cat([fake_samples, b_inputs, o_inputs], dim=-1)
            
            # compute density ration reward
            log_r_u = self.clf(fake_inputs).squeeze(-1)
            log_r_a = log_r_u.mean(0)
            log_r = torch.sum(a_samples * log_r_a, dim=-1)
            kl = kl_divergence(a_samples, a_old_samples)
            a_loss = torch.mean(mask_samples * (log_r + kl))
            
            loss_pi = -torch.mean(torch.sum(mask * logp_pi, dim=0))
            loss_obs = -torch.mean(torch.sum(mask * logp_obs, dim=0))
            plan_error = self.planner_loss(b)
            loss_plan = torch.mean(plan_error)
            
            total_loss = a_loss + self.mle_penalty * loss_pi + self.obs_penalty * loss_obs + self.plan_penalty * loss_plan
            
            if train:
                total_loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.agent_optimizer.step()
                self.agent_optimizer.zero_grad()
                self.clf_optimizer.zero_grad()
            # print("a_loss", a_loss.data.item())
        
        # train components
        with torch.no_grad():
            ctl_dist_old = self.agent.ctl_model.get_distribution_class(transform=False, requires_grad=False)
            [logp_pi, logp_obs, b, a] = self.agent(o, u)

        for i in range(m_step):
            [b_samples, a_samples, o_samples], mask_samples = self.sample_batch(
                [b[:-1], a[:-1], o], mask, 0, 1, self.batch_size
            )
            
            fake_samples = self.agent.ctl_model.sample((self.num_samples,))
            fake_samples = fake_samples.repeat_interleave(b_samples.shape[0], 1)
            
            b_inputs = b_samples.unsqueeze(0).unsqueeze(-2)
            b_inputs = b_inputs.repeat_interleave(self.num_samples, 0)
            b_inputs = b_inputs.repeat_interleave(self.agent.act_dim, -2)

            o_inputs = o_samples.unsqueeze(0).unsqueeze(-2)
            o_inputs = o_inputs.repeat_interleave(self.num_samples, 0)
            o_inputs = o_inputs.repeat_interleave(self.agent.act_dim, -2)
            
            fake_inputs = torch.cat([fake_samples, b_inputs, o_inputs], dim=-1)
            
            log_r_u = self.clf(fake_inputs).squeeze(-1)
            ctl_dist = self.agent.ctl_model.get_distribution_class(transform=False)
            kl = torch.distributions.kl.kl_divergence(ctl_dist, ctl_dist_old)
            
            w = a_samples / a_samples.mean(0, keepdim=True)
            cmp_loss = torch.mean(
                mask_samples.unsqueeze(-1) * w * (log_r_u.mean(0) + kl), dim=0
            ).sum()
            
            if train:
                cmp_loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.agent_optimizer.step()
                self.agent_optimizer.zero_grad()
                self.clf_optimizer.zero_grad()
            
            # print("cmp_loss", cmp_loss.data.item())
        
        loss_pi = -torch.mean(torch.sum(mask * logp_pi, dim=0))
        loss_obs = -torch.mean(torch.sum(mask * logp_obs, dim=0))
        loss = loss_pi + self.obs_penalty * loss_obs + self.plan_penalty * loss_plan
        
        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = float("nan")
        logp_pi_np = (nan_mask * logp_pi).data.numpy()
        logp_obs_np = (nan_mask * logp_obs).data.numpy()
        stats_dict = {
            "loss": loss.data.numpy(),
            "loss_pi": loss_pi.data.numpy(),
            "loss_obs": loss_obs.data.numpy(),
            "loss_plan": loss_plan.data.numpy(),
            "logp_pi_mean": np.nanmean(logp_pi_np),
            "logp_pi_std": np.nanstd(logp_pi_np),
            "logp_pi_min": np.nanmin(logp_pi_np),
            "logp_pi_max": np.nanmax(logp_pi_np),
            "logp_obs_mean": np.nanmean(logp_obs_np),
            "logp_obs_std": np.nanstd(logp_obs_np),
            "logp_obs_min": np.nanmin(logp_obs_np),
            "logp_obs_max": np.nanmax(logp_obs_np),
            "loss_plan_mean": plan_error.data.mean().numpy(),
            "loss_plan_std": plan_error.data.std().numpy(),
            "loss_plan_min": plan_error.data.min().numpy(),
            "loss_plan_max": plan_error.data.max().numpy(),
            "loss_clf": clf_loss.data.numpy(),
            "loss_a": a_loss.data.numpy(),
            "loss_cmp": cmp_loss.data.numpy()
        }
        return loss, stats_dict