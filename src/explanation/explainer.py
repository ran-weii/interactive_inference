import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import torch
import torch.distributions as torch_dist

def compute_markov_stationary_dist(B):
    evals, evecs = np.linalg.eig(B.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    return stationary

def compute_controlled_transition(transition, policy):
    return torch.einsum("kij, ik -> ij", transition, policy)

""" TODO: 
plot controlled transition matrix and stationary distribution
plot observation classification on samples
plot action classification on samples
"""
class VINExplainer:
    """ VIN agent explainer """
    def __init__(self, agent, feature_set, action_set):
        self.state_dim = agent.state_dim
        self.act_dim = agent.act_dim
        self.obs_dim = agent.obs_dim
        self.ctl_dim = agent.ctl_dim
        self.feature_set = feature_set
        self.action_set = action_set

        self.agent = agent

        # compute obs mean and variance from samples
        num_samples = 100
        obs_samples = agent.obs_model.sample((num_samples,)).squeeze(-3)
        
        # init attributes
        with torch.no_grad():
            self.obs_mean = torch.mean(obs_samples, dim=0)
            self.obs_variance = torch.var(obs_samples, dim=0)
            self.ctl_mean = agent.ctl_model.mean().squeeze(0)
            self.ctl_variance = agent.ctl_model.variance().squeeze(0)
            # self.b0 = agent.rnn.init_hidden().squeeze(0).flatten()
            self.transition = agent.rnn.compute_transition().squeeze(0)
            self.target_dist = agent.compute_target_dist().flatten()
            self.pi0 = agent.compute_pi0().squeeze(0)
            self.policy = agent.compute_policy(torch.eye(agent.state_dim)).squeeze(0)

            self.controlled_transition = compute_controlled_transition(self.transition, self.policy)
            self.stationary_dist = torch.from_numpy(compute_markov_stationary_dist(self.controlled_transition))

        self.idx_sort_ctl = torch.arange(self.agent.act_dim)
        self.idx_sort_obs = torch.arange(self.agent.state_dim)
        self.key_feature = ""
        
    def sort_state_action(self, by_feature):
        """
        Args:
            by_feature (str): feature name to sort state
        """
        self.key_feature = by_feature
        self.idx_feature = self.feature_set.index(by_feature)

        self.idx_sort_ctl = torch.argsort(self.ctl_mean[:, 0]) # sort first dimension by default
        self.idx_sort_obs = torch.argsort(self.obs_mean[:, self.idx_feature])
        
        # sort components
        self.ctl_mean = self.ctl_mean[self.idx_sort_ctl]
        self.ctl_variance = self.ctl_variance[self.idx_sort_ctl]

        self.obs_mean = self.obs_mean[self.idx_sort_obs]
        self.obs_variance = self.obs_variance[self.idx_sort_obs]

        # self.b0 = self.b0[self.idx_sort_obs]

        self.transition = self.transition[self.idx_sort_ctl]
        self.transition = self.transition[:, self.idx_sort_obs]
        self.transition = self.transition[:, :, self.idx_sort_obs]

        self.target_dist = self.target_dist[self.idx_sort_obs]

        self.pi0 = self.pi0[self.idx_sort_ctl]
        self.pi0 = self.pi0[:, self.idx_sort_obs]

        self.policy = self.policy[self.idx_sort_obs]
        self.policy = self.policy[:, self.idx_sort_ctl]

        self.controlled_transition = self.controlled_transition[self.idx_sort_obs]
        self.controlled_transition = self.controlled_transition[:, self.idx_sort_obs]

        self.stationary_dist = self.stationary_dist[self.idx_sort_obs]

    def plot_action_component_pdfs(self, num_grids=200, figsize=(6, 4)):
        """ Plot a 1 x ctl_dim panel of action component pdfs. 
        Each component is colored by the first dimension magnitude. 
        """
        num_samples = 100
        with torch.no_grad():
            ctl_sample = self.agent.ctl_model.sample((num_samples,)).squeeze(1)
            
        grid_min = ctl_sample.flatten(0, 1).min(0)[0]
        grid_max = ctl_sample.flatten(0, 1).max(0)[0]
        grid_width = grid_max - grid_min
        grid_min -= grid_width * 0.1
        grid_max += grid_width * 0.1
        grid = [torch.linspace(grid_min[i], grid_max[i], num_grids).view(-1, 1) for i in range(self.ctl_dim)]

        pdf = [0] * self.ctl_dim
        for i in range(self.ctl_dim):
            pdf[i] = torch_dist.Normal(self.ctl_mean[:, i], self.ctl_variance[:, i]**0.5).log_prob(grid[i]).exp()

        cmap = mpl.cm.get_cmap("viridis")
        fig, ax = plt.subplots(1, self.ctl_dim, figsize=figsize)
        if self.ctl_dim == 1:
            ax = [ax]

        for i in range(self.ctl_dim):
            for j in range(self.act_dim):
                ax[i].plot(grid[i], pdf[i][:, j], c=cmap(j/self.act_dim))
            ax[i].set_xlabel(self.action_set[i])

        plt.suptitle("ctl component density sorted by dds")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, ax

    def plot_observation_component_pdfs(self, color_by_value=False, num_grids=200, figsize=(12, 4)):
        """ Plot a 1 x obs_dim panel of observation component pdfs. 
        Each component is either colored by the key feature or the target dist pdf value.
        """
        # compute grid size
        num_samples = 300
        with torch.no_grad():
            obs_sample = self.agent.obs_model.sample((num_samples,)).squeeze(1)

        num_grids = 200
        grid_min = obs_sample.flatten(0, 1).min(0)[0]
        grid_max = obs_sample.flatten(0, 1).max(0)[0]
        grid_width = grid_max - grid_min
        grid_min -= grid_width * 0.1
        grid_max += grid_width * 0.1
        grid = [torch.linspace(grid_min[i], grid_max[i], num_grids).view(-1, 1) for i in range(self.obs_dim)]
        
        # additional sorting
        if color_by_value:
            idx_sort_obs = torch.argsort(self.target_dist, descending=False)
            obs_mean = self.obs_mean[idx_sort_obs]
            obs_variance = self.obs_variance[idx_sort_obs]
        else:
            obs_mean = self.obs_mean
            obs_variance = self.obs_variance
        
        pdf = [0] * self.obs_dim
        for i in range(self.obs_dim):
            pdf[i] = torch_dist.Normal(obs_mean[:, i], obs_variance[:, i]**0.5).log_prob(grid[i]).exp()
        
        cmap = mpl.cm.get_cmap("viridis")
        fig, ax = plt.subplots(1, self.obs_dim, figsize=figsize)
        for i in range(self.obs_dim):
            for j in range(self.state_dim): 
                ax[i].plot(grid[i], pdf[i][:, j], c=cmap(j/self.state_dim))
            ax[i].set_xlabel(self.feature_set[i])
        
        by = "target value" if color_by_value else self.key_feature
        plt.suptitle(f"obs component density colored by {by}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, ax

    """ NOTE: consider plotting the colors with mixture pdfs """
    def plot_observation_scatter_matrix(
        self, color_by="none", plot_component_centers=True, num_samples=300, figsize=(8, 8)
        ):
        """ Plot observation component samples as a scatter matrix. 
        Each component is colored either by the key feature or the target dist pdf value 

        Args:
            color_by (str): self attribute to resort the observations by. choices=[target_dist, stationary_dist]
            plot_component_centers (bool): if true, plot component centers with red x. 
        """
        with torch.no_grad():
            obs_sample = self.agent.obs_model.sample((num_samples,)).squeeze(1) 
            obs_sample = obs_sample[:, self.idx_sort_obs]
        
        if color_by == "target_dist":
            idx_sort = torch.argsort(self.target_dist, descending=False)
        elif color_by == "stationary_dist":
            idx_sort = torch.argsort(self.stationary_dist, descending=False)
        else:
            color_by = self.key_feature
            idx_sort = torch.arange(self.state_dim)
        obs_sample = obs_sample[:, idx_sort]
        
        # pack as df
        state_index = torch.arange(self.state_dim).view(1, -1, 1).repeat_interleave(num_samples, 0)
        data = torch.cat([obs_sample, state_index], dim=-1).transpose(0, 1).flatten(0, 1)
        df = pd.DataFrame(
            data.numpy(),
            columns=self.feature_set + ["state_index"]
        )
        
        fig, ax = plt.subplots(self.obs_dim, self.obs_dim, figsize=figsize)
        pd.plotting.scatter_matrix(
            df[self.feature_set], 
            ax=ax,
            c=df["state_index"],
            diagonal="kde",
        )

        # plot cluster centers
        if plot_component_centers:
            for i in range(self.obs_dim):
                for j in range(self.obs_dim):
                    if i != j:
                        ax[i, j].plot(
                            self.obs_mean[:, j], 
                            self.obs_mean[:, i], 
                            "rx", ms=figsize[0]
                        )
        
        # by = "target value" if color_by_value else self.key_feature
        plt.suptitle(f"obs component scatter matrix colored by {color_by}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, ax
    
    def plot_observation_scatter_3d(self, x="lv_s_rel", y="lv_ds_rel", z="lv_inv_tau", color_var="state_index"):
        """ Plot observation component samples as plotly 3d scatter plot """
        num_samples = 200
        with torch.no_grad():
            obs_samples = self.agent.obs_model.sample((num_samples,)).squeeze(-3)
            target_dist = self.agent.compute_target_dist()
            target_dist = torch.repeat_interleave(target_dist, num_samples, dim=0).unsqueeze(-1)
            state_index = torch.arange(self.state_dim).view(1, -1, 1)
            state_index = torch.repeat_interleave(state_index, num_samples, dim=0)
            data = torch.cat([obs_samples, target_dist, state_index], dim=-1).flatten(0, 1).numpy()
            df = pd.DataFrame(data, columns=self.feature_set + ["target_dist", "state_index"])

        fig = px.scatter_3d(
            df, x=x, y=y, z=z, 
            color=color_var,
            size_max=2
        )
        return fig

    def plot_transition_matrix(self, num_cols=5, annot=False, cbar=False, figsize=(12, 6)):
        """ Plot a num_rows x num_cols panel of transition matrix heatmaps """
        num_rows = np.ceil(len(self.transition) / num_cols).astype(int)
        fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        ax = ax.flat
        for i in range(self.act_dim):
            sns.heatmap(self.transition[i].numpy(), fmt=".2f", annot=annot, cbar=cbar, ax=ax[i])
            ax[i].set_xlabel("next_state")
            ax[i].set_ylabel("state")
            ax[i].set_title(f"ctl={self.ctl_mean[i, -1]:.2f}")

        plt.suptitle("transition matrices")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, ax
    
    def plot_controlled_transition_matrix(self, annot=False, cbar=True, figsize=(6, 6)):
        """ Plot controlled transition matrix as heatmap """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(self.controlled_transition, fmt=".2f", annot=annot, cbar=cbar, ax=ax)
        ax.set_xlabel("next state")
        ax.set_ylabel("state")

        plt.suptitle("controlled transition matrix")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, ax
    
    def plot_controlled_stationary_dist(self, figsize=(6, 4)):
        """ Plot stationary distribution of controlled transition matrix as a bar chart """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.bar(torch.arange(self.state_dim), self.stationary_dist)
        ax.set_xlabel("state")
        ax.set_ylabel("stationary dist pmf")
        plt.suptitle("controlled transition matrix stationary distribution")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, ax

    def plot_target_dist(self, sort_by=None, figsize=(6, 4)):
        """ Plot target dist pdf as bar chart """
        if sort_by is None:
            target_dist = self.target_dist.clone()
            obs_mean = self.obs_mean[:, self.idx_feature]
        else:
            target_dist = self.target_dist.clone()
            idx_feature = self.feature_set.index(sort_by)
            idx_sort = torch.argsort(self.obs_mean[:, idx_feature])

            target_dist = target_dist[idx_sort]
            obs_mean = self.obs_mean[idx_sort, idx_feature]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(obs_mean, target_dist, "-o")
        ax.set_xlabel(sort_by)
        ax.set_ylabel("target dist pdf")
        ax.set_title(f"target dist pdf vs {self.key_feature}, beta={self.agent.beta}")
        plt.tight_layout()
        return fig, ax
    
    def plot_policy_prior(self, figsize=(8, 6)):
        """ Plot prior policy as heat map """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(self.pi0.numpy().round(2).T, annot=True, cbar=False, ax=ax)
        ax.set_xlabel("dds")
        ax.set_ylabel(self.key_feature)
        ax.set_xticklabels([f"{t:.2f}" for t in self.ctl_mean[:, -1]])
        ax.set_yticklabels([f"{t:.2f}" for t in self.obs_mean[:, self.idx_feature]], rotation=0)
        ax.set_title(f"default policy, beta={self.agent.beta}")
        plt.tight_layout()
        return fig, ax

    def plot_policy(self, sort_by=None, figsize=(8, 6)):
        """ Plot policy as heatmap """
        if sort_by is None:
            policy = self.policy.numpy()
            sort_by = self.key_feature
            obs_means = [f"{t:.2f}" for t in self.obs_mean[:, self.idx_feature]]
        else:
            policy = self.policy.numpy()

            idx_feature = self.feature_set.index(sort_by)
            idx_sort = torch.argsort(self.obs_mean[:, idx_feature])
            policy = policy[idx_sort]
            obs_means = [f"{t:.2f}" for t in self.obs_mean[idx_sort, idx_feature]]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(policy, fmt=".2f", annot=True, cbar=False, ax=ax)
        ax.set_xlabel("dds")
        ax.set_ylabel(sort_by)
        ax.set_xticklabels([f"{t:.2f}" for t in self.ctl_mean[:, -1]])
        ax.set_yticklabels(obs_means, rotation=0)
        ax.set_title(f"planned policy, beta={self.agent.beta}")
        plt.tight_layout()
        return fig, ax