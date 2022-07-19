import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.distributions.utils import poisson_pdf

""" TODO: add model explainer class """
def get_active_inference_parameters(agent):
    """ 
    Args:
        agent (ActiveInference): active inference agent object

    Returns:
        theta (dict): active inference parameter dict
    """
    A_mu = agent.obs_model.mean().data.squeeze(0)
    A_sd = torch.sqrt(agent.obs_model.variance()).data.squeeze(0)

    """ TODO: temporary handle """
    if agent.hmm.rank == 0:
        B = torch.softmax(agent.hmm.B, dim=-1).data.squeeze(0)
    else:
        B = torch.softmax(agent.hmm.B.transition, dim=-1).data.squeeze(0)

    C = torch.softmax(agent.rwd_model.C, dim=-1).data.squeeze(0)
    D = torch.softmax(agent.hmm.D, dim=-1).data.squeeze(0)
    F_mu = agent.ctl_model.mean().data.squeeze(0)
    F_sd = torch.sqrt(agent.ctl_model.variance()).data.squeeze(0)
    tau_dist = poisson_pdf(agent.planner.tau.exp(), agent.H).data.squeeze(0)
    tau = tau_dist.dot(torch.arange(len(tau_dist)) + 1.)
    
    theta = {
        "A_mu": A_mu.numpy(),
        "A_sd": A_sd.numpy(),
        "B": B.numpy(),
        "C": C.numpy(),
        "D": D.numpy(),
        "F_mu": F_mu.numpy(),
        "F_sd": F_sd.numpy(),
        "tau_dist": tau_dist.numpy(),
        "tau": tau.numpy()
        
    }
    return theta

def markov_stationary_dist(B):
    evals, evecs = np.linalg.eig(B.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    return stationary


class ModelExplainer:
    def __init__(self, agent, obs_fields):
        self.theta = get_active_inference_parameters(agent)
        self.agent = agent
        self.init_parameters()

        self.sort_a_id = None
        self.sort_s_id = None
        self.A_fields = obs_fields
    
    def init_parameters(self):
        self.A_mu = self.theta["A_mu"]
        self.A_sd = self.theta["A_sd"]
        self.B = self.theta["B"]
        self.C = self.theta["C"]
        self.D = self.theta["D"]
        self.F_mu = self.theta["F_mu"]
        self.F_sd = self.theta["F_sd"]
        self.tau = self.theta["tau_dist"]
        
        self.pi = self.get_policy()
        self.B_pi = np.sum(self.pi[..., None] * self.B, axis=0)
        self.S = markov_stationary_dist(self.B_pi)

        """ debug efe agent reward """
        self.r, self.ekl, self.ent = self.get_reward()
    
    def get_reward(self):
        """ TODO: temporary handle """
        if self.agent.hmm.rank == 0:
            B = torch.softmax(self.agent.hmm.B, dim=-1).data
        else:
            B = torch.softmax(self.agent.hmm.B.transition, dim=-1).data
        # B = torch.softmax(self.agent.hmm.B, dim=-1)
        r = self.agent.rwd_model(B, B).data.squeeze(0)
        ent = -self.agent.obs_model.entropy().data.squeeze(0)
        ekl = (r - ent).squeeze(0)
        return r.numpy(), ekl.numpy(), ent.numpy()

    def get_policy(self):
        theta = self.agent.get_default_parameters()
        Q = self.agent.planner.plan(theta)
        pi = torch.softmax(Q, dim=-2).squeeze(0)
        return pi.data.numpy()

    def sort(self, by=None):
        assert by in [None, "C", "D", "S"]
        # sort actions
        if self.F_mu.shape[1] == 1:
            sort_a_id = np.argsort(self.F_mu[:, 0])[::-1]
        elif self.F_mu.shape[1] == 2:
            psi = np.arctan2(self.F_mu[:, 0], self.F_mu[:, 1])
            sort_a_id = np.argsort(psi)
        else:
            sort_a_id = np.arange(self.F_mu.shape[0])

        # sort states
        if by == "C":
            sort_s_id = np.argsort(self.C)[::-1]
        elif by == "D":
            sort_s_id = np.argsort(self.D)[::-1]
        elif by == "S":
            sort_s_id = np.argsort(self.S)[::-1]

        self.A_mu = self.A_mu[sort_s_id]
        self.A_sd = self.A_sd[sort_s_id]
        self.B = self.B[sort_a_id]
        self.B = self.B[:, sort_s_id]
        self.B = self.B[:, :, sort_s_id]
        self.C = self.C[sort_s_id]
        self.D = self.D[sort_s_id]
        self.F_mu = self.F_mu[sort_a_id]
        self.F_sd = self.F_sd[sort_a_id]

        self.pi = self.pi[sort_a_id]
        self.pi = self.pi[:, sort_s_id]
        self.B_pi = self.B_pi[sort_s_id]
        self.B_pi = self.B_pi[:, sort_s_id]
        self.S = self.S[sort_s_id]

        """ debug efe agent reward """
        self.r = self.r[sort_a_id]
        self.r = self.r[:, sort_s_id]
        self.ekl = self.ekl[sort_a_id]
        self.ekl = self.ekl[:, sort_s_id]
        self.ent = self.ent[sort_s_id]

        self.sort_a_id = sort_a_id
        self.sort_s_id = sort_s_id

    def plot_A(self, ax1, ax2, annot=True):
        df_A_mu = pd.DataFrame(self.A_mu, columns=self.A_fields)
        df_A_sd = pd.DataFrame(self.A_sd, columns=self.A_fields)

        sns.heatmap(df_A_mu.round(3), annot=annot, cbar=False, ax=ax1)
        sns.heatmap(df_A_sd.round(3), annot=annot, cbar=False, ax=ax2)

        ax1.set_xlabel("obs mean")
        ax1.set_ylabel("state")
        ax1.set_xticklabels(
            ax1.get_xticklabels(), rotation=45, horizontalalignment="right"
        )

        ax2.set_xlabel("obs std")
        ax2.set_ylabel("state")
        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=45, horizontalalignment="right"
        )

        plt.tight_layout()
        return ax1, ax2

    def plot_B(self, axes, annot=False):
        act_dim = self.B.shape[0]
        for i, x in enumerate(axes.flat):
            sns.heatmap(
                self.B[i], cmap="Greys", 
                annot=annot, cbar=False, ax=x
            )
            x.set_xlabel("next state")
            x.set_ylabel("state")
            x.set_title(f"act {i}")
            if (i + 1) == act_dim:
                break
        return axes
    
    def plot_B_pi(self, ax, annot=False, cbar=False):
        sns.heatmap(
            self.B_pi.round(3), cmap="Greys", annot=annot, 
            cbar=cbar, ax=ax
        )
        ax.set_xlabel("next state")
        ax.set_ylabel("state")
        return ax

    def plot_C(self, ax):
        ax.bar(np.arange(len(self.C)), self.C)
        ax.set_xlabel("state")
        ax.set_ylabel("pdf (C)")
        return ax

    def plot_D(self, ax):
        ax.bar(np.arange(len(self.D)), self.D)
        ax.set_xlabel("state")
        ax.set_ylabel("pdf (D)")
        return ax

    def plot_F(self, ax1, ax2, annot=True):
        sns.heatmap(
            self.F_mu.round(3), cmap="vlag", annot=annot, cbar=False, ax=ax1
        )
        sns.heatmap(
            self.F_sd.round(3), annot=annot, cbar=False, ax=ax2
        )

        ax1.set_xlabel("ctl mean")
        ax1.set_ylabel("state")

        ax2.set_xlabel("ctl std")
        ax2.set_ylabel("state")

        plt.tight_layout()
        return ax1, ax2

    def plot_tau(self, ax):
        ax.bar(np.arange(len(self.tau)), self.tau)
        ax.set_xlabel("plan horizon")
        ax.set_ylabel("pdf")
        return ax

    def plot_S(self, ax):
        ax.bar(np.arange(len(self.S)), self.S)
        ax.set_xlabel("state")
        ax.set_ylabel("stationary pdf")
        return ax

    def plot_pi(self, ax, annot=True, cbar=False):
        sns.heatmap(
            self.pi.T.round(3), cmap="Greys", annot=annot, cbar=cbar, ax=ax
        )
        ax.set_xlabel("action")
        ax.set_ylabel("state")
        return ax
    
    def plot_r(self, ax):
        ax.bar(np.arange(len(self.ent)), self.ent)
        ax.set_xlabel("state")
        ax.set_ylabel("obs entropy")
        return ax

    def plot_episode(self, sim_data, obs_keys, annot=False, cbar=False, figsize=(10, 8)):
        b = sim_data["ego"]["b"].T
        a = sim_data["ego"]["a"].T
        v = sim_data["ego"]["v"].T
        obs = sim_data["ego"]["obs"]
        ctl = sim_data["ego"]["ctl"]
        timestamp = np.arange(a.shape[1])

        df_obs = pd.DataFrame(obs, columns=self.A_fields)

        if self.sort_a_id is not None:
            a = a[self.sort_a_id]
            v = v[self.sort_a_id]
        if self.sort_s_id is not None:
            b = b[self.sort_s_id]

        fig, ax = plt.subplots(4 + len(obs_keys), 1, figsize=figsize, sharex=True)
        sns.heatmap(b, cmap="rocket", annot=annot, cbar=cbar, ax=ax[0])
        sns.heatmap(a, cmap="rocket", annot=annot, cbar=cbar, ax=ax[1])
        sns.heatmap(v, cmap="rocket", annot=annot, cbar=cbar, ax=ax[2])
        ax[3].plot(timestamp, ctl)

        for i in range(len(obs_keys)):
            ax[i + 3].plot(timestamp, df_obs[obs_keys[i]])
            ax[i + 3].set_ylabel(obs_keys[i])

        ax[0].set_ylabel("belief")
        ax[1].set_ylabel("action")
        ax[2].set_ylabel("a value")
        ax[3].set_ylabel("control")
        ax[-1].set_xlabel("time (0.1.s)")
        ax[-1].set_xticklabels(
            ax[-1].get_xticklabels(), rotation=45, horizontalalignment="right"
        )

        plt.tight_layout()
        return fig, ax

def plot_active_inference_parameters(theta, figsize=(15, 12), n_round=2, cmap="viridis"):
    """
    Args:
        theta (dict): active inference parameters dict.
        figsize (tuple, optional): figure size. Defaults to (15, 12).
        n_round (int, optional): heatmap display rounding digits. Defaults to 2.
        cmap (str, optional): color map. Defaults to "viridis".

    Returns:
        _type_: _description_
    """
    theta = {k: v.round(n_round) for k, v in theta.items()}
    
    num_cols = 3
    num_rows = 2 + np.ceil(len(theta["B"]) / 3).astype(int)
    
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    sns.heatmap(theta["A_mu"], annot=True, cbar=False, cmap=cmap, ax=ax[0, 0])
    sns.heatmap(theta["A_sd"], annot=True, cbar=False, cmap=cmap, ax=ax[0, 1])
    sns.heatmap(theta["C"].reshape(-1, 1), annot=True, cbar=False, cmap=cmap, ax=ax[0, 2])
    sns.heatmap(theta["F_mu"], annot=True, cbar=False, cmap=cmap, ax=ax[1, 0])
    sns.heatmap(theta["F_sd"], annot=True, cbar=False, cmap=cmap, ax=ax[1, 1])
    sns.heatmap(theta["D"].reshape(-1, 1), annot=True, cbar=False, cmap=cmap, ax=ax[1, 2])
    
    ax[0, 0].set_xlabel("A_mu")
    ax[0, 0].set_ylabel("state")
    ax[0, 1].set_xlabel("A_sd")
    ax[0, 1].set_ylabel("state")
    ax[0, 2].set_xlabel("C (tau = {:.2f})".format(theta["tau"] + 1))
    ax[0, 2].set_ylabel("state")
    ax[1, 0].set_xlabel("F_mu")
    ax[1, 0].set_ylabel("act")
    ax[1, 1].set_xlabel("F_sd")
    ax[1, 1].set_ylabel("act")
    ax[1, 2].set_xlabel("D")
    ax[1, 2].set_ylabel("state")
    
    counter = 0
    for i in range(2, num_rows):
        for j in range(num_cols):
            sns.heatmap(
                theta["B"][counter], annot=True, cbar=False, cmap=cmap, ax=ax[i, j]
            )
            ax[i, j].set_xlabel("next state")
            ax[i, j].set_ylabel("state")
            ax[i, j].set_title(f"B[{counter}]")
            
            counter += 1
            if counter + 1 > len(theta["B"]):
                break
        if counter + 1 > len(theta["B"]):
                break
        
    plt.tight_layout()
    return fig, ax