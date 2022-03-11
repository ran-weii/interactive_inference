import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def get_active_inference_parameters(agent):
    """ 
    Args:
        agent (ActiveInference): active inference agent object

    Returns:
        theta (dict): active inference parameter dict
    """
    A_mu = agent.obs_model.mean().data.squeeze(0)
    A_sd = torch.sqrt(agent.obs_model.variance()).data.squeeze(0)
    B = torch.softmax(agent.hmm.B, dim=-1).data.squeeze(0)
    C = torch.softmax(agent.C, dim=-1).data.squeeze(0)
    D = torch.softmax(agent.hmm.D, dim=-1).data.squeeze(0)
    F_mu = agent.ctl_model.mean().data.squeeze(0)
    F_sd = torch.sqrt(agent.ctl_model.variance()).data.squeeze(0)
    
    theta = {
        "A_mu": A_mu.numpy(),
        "A_sd": A_sd.numpy(),
        "B": B.numpy(),
        "C": C.numpy(),
        "D": D.numpy(),
        "F_mu": F_mu.numpy(),
        "F_sd": F_sd.numpy()
    }
    return theta

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
    ax[0, 2].set_xlabel("C")
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