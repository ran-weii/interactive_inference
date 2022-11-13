import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.agents.vin_agent import VINAgent
from src.explanation.explainer import VINExplainer
from src.explanation.imagination import simulate_imagination

def load_agent():
    exp_name = "11-08-2022 19-52-30"
    exp_path = os.path.join("../exp", "agents", "vin", exp_name)

    # load config
    with open(os.path.join(exp_path, "args.json"), "rb") as f:
        config = json.load(f)

    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))
    state_dict = state_dict if "model_state_dict" not in state_dict.keys() else state_dict["model_state_dict"]
    state_dict = {k.replace("agent.", ""): v.cpu() for (k, v) in state_dict.items() if "agent." in k}

    # load agent
    feature_set = config["feature_set"]
    action_set = config["action_set"]

    obs_dim = len(feature_set)
    ctl_dim = len(action_set)

    agent = VINAgent(
        config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
        config["horizon"], alpha=config["alpha"], beta=config["beta"], obs_model=config["obs_model"],
        obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], rwd=config["rwd"]
    )

    agent.load_state_dict(state_dict, strict=False)
    agent.eval()

    for p in agent.parameters():
        p.requires_grad = False
    return agent, feature_set, action_set

def test_vin_explainer():
    agent, feature_set, action_set = load_agent()

    # init explainer
    by_feature = "lv_inv_tau"
    explainer = VINExplainer(agent, feature_set, action_set)
    explainer.sort_state_action(by_feature)

    explainer.plot_action_component_pdfs()
    explainer.plot_observation_scatter_matrix(color_by="stationary_dist")
    explainer.plot_transition_matrix()
    explainer.plot_target_dist(sort_by="lv_inv_tau")
    explainer.plot_policy(sort_by="lv_inv_tau")
    explainer.plot_controlled_transition_matrix()
    explainer.plot_controlled_stationary_dist()
    plt.show()
    
    # fig = explainer.plot_observation_scatter_3d()
    # fig.show()

def test_imagination():
    agent, feature_set, action_set = load_agent()
    
    s0 = 0
    max_steps = 1000
    data = simulate_imagination(agent, s0, max_steps)

    # plot trajectory
    fig, ax = plt.subplots(4, 2, figsize=(12, 8), sharex=True)

    # plot observables
    for i in range(3):
        ax[i, 0].plot(data["o"][:, i])
        ax[i, 0].set_ylabel(feature_set[i])

    ax[-1, 0].plot(data["u"][:, 0])
    ax[-1, 0].set_ylabel("action")

    # plot latent
    sns.heatmap(data["pi_s"].T, cbar=False, ax=ax[0, 1])
    sns.heatmap(data["b"].T, cbar=False, ax=ax[1, 1])
    sns.heatmap(data["pi"].T, cbar=False, ax=ax[2, 1])

    ax[0, 1].set_ylabel("latent state")
    ax[1, 1].set_ylabel("agent belief")
    ax[2, 1].set_ylabel("agent policy")

    plt.tight_layout()

    # plot observation-control association
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    for i in range(3):
        ax[i].plot(data["o"][:-1, i], data["u"][:, 0], ".")
        ax[i].set_xlabel(feature_set[i])
        ax[i].set_ylabel("dds")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(0)
    # test_vin_explainer()
    # test_imagination()