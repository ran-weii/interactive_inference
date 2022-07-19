import torch

class Inspector:
    def __init__(self, agent):
        self.agent = agent
        
        with torch.no_grad():
            self.efe = agent.efe.data.squeeze(0)
            self.target_dist = agent.target_dist.data.squeeze(0)
            self.reward = agent.reward.data.squeeze(0)
            self.transition = agent.transition.data.squeeze(0)
            self.value = agent.value.data.squeeze(-3)
            self.passive_dynamics = agent.passive_dynamics.data
            self.obs_mean = agent.obs_model.mean().data.squeeze(0)
            self.obs_variance = agent.obs_model.variance().data.squeeze(0)

        self._sort_matrices()

    def _sort_matrices(self):
        sort_id = torch.argsort(self.efe, descending=True)

        self.efe = self.efe[sort_id]
        self.target_dist = self.target_dist[sort_id]
        self.reward = self.reward[:, sort_id]
        self.transition = self.transition[:, sort_id, :]
        self.transition = self.transition[:, :, sort_id]
        self.value = self.value[:, :, sort_id]
        self.passive_dynamics = self.passive_dynamics[sort_id, :]
        self.passive_dynamics = self.passive_dynamics[:, sort_id]
        self.obs_mean = self.obs_mean[sort_id]
        self.obs_variance = self.obs_variance[sort_id]
        