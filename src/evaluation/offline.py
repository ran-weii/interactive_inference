import torch

def eval_actions_batch(agent, loader, sample_method="ace", num_samples=30):
    """ Evaluate agent action selection
    
    Args:
        agent (Agent): agent object
        loader (DataLoader): torch dataloader object
        sample_method (str, optional): sample method. 
            chioces=["bme", "ace", "acm"]. Default="ace
        num_samples (int, optional): number of samples. Default=30

    Returns:
        u_true_pad (torch.tensor): nan padded true actions. size=[max_T, loader_size, ctl_dim]
        u_sample_pad (torch.tensor): nan padded sample actions. size=[num_samples, max_T, loader_size, ctl_dim]
    """
    agent.eval()
    
    u_true_batch, u_sample_batch = [], []
    for i, batch in enumerate(loader):
        pad_batch, mask = batch
        o = pad_batch["obs"]
        u = pad_batch["act"]

        agent.reset()
        with torch.no_grad():
            u_sample, _ = agent.choose_action_batch(
                o, u, sample_method=sample_method, num_samples=num_samples
            )
            
            nan_mask = mask.clone()
            nan_mask[mask == 0] = torch.nan
            u_mask = u * nan_mask.unsqueeze(-1)
            u_sample_mask = u_sample * nan_mask.unsqueeze(0).unsqueeze(-1)

            u_true_batch.append(u_mask)
            u_sample_batch.append(u_sample_mask)
    
    # pad batches
    max_len = max([len(u) for u in u_true_batch])
    u_true_pad = []
    u_sample_pad = []
    for (u_true, u_sample) in zip(u_true_batch, u_sample_batch):
        pad_len = max_len - len(u_true)
        padding1 = torch.nan * torch.ones(list(u_true.shape)[1:]).unsqueeze(0)
        padding2 = padding1.clone().unsqueeze(0).repeat_interleave(len(u_sample), 0)

        u_true = torch.cat([u_true, padding1.repeat_interleave(pad_len, -3)], dim=-3)
        u_sample = torch.cat([u_sample, padding2.repeat_interleave(pad_len, -3)], dim=-3)
        
        u_true_pad.append(u_true)
        u_sample_pad.append(u_sample)
    
    u_true_pad = torch.cat(u_true_pad, dim=-2).data.numpy()
    u_sample_pad = torch.cat(u_sample_pad, dim=-2).data.numpy()
    return u_true_pad, u_sample_pad

def eval_actions_episode(agent, obs, ctl, sample_method="ace", num_samples=30):
    """ Evaluate agent action selection for an episode
    
    Args:
        agent (Agent): agent object
        obs (torch.tensor): observation sequence. size=[T, obs_dim]
        ctl (torch.tensor): action sequence. size=[T, ctl_dim]
        sample_method (str, optional): sample method. 
            chioces=["bme", "ace", "acm"]. Default="ace
        num_samples (int, optional): number of samples. Default=30
    
    Returns:
        u_sample (torch.tensor): sampled action sequence. size=[num_samples, T, ctl_dim]
        act_loss (torch.tensor): action loss. size=[batch_size]
    """
    agent.eval()
    
    with torch.no_grad():
        u_sample, _, hidden = agent.choose_action_batch(
            obs.unsqueeze(-2), ctl.unsqueeze(-2), 
            sample_method=sample_method, num_samples=num_samples, return_hidden=True
        )
        act_loss, _ = agent.act_loss(
            obs.unsqueeze(-2), ctl.unsqueeze(-2), 
            torch.ones(len(obs), 1), hidden
        )
    return u_sample.squeeze(-2), act_loss

def eval_dynamics_episode(model, obs, ctl, sample_method="ace", num_samples=30):
    """ Evaluate dynamics model prediction for an episode
    
    Args:
        model (nn.Module): dynamics model object with predict method.
        obs (torch.tensor): observation sequence. size=[T, obs_dim]
        ctl (torch.tensor): action sequence. size=[T, ctl_dim]
        sample_method (str, optional): sample method. 
            chioces=["bme", "ace", "acm"]. Default="ace
        num_samples (int, optional): number of samples. Default=30
    
    Returns:
        o_sample (torch.tensor): sampled action sequence. size=[num_samples, T, ctl_dim]
    """
    model.eval()

    with torch.no_grad():
        o_sample, _ =model.predict(
            obs.unsqueeze(-2), ctl.unsqueeze(-2), sample_method=sample_method, num_samples=num_samples
        )
        o_sample = o_sample.squeeze(-2)
    return o_sample

def sample_action_components(model, num_samples=50):
    """ Sample from each component in the control modle 
    
    Args:
        model (nn.Module): contorl model.
        num_samples (int, optional): number of samples to draw from each component. Default=50
    
    Returns:
        u_sample (torch.tensor): sampled actions. size=[num_samples, act_dim, ctl_dim]
    """
    with torch.no_grad():
        u_samples = model.sample((num_samples,)).squeeze(-3).data
    return u_samples