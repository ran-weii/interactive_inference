import torch

def eval_actions_epoch(agent, loader, sample_method="ace", num_samples=30):
    """ Evaluate agent action selection
    
    Args:
        agent (Agent): agent object
        loader (DataLoader): torch dataloader object
        sample_method (str, optional): sample method. 
            chioces=["bme", "ace", "acm"]. Default="ace
        num_samples (int, optional): number of samples. Default=30
    """
    agent.eval()
    
    u_batch, u_sample_batch = [], []
    for i, batch in enumerate(loader):
        o, u, mask = batch
        agent.reset()
        with torch.no_grad():
            u_sample, _ = agent.choose_action_batch(
                o, u, sample_method=sample_method, num_samples=num_samples
            )
            
            nan_mask = mask.clone()
            nan_mask[mask == 0] = torch.nan
            u_mask = u * nan_mask.unsqueeze(-1)
            u_sample_mask = u_sample * nan_mask.unsqueeze(0).unsqueeze(-1)
            u_batch.append(u_mask)
            u_sample_batch.append(u_sample_mask)
    
    u_batch = torch.cat(u_batch, dim=1).data.numpy()
    u_sample_batch = torch.cat(u_sample_batch, dim=1).data.numpy()
    return u_batch, u_sample_batch

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
    """
    agent.eval()
    
    with torch.no_grad():
        u_sample, _ = agent.choose_action_batch(
            obs.unsqueeze(-2), ctl.unsqueeze(-2), 
            sample_method=sample_method, num_samples=num_samples
        )
    return u_sample.squeeze(-2)

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
        o_sample, _, _, _ =model.predict(
            obs.unsqueeze(-2), ctl.unsqueeze(-2), prior=False, inference=True, 
            sample_method=sample_method, num_samples=num_samples
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