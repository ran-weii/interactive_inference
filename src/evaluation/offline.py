import torch

def pad_batches(batches):
    max_len = max([b.shape[-3] for b in batches])
    pad_batches = []
    for b in batches:
        pad_len = max_len - b.shape[-3]
        padding = torch.zeros(list(b.shape[:-3]) + list(b.shape[-2:])).unsqueeze(-3)
        padding = padding.repeat_interleave(pad_len, dim=-3)
        pad_batches.append(torch.cat([b, padding], dim=-3))
        
    pad_batches = torch.cat(pad_batches, dim=-2)
    return pad_batches

def eval_actions_batch(agent, loader, test_posterior=True, sample_method="ace", num_samples=30):
    """ Evaluate agent action selection
    
    Args:
        agent (Agent): agent object
        loader (DataLoader): torch dataloader object
        test_posterior (bool): whether to test posterior. Default=True
        sample_method (str, optional): sample method. 
            chioces=["bme", "ace", "acm"]. Default="ace
        num_samples (int, optional): number of samples. Default=30

    Returns:
        u_true_pad (torch.tensor): nan padded true actions. size=[num_samples, max_T, loader_size, ctl_dim]
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
            if hasattr(agent, "encoder"):
                if test_posterior:
                    z = agent.get_posterior_dist(o, u, mask).mean
                else:
                    z = agent.get_prior_dist().mean.repeat_interleave(o.shape[-2], dim=0)
            else:
                z = None
                
            u_sample, _ = agent.choose_action_batch(
                o, u, z=z, sample_method=sample_method, num_samples=num_samples
            )
            
            nan_mask = mask.clone()
            nan_mask[mask == 0] = torch.nan
            u_mask = u * nan_mask.unsqueeze(-1)
            u_mask = u_mask.unsqueeze(0).repeat_interleave(num_samples, dim=0)
            u_sample_mask = u_sample * nan_mask.unsqueeze(0).unsqueeze(-1)
            
            u_true_batch.append(u_mask)
            u_sample_batch.append(u_sample_mask)

    u_true = pad_batches(u_true_batch)
    u_sample = pad_batches(u_sample_batch)
    return u_true, u_sample

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