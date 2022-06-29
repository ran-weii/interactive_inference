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
            u_sample = agent.choose_action_batch(
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