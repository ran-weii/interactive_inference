import time
import pprint
import numpy as np
from src.evaluation.online import eval_episode

class Logger():
    def __init__(self):
        self.epoch_dict = dict()
        self.history = []
        self.test_episodes = []
    
    def push(self, stats_dict):
        for key, val in stats_dict.items():
            if not (key in self.epoch_dict.keys()):
                self.epoch_dict[key] = []
            self.epoch_dict[key].append(val)

    def log(self):
        stats = dict()
        for key, val in self.epoch_dict.items():
            if isinstance(val[0], np.ndarray) or len(val) > 1:
                vals = np.stack(val)
                stats[key + "_avg"] = np.mean(vals)
                stats[key + "_std"] = np.std(vals)
                stats[key + "_min"] = np.min(vals)
                stats[key + "_max"] = np.max(vals)
            else:
                stats[key] = val[-1]

        pprint.pprint({k: np.round(v, 4) for k, v, in stats.items()})
        self.history.append(stats)

        # erase epoch stats
        self.epoch_dict = dict()
    
    def log_test_episode(self, sim_states, track_data):
        self.test_episodes.append({"sim_states": sim_states, "track_data": track_data})


def train(
    env, model, epochs, steps_per_epoch=1000, 
    update_after=3000, update_every=50, log_test_every=10
    ):
    """
    Args:
        env (Simulator): simulator environment
        model (Model): trainer model
        epochs (int): training epochs
        steps_per_epoch (int, optional): number of environment steps per epoch. Default=1000
        update_after (int, optional): initial burn-in steps before training. Default=3000
        update_every (int, optional): number of environment steps between training. Default=50
        log_test_every (int, optional): epochs between logging test episode. Default=10
    """
    logger = Logger()

    total_steps = epochs * steps_per_epoch
    start_time = time.time()
    
    model.agent.reset()
    eps_id = np.random.choice(np.arange(len(env.dataset)))
    obs, eps_return, eps_len = env.reset(eps_id), 0, 0
    for t in range(total_steps):
        ctl = model.choose_action(obs)
        next_obs, reward, done, info = env.step(ctl)
        eps_return += reward
        eps_len += 1

        # env done handeling
        done = True if info["terminated"] == True else done
        
        model.replay_buffer(obs, ctl, reward, done)
        obs = next_obs

        # end of trajectory handeling
        if done:
            model.replay_buffer.push()
            logger.push({"eps_return": eps_return/eps_len})
            logger.push({"eps_len": eps_len})
            
            model.agent.reset()
            eps_id = np.random.choice(np.arange(len(env.dataset)))
            obs, eps_return, eps_len = env.reset(eps_id), 0, 0

        # train model
        if t >= update_after and t % update_every == 0:
            train_stats = model.take_gradient_step(logger)

        # end of epoch handeling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            logger.push({"epoch": epoch})
            logger.push({"time": time.time() - start_time})
            logger.log()
            print()
            
            if t > update_after and epoch % log_test_every == 0:
                eval_eps_id = np.random.choice(np.arange(len(env.dataset)))
                sim_states, sim_acts, track_data, rewards = eval_episode(env, model.agent, eval_eps_id)
                logger.log_test_episode(sim_states, track_data)
                print(f"test mean reward: {np.mean(rewards)}\n")
    
    # final test episode
    eval_eps_id = np.random.choice(np.arange(len(env.dataset)))
    sim_states, sim_acts, track_data = eval_episode(env, model.agent, eval_eps_id)
    logger.log_test_episode(sim_states, track_data)
    return model, logger
