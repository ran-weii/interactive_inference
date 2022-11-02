import os
import time
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch
from src.visualization.utils import plot_history

class SaveCallback:
    def __init__(self, arglist, cp_history=None):
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist.exp_path, "agents")
        agent_path = os.path.join(exp_path, arglist.agent)
        save_path = os.path.join(agent_path, date_time)
        model_path = os.path.join(save_path, "model")
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(agent_path):
            os.mkdir(agent_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(arglist), f)

        self.save_path = save_path
        self.model_path = model_path
        self.cp_history = cp_history
        self.cp_every = arglist.cp_every

        self.num_test_eps = 0
        self.iter = 0

    def __call__(self, model, history):
        self.iter += 1
        if self.iter % self.cp_every != 0:
            return
        
        # save history
        df_history = pd.DataFrame(history)
        self.save_history(model, df_history)
        
        # save model
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state_dict, os.path.join(self.model_path, f"model_{self.iter}.pt"))
        print(f"\ncheckpoint saved at: {self.save_path}\n")
    
    def save_history(self, model, df_history):
        if self.cp_history is not None:
            df_history["epoch"] += self.cp_history["epoch"].values[-1] + 1
            df_history["time"] += self.cp_history["time"].values[-1]
            df_history = pd.concat([self.cp_history, df_history], axis=0)
        df_history.to_csv(os.path.join(self.save_path, "history.csv"), index=False)
        
        # save history plot
        fig_history, _ = plot_history(df_history, model.loss_keys)
        fig_history.savefig(os.path.join(self.save_path, "history.png"), dpi=100)

        plt.clf()
        plt.close()

    def save_model(self, model):
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state_dict, os.path.join(self.save_path, "model.pt"))


def train(model, train_loader, test_loader, epochs, callback=None, verbose=1):
    history = []
    start_time = time.time()
    for e in range(epochs):
        train_stats = model.run_epoch(train_loader, train=True)
        test_stats = model.run_epoch(test_loader, train=False)
        
        tnow = time.time() - start_time
        train_stats.update({"epoch": e, "time": tnow})
        test_stats.update({"epoch": e, "time": tnow})
        history.append(train_stats)
        history.append(test_stats)

        if (e + 1) % verbose == 0:
            s = model.stdout(train_stats, test_stats)
            print("e: {}/{}, {}, t: {:.2f}".format(e + 1, epochs, s, tnow))

        if callback is not None:
            callback(model, history)

    df_history = pd.DataFrame(history)
    return model, df_history