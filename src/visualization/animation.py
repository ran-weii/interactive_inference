import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.visualization.map_vis import plot_ways

def animate(map_data, states, agents, title="", annot=False):
    fig = plt.figure()
    ax = plt.axes()
    av = AnimationVisualizer(
        ax, map_data, states, agents, title=title, annot=annot
    )
    ani = animation.FuncAnimation(
        fig, av.update, frames=len(states),
        interval=20, blit=True, init_func=av.initfun,repeat=False)
    return ani

def save_animation(ani, path):
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate=1800)
    ani.save(path, writer)

def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center

def polygon_xy_from_motionstate(x, y, psi, width, length):
    lowleft = (x - length / 2., y - width / 2.)
    lowright = (x + length / 2., y - width / 2.)
    upright = (x + length / 2., y + width / 2.)
    upleft = (x - length / 2., y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([x, y]), yaw=psi)

class AnimationVisualizer:
    def __init__(self, ax, map_data, states, track_data, title="", annot=False):
        self.ax = ax
        self.map_data = map_data
        self.title = title
        self.annot = annot

        ego = track_data["ego"]
        agents = track_data["agents"]
        
        x_ego = states[:, 0].reshape(-1, 1)
        y_ego = states[:, 1].reshape(-1, 1)
        psi_ego = states[:, 4].reshape(-1, 1)
        l_ego = ego[:, 5].reshape(-1, 1)
        w_ego = ego[:, 6].reshape(-1, 1)
        id_ego = ego[:, 9].reshape(-1, 1)

        x_ego_data = ego[:, 0].reshape(-1, 1)
        y_ego_data = ego[:, 1].reshape(-1, 1)
        psi_ego_data = ego[:, 4].reshape(-1, 1)
        l_ego_data = ego[:, 5].reshape(-1, 1)
        w_ego_data = ego[:, 6].reshape(-1, 1)
        id_ego_data = ego[:, 9].reshape(-1, 1)

        x_agents = agents[:, :, 0]
        y_agents = agents[:, :, 1]
        psi_agents = agents[:, :, 4]
        l_agents = agents[:, :, 5]
        w_agents = agents[:, :, 6]
        id_agents = agents[:, :, 9]

        self.x = np.concatenate([x_ego, x_ego_data, x_agents], axis=1)
        self.y = np.concatenate([y_ego, y_ego_data, y_agents], axis=1)
        self.psi = np.concatenate([psi_ego, psi_ego_data, psi_agents], axis=1)
        self.l = np.concatenate([l_ego, l_ego_data, l_agents], axis=1)
        self.w = np.concatenate([w_ego, w_ego_data, w_agents], axis=1)
        self.id_ = np.concatenate([id_ego, id_ego_data, id_agents], axis=1)
        self.num_agents = agents.shape[1]
    
    @property
    def assets(self):
        return self._carrects + self._cartexts

    def initfun(self):
        plot_ways(self.map_data, self.ax, annot=False)
        self.ax.set_title(self.title)

        carrects = []
        cartexts = []
        for i in range(self.num_agents + 1):
            color = "tab:blue"
            alpha = 1
            if i == 0:
                color = "tab:orange"
            elif i == 1:
                color = "tab:green"
                alpha = 0.4
            else:
                pass
            rectpts = np.array([(-1.,-1.), (1.,-1), (1.,1.), (-1.,1.)])
            rect = matplotlib.patches.Polygon(
                rectpts, closed=True, color=color, zorder=20, ec='k', alpha=alpha
            )
            self.ax.add_patch(rect)
            carrects.append(rect) 
            cartexts.append(self.ax.text(0, 0, ""))
        self._carrects = carrects
        self._cartexts = cartexts
        return self.assets

    def update(self, frame):
        self.ax.title.set_text(f"{self.title} (frame={frame})")
        for i, (carrect, cartext) in enumerate(zip(self._carrects, self._cartexts)):
            x = self.x[frame][i]
            y = self.y[frame][i]
            psi = self.psi[frame][i]
            l = self.l[frame][i]
            w = self.w[frame][i]
            id_ = self.id_[frame][i]
            if np.isnan(x):
                x, y = -1, -1
            rectpts = polygon_xy_from_motionstate(x, y, psi, w, l)
            carrect.set_xy(rectpts)

            if self.annot:
                cartext.set_position((x, y))
                cartext.set_text(f"{id_:.0f}")
        return self.assets