import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.visualization.map_vis import plot_ways

def animate(map_data, states, agents, **kwargs):
    fps = kwargs.get('fps', 15)
    bitrate = kwargs.get('bitrate', 1800)
    enc = kwargs.get('encoder', 'ffmpeg')
    iv = kwargs.get('interval', 20)
    blit = kwargs.get('blit', True)

    fig = plt.figure()
    ax = plt.axes()
    av = AnimationVisualizer(ax, map_data, states, agents)
    ani = animation.FuncAnimation(
        fig, av.update, frames=len(states),
        interval=iv, blit=blit, init_func=av.initfun,repeat=False)

    Writer = animation.writers[enc]
    writer = Writer(fps=fps, bitrate=bitrate)
    ani.save('/Users/rw422/Documents/render_ani.mp4', writer)

def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center

def polygon_xy_from_motionstate(x, y, psi, width, length):
    lowleft = (x - length / 2., y - width / 2.)
    lowright = (x + length / 2., y - width / 2.)
    upright = (x + length / 2., y + width / 2.)
    upleft = (x - length / 2., y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([x, y]), yaw=psi)

class AnimationVisualizer:
    def __init__(self, ax, map_data, states, track_data):
        self.ax = ax
        self.map_data = map_data

        ego = track_data["ego"]
        agents = track_data["agents"]
        
        x_ego = states[:, 0].reshape(-1, 1)
        y_ego = states[:, 1].reshape(-1, 1)
        psi_ego = states[:, 4].reshape(-1, 1)
        l_ego = ego[:, 5].reshape(-1, 1)
        w_ego = ego[:, 6].reshape(-1, 1)

        x_agents = agents[:, :, 0]
        y_agents = agents[:, :, 1]
        psi_agents = agents[:, :, 4]
        l_agents = agents[:, :, 5]
        w_agents = agents[:, :, 6]

        self.x = np.concatenate([x_ego, x_agents], axis=1)
        self.y = np.concatenate([y_ego, y_agents], axis=1)
        self.psi = np.concatenate([psi_ego, psi_agents], axis=1)
        self.l = np.concatenate([l_ego, l_agents], axis=1)
        self.w = np.concatenate([w_ego, w_agents], axis=1)
        self.num_agents = agents.shape[1]
    
    @property
    def assets(self):
        return self._carrects

    def initfun(self):
        plot_ways(self.map_data, self.ax, annot=False)
        
        carrects = []
        for i in range(self.num_agents + 1):
            color = "tab:orange" if i == 0 else "tab:blue"
            rectpts = np.array([(-1.,-1.), (1.,-1), (1.,1.), (-1.,1.)])
            rect = matplotlib.patches.Polygon(rectpts, closed=True, color=color, zorder=20, ec='k')
            self.ax.add_patch(rect)
            carrects.append(rect) 
        self._carrects = carrects
        return self.assets

    def update(self, frame):
        for i, carrect in enumerate(self._carrects):
            if not np.isnan(self.x[frame][i]):
                x = self.x[frame][i]
                y = self.y[frame][i]
                psi = self.psi[frame][i]
                l = self.l[frame][i]
                w = self.w[frame][i]
                rectpts = polygon_xy_from_motionstate(x, y, psi, w, l)
                carrect.set_xy(rectpts)
        return self.assets