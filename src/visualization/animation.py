import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.simulation.simulator import STATE_KEYS

from src.visualization.map_vis import plot_ways

def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center

def polygon_xy_from_motionstate(x, y, psi, width, length):
    lowleft = (x - length / 2., y - width / 2.)
    lowright = (x + length / 2., y - width / 2.)
    upright = (x + length / 2., y + width / 2.)
    upleft = (x - length / 2., y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([x, y]), yaw=psi)

def animate(map_data, sim_data, state_keys=STATE_KEYS, title="", annot=False, axis_on=True):
    """
    Args:
        map_data (MapReader): map reader object.
        sim_data (dict): simulator ._data object.
        state_keys (dict): simulator state dict.
        title (str): animation title.
        annot (bool): whether to annotate agents.
        axis_on (bool): whether to turn on axis.
    """
    fig = plt.figure()
    ax = plt.axes()
    av = AnimationVisualizer(
        ax, map_data, sim_data, state_keys, title=title, annot=annot, axis_on=axis_on
    )
    ani = animation.FuncAnimation(
        fig, av.update, frames=len(sim_data),
        interval=20, blit=True, init_func=av.initfun,repeat=False)
    return ani

def save_animation(ani, path):
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate=1800)
    ani.save(path, writer)


class AnimationVisualizer:
    """ Animation visualizer with lead vehicle and lidar animation 
    
        Lead vehicle is drawn with a red dot. Lidar beams are drawn with greed line
    """
    def __init__(self, ax, map_data, sim_data, state_keys=STATE_KEYS, title="", annot=False, axis_on=True):
        self.ax = ax
        self.title = title
        self.annot = annot
        self.axis_on = axis_on

        self.map_data = map_data
        self.sim_data = sim_data
        
        self.x_id = state_keys.index("x")
        self.y_id = state_keys.index("y")
        self.psi_id = state_keys.index("psi_rad")
        self.l_id = state_keys.index("length")
        self.w_id = state_keys.index("width")
        self.id_id = state_keys.index("track_id")
        
        self.max_agents = max([d["sim_state"]["agent_states"].shape[0] for d in self.sim_data])
        self.sensor_names = list(sim_data[0]["sensor_obs"].keys())
    
    @property
    def assets(self):
        return self._carrects + self._cartexts + self._sensor_lines

    def initfun(self):
        plot_ways(self.map_data, self.ax, annot=False)
        self.ax.set_title(self.title)
        
        # init assets
        carrects = []
        cartexts = []
        sensor_lines = []

        # init car rects
        for i in range(self.max_agents + 2):
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

        # init lidar lines
        if "LidarSensor" in self.sensor_names:
            for i in range(len(self.sim_data[0]["sensor_pos"]["LidarSensor"])):
                line, = self.ax.plot([0], [0], "g-")
                sensor_lines.append(line)
        
        # init lv line
        if "LeadVehicleSensor" in self.sensor_names:
            line, = self.ax.plot([0], [0], "ro", zorder=30)
            sensor_lines.append(line)

        self._carrects = carrects
        self._cartexts = cartexts
        self._sensor_lines = sensor_lines
        return self.assets

    def update(self, frame):
        ego_state = self.sim_data[frame]["sim_state"]["ego_state"]
        ego_true_state = self.sim_data[frame]["sim_state"]["ego_true_state"]
        agent_states = self.sim_data[frame]["sim_state"]["agent_states"]

        self.ax.title.set_text(f"{self.title} (frame={frame})")
        if not self.axis_on:
            self.ax.title.set_text(f"")
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)

        x_ego = ego_state[self.x_id]
        y_ego = ego_state[self.y_id]
        psi_ego = ego_state[self.psi_id]
        l_ego = ego_state[self.l_id]
        w_ego = ego_state[self.w_id]
        id_ego = ego_state[self.id_id]

        x_ego_true = ego_true_state[self.x_id]
        y_ego_true = ego_true_state[self.y_id]
        psi_ego_true = ego_true_state[self.psi_id]
        l_ego_true = ego_true_state[self.l_id]
        w_ego_true = ego_true_state[self.w_id]
        id_ego_true = ego_true_state[self.id_id]

        x_agents = agent_states[:, self.x_id]
        y_agents = agent_states[:, self.y_id]
        psi_agents = agent_states[:, self.psi_id]
        l_agents = agent_states[:, self.l_id]
        w_agents = agent_states[:, self.w_id]
        id_agents = agent_states[:, self.id_id]

        x = np.hstack([[x_ego, x_ego_true], x_agents])
        y = np.hstack([[y_ego, y_ego_true], y_agents])
        psi = np.hstack([[psi_ego, psi_ego_true], psi_agents])
        l = np.hstack([[l_ego, l_ego_true], l_agents])
        w = np.hstack([[w_ego, w_ego_true], w_agents])
        id_ = np.hstack([[id_ego, id_ego_true], id_agents])
        
        # plot car rects
        num_agents = len(x)
        for i, (carrect, cartext) in enumerate(zip(self._carrects, self._cartexts)):
            if (i + 1) <= num_agents:
                rectpts = polygon_xy_from_motionstate(x[i], y[i], psi[i], w[i], l[i])
                if self.annot:
                    cartext.set_position((x[i], y[i]))
                    cartext.set_text(f"{id_[i]:.0f}")
            else:
                rectpts = polygon_xy_from_motionstate(-1, -1, -1, 0, 0)
                if self.annot:
                    cartext.set_position((-1, -1))
            carrect.set_xy(rectpts)
        
        # plot lidar lines 
        if "LidarSensor" in self.sensor_names:
            lidar_pos = self.sim_data[frame]["sensor_pos"]["LidarSensor"]
            for i in range(len(lidar_pos)):
                self._sensor_lines[i].set_data(
                    [x_ego, lidar_pos[i, 0]], [y_ego, lidar_pos[i, 1]]
                )

        # plot lv line 
        if "LeadVehicleSensor" in self.sensor_names:
            lv_pos = self.sim_data[frame]["sensor_pos"]["LeadVehicleSensor"]
            self._sensor_lines[-1].set_data([lv_pos[0]], [lv_pos[1]])
        return self.assets