import numpy as np
import pandas as pd

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import CustomJS, HoverTool, Slider
from bokeh.layouts import column

def get_vector_dict(x, y, vx, vy, psi_rad, global_coor=True):
    """ Build two point vector dict centered at (x, y) with heading psi_rad
    
    Args:
        x (float): vehicle x position
        y (float): vehicle y position
        vx (float): vector x value 
        vy (float): vector y value 
        psi_rad (float): vehicle heading
        global_coor (bool, optional): vector in global coordinate

    Returns:
        out (dict): two point vector dict
    """
    tan = np.tan(psi_rad)
    
    if global_coor:
        v_norm = np.sqrt(vx**2 + vy**2)
        if vx < 0:
            v_norm *= -1    
        out = {"vx_psi": [x, x + v_norm], "vy_psi": [y, y + v_norm * tan]}
    else:
        out = {"vx_psi": [x, x + vx], "vy_psi": [y, y + vy * tan]}
    return out

def build_bokeh_sources(track_data, df_lanelet, ego_fields, agent_fields, acc_true=None, acc_pred=None):
    # build lanelet data source
    df_lanelet["color"] = [d["color"] for d in df_lanelet["vis_dict"]]
    df_lanelet["dash"] = [d["dashes"] if "dashes" in d.keys() else "solid" for d in df_lanelet["vis_dict"]]
    df_lanelet["linewidth"] = [d["linewidth"] for d in df_lanelet["vis_dict"]]

    df_lanelet["max_x"] = [max(x) for x in df_lanelet["x"]]
    df_lanelet["min_x"] = [min(x) for x in df_lanelet["x"]]
    df_lanelet["max_y"] = [max(y) for y in df_lanelet["y"]]
    df_lanelet["min_y"] = [min(y) for y in df_lanelet["y"]]

    lanelet_source = ColumnDataSource(df_lanelet)
    
    # build trajectory data sources
    ego_data = track_data["ego"]
        
    frames = []
    for i in range(len(ego_data)):
        df_ego = pd.DataFrame(ego_data[i].reshape(1, -1), columns=ego_fields)
        ego_speed_dict = get_vector_dict(
            df_ego["x"].iloc[0],
            df_ego["y"].iloc[0],
            df_ego["vx"].iloc[0],
            df_ego["vy"].iloc[0],
            df_ego["psi_rad"].iloc[0],
            global_coor=True
        )
        
        agent_data = track_data["agents"][i]
        agent_data[agent_data == -1] = np.nan
        
        df_agent = pd.DataFrame(agent_data, columns=agent_fields)
        df_agent = df_agent.loc[df_agent["dist_to_ego"] != 0]
        
        frames.append({
            "ego": ColumnDataSource(df_ego),
            "agents": ColumnDataSource(df_agent),
            "ego_speed": ColumnDataSource(data=ego_speed_dict)
        })
        
        """ TODO: come up with a better solution for adding action predictions """
        if acc_true is not None:
            acc_true_dict = get_vector_dict(
                df_ego["x"].iloc[0],
                df_ego["y"].iloc[0],
                acc_true[i, 0] * 100,
                acc_true[i, 1] * 100,
                df_ego["psi_rad"].iloc[0],
                global_coor=False
            )
            frames[i]["ego_acc_true"] = ColumnDataSource(data=acc_true_dict)
            
        if acc_pred is not None:
            acc_pred_dict = get_vector_dict(
                df_ego["x"].iloc[0],
                df_ego["y"].iloc[0],
                acc_pred[i, 0] * 100,
                acc_pred[i, 1] * 100,
                df_ego["psi_rad"].iloc[0],
                global_coor=False
            )
            frames[i]["ego_acc_pred"] = ColumnDataSource(data=acc_pred_dict)
    return frames, lanelet_source

def visualize_scene(frames, lanelet_source, title="", plot_width=900):
    max_x, min_x = lanelet_source.data["max_x"].max() + 10, lanelet_source.data["min_x"].min() - 10
    max_y, min_y = lanelet_source.data["max_y"].max() + 10, lanelet_source.data["min_y"].min() - 10
    
    agent_hover = HoverTool(
        mode="mouse",
        names=["ego", "agents"],
        tooltips=[
            ("Track id", "@track_id{00}"),
            ("Vel", "(@vx{00.0}, @vy{00.0})"),
            ("Psi", "@psi_rad{00.0}"),
            ("Lane pos", "(@lane_left_min_dist{00.0}, @lane_right_min_dist{00.0})")
        ]
    )

    f = figure(
        title=title,
        plot_width=plot_width,
        aspect_ratio=((max_x - min_x) / (max_y - min_y)),
        x_range=(min_x, max_x),
        y_range=(min_y, max_y),
        background_fill_color="grey",
        background_fill_alpha=0.5,
        tools=["pan", "wheel_zoom", agent_hover, "reset", "save"],
        active_scroll="wheel_zoom"
    )
    f.xgrid.grid_line_alpha=0.7
    f.ygrid.grid_line_alpha=0.7
    
    f.patches(
        xs="x", 
        ys="y", 
        color="color", 
        line_width="linewidth",
        line_dash="dash",
        source=lanelet_source
    )
    f.rect(
        x="x", 
        y="y", 
        width="length", 
        height="width", 
        angle="psi_rad", 
        color="red",
        legend_label="Ego",
        source=frames[0]["ego"],
        name="ego"
    )
    f.rect(
        x="x", 
        y="y", 
        width="length", 
        height="width", 
        angle="psi_rad", 
        color="Blue", 
        legend_label="Agent",
        source=frames[0]["agents"],
        name="agents"
    )
    if "ego_acc_true" in list(frames[0].keys()):
        f.line(
            x="vx_psi",
            y="vy_psi",
            line_width=3,
            line_alpha=1,
            color="red",
            source=frames[0]["ego_acc_true"],
            legend_label="Ego acc true"
        )
        f.line(
            x="vx_psi",
            y="vy_psi",
            line_width=3,
            line_alpha=1,
            color="green",
            source=frames[0]["ego_acc_pred"],
            legend_label="Ego acc pred"
        )
        
        js_string = """
        sources["ego"].data = frames[cb_obj.value]["ego"].data;
        sources["agents"].data = frames[cb_obj.value]["agents"].data;
        sources["ego_acc_true"].data = frames[cb_obj.value]["ego_acc_true"].data;
        sources["ego_acc_pred"].data = frames[cb_obj.value]["ego_acc_pred"].data;
        
        sources["ego"].change.emit();
        sources["agents"].change.emit();
        """
    else:
        f.line(
            x="vx_psi",
            y="vy_psi",
            line_width=3,
            line_alpha=1,
            color="red",
            source=frames[0]["ego_speed"],
            legend_label="Ego speed"
        )
    
        js_string = """
        sources["ego"].data = frames[cb_obj.value]["ego"].data;
        sources["agents"].data = frames[cb_obj.value]["agents"].data;
        sources["ego_speed"].data = frames[cb_obj.value]["ego_speed"].data;
        
        sources["ego"].change.emit();
        sources["agents"].change.emit();
        """
        
    slider_callback = CustomJS(
        args=dict(figure=f, sources=frames[0], frames=frames), 
        code=js_string
    )

    slider = Slider(start=0, end=len(frames), value=0, step=1, title="frame")
    slider.js_on_change("value", slider_callback)

    layout = column(f, slider)
    return layout