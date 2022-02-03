import numpy as np
import pandas as pd

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import CustomJS, HoverTool, Slider
from bokeh.layouts import column

def build_bokeh_sources(track_data, df_lanelet, ego_fields, agent_fields):
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
        
        agent_data = track_data["agents"][i]
        agent_data[agent_data == -1] = np.nan
        
        df_agent = pd.DataFrame(agent_data, columns=agent_fields)
        df_agent = df_agent.loc[df_agent["dist_to_ego"] != 0]
        
        frames.append({
            "ego": ColumnDataSource(df_ego),
            "agents": ColumnDataSource(df_agent)
        })
    return frames, lanelet_source

def visualize_scene(frames, lanelet_source, plot_width=900):
    max_x, min_x = lanelet_source.data["max_x"].max() + 10, lanelet_source.data["min_x"].min() - 10
    max_y, min_y = lanelet_source.data["max_y"].max() + 10, lanelet_source.data["min_y"].min() - 10
    
    agent_hover = HoverTool(
        mode="mouse",
        names=["ego", "agents"],
        tooltips=[
            ("Track id", "@track_id{00}"),
            ("Vel", "(@vx{00.0}, @vy{00.0})"),
            ("Psi", "@psi_rad{00.0}")
        ]
    )

    f = figure(
        plot_width=plot_width,
        aspect_ratio=((max_x - min_x) / (max_y - min_y)),
        x_range=(min_x, max_x),
        y_range=(min_y, max_y),
        background_fill_color="grey",
        background_fill_alpha=0.5,
        tools=["pan", "wheel_zoom", agent_hover, "reset", "save"],
        active_scroll="wheel_zoom"
    )
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

    js_string = """
    sources["ego"].data = frames[cb_obj.value]["ego"].data;
    sources["agents"].data = frames[cb_obj.value]["agents"].data;

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