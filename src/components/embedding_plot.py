import plotly.graph_objects as go
import numpy as np


def make_plot(frame, color_by="item_type", view_mode="Items only", user=None, user_points=None, selected_ids=(), focus_dense=False):
    color = color_by if color_by in frame.columns else "item_type"
    palette = {"music": "#20c7df", "podcast": "#ff7a69"}
    cycle = ["#7c83ff", "#ffb454", "#57d6b2", "#e788d8", "#a6d77d", "#fb8f5b", "#69a8ff", "#e7d276"]
    categories = frame[color].astype(str).drop_duplicates().tolist()
    fig = go.Figure()
    color_by_value = {value: palette.get(value, cycle[i % len(cycle)]) for i, value in enumerate(categories)}
    # A single interleaved item trace prevents later category traces from
    # painting a solid color over the center of a dense 3D cloud.
    order = np.random.default_rng(17).permutation(len(frame))
    points = frame.iloc[order]
    point_colors = points[color].astype(str).map(color_by_value).to_numpy()
    fig.add_trace(go.Scatter3d(
        x=points.x.to_numpy(dtype=np.float32), y=points.y.to_numpy(dtype=np.float32), z=points.z.to_numpy(dtype=np.float32),
        mode="markers", name="Items", showlegend=False,
        customdata=points[["item_id", "item_name", color]].astype(str).to_numpy(),
        marker={"size": 2.0, "opacity": .24, "color": point_colors},
        hovertemplate="<b>%{customdata[1]}</b><br>ID: %{customdata[0]}<br>%{customdata[2]}<br>x: %{x:.2f} · y: %{y:.2f} · z: %{z:.2f}<extra></extra>"))
    # Keep a compact category legend while the points themselves share one trace.
    for value in categories:
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", name=value,
                      marker={"size": 4, "color": color_by_value[value]}, hoverinfo="skip"))
    axis_range = None
    if focus_dense and not frame.empty:
        axis_range = {}
        for axis in ("x", "y", "z"):
            lo, hi = np.quantile(frame[axis].to_numpy(dtype=np.float32), [.01, .99])
            pad = max((hi - lo) * .05, .1)
            axis_range[axis] = [float(lo - pad), float(hi + pad)]
    if selected_ids:
        selected = frame[frame.item_id.isin(selected_ids)]
        if not selected.empty:
            fig.add_trace(go.Scatter3d(x=selected.x, y=selected.y, z=selected.z, mode="markers", name="Selected",
                         marker={"symbol": "diamond", "size": 9, "color": "#ff6b35", "line": {"width": 2, "color": "white"}},
                         text=selected.item_name, customdata=selected[["item_id"]], hovertemplate="%{text}<extra>Selected</extra>"))
    if view_mode in ("Users only", "Both") and user_points is not None:
        fig.add_trace(go.Scatter3d(x=user_points.x, y=user_points.y, z=user_points.z, mode="markers", name="Users",
                     marker={"symbol": "diamond", "size": 5, "color": "#eee8ff", "opacity": .6},
                     text=user_points.user_id, hovertemplate="%{text}<extra>User embedding (preferred-topic centroid)</extra>"))
    if view_mode in ("Users only", "Both") and user is not None:
        fig.add_trace(go.Scatter3d(x=[user["x"]], y=[user["y"]], z=[user["z"]], mode="markers", name=user["user_id"],
                     marker={"symbol": "diamond", "size": 11, "color": "#f4a9ff", "line": {"color": "white", "width": 2}},
                     hovertemplate=f"{user['user_id']}<extra>User embedding</extra>"))
    if view_mode == "Users only":
        fig.update_traces(visible="legendonly", selector=lambda t: t.name not in {"Users", "Selected", user["user_id"] if user else ""})
    fig.update_layout(height=720, margin={"l": 0, "r": 0, "t": 24, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#e9ecf5"}, legend_title_text=color.replace("_", " ").title(),
                      scene={"bgcolor": "rgba(0,0,0,0)", "aspectmode": "cube",
                             "xaxis_title": "Projection dimension 1", "yaxis_title": "Projection dimension 2", "zaxis_title": "Projection dimension 3",
                             "xaxis": {"gridcolor": "rgba(180,190,220,.15)", "zerolinecolor": "rgba(180,190,220,.22)", **({"range": axis_range["x"]} if axis_range else {})},
                             "yaxis": {"gridcolor": "rgba(180,190,220,.15)", "zerolinecolor": "rgba(180,190,220,.22)", **({"range": axis_range["y"]} if axis_range else {})},
                             "zaxis": {"gridcolor": "rgba(180,190,220,.15)", "zerolinecolor": "rgba(180,190,220,.22)", **({"range": axis_range["z"]} if axis_range else {})}},
                      dragmode="turntable")
    return fig


def make_clickable_plot(frame, color_by="item_type", view_mode="Items only", user=None, user_points=None,
                       selected_ids=(), height=660, axis_ranges=None):
    """2D projection with native Streamlit point-selection events."""
    color = color_by if color_by in frame.columns else "item_type"
    palette = {"music": "#20c7df", "podcast": "#ff7a69"}
    cycle = ["#7c83ff", "#ffb454", "#57d6b2", "#e788d8", "#a6d77d", "#fb8f5b", "#69a8ff", "#e7d276"]
    categories = frame[color].astype(str).drop_duplicates().tolist()
    fig = go.Figure()
    color_by_value = {value: palette.get(value, cycle[i % len(cycle)]) for i, value in enumerate(categories)}
    # Keep one selectable trace per category. Use light markers so dense
    # neighborhoods show structure without filling in as solid disks.
    for value in categories:
        group = frame[frame[color].astype(str) == value]
        order = np.random.default_rng(17).permutation(len(group))
        points = group.iloc[order]
        fig.add_trace(go.Scattergl(
            x=points.x, y=points.y, mode="markers", name=value,
            customdata=points[["item_id", "item_name", "item_type", "genre", "release_year", "popularity_score"]].to_numpy(),
            marker={"size": 3, "opacity": .24, "color": color_by_value[value]},
            hovertemplate="<b>%{customdata[1]}</b><br>ID: %{customdata[0]}<br>%{customdata[2]} · %{customdata[3]} · %{customdata[4]}<br>Popularity: %{customdata[5]:.2f}<extra></extra>"))
    if selected_ids:
        selected = frame[frame.item_id.isin(selected_ids)]
        if not selected.empty:
            fig.add_trace(go.Scattergl(x=selected.x, y=selected.y, mode="markers", name="Selected",
                customdata=selected[["item_id"]], marker={"symbol": "diamond", "size": 12, "color": "#ff6b35", "line": {"width": 2, "color": "white"}},
                hovertemplate="Selected · %{customdata[0]}<extra></extra>"))
    if view_mode in ("Users only", "Both") and user_points is not None:
        fig.add_trace(go.Scattergl(x=user_points.x, y=user_points.y, mode="markers", name="Users", text=user_points.user_id,
            marker={"symbol": "diamond", "size": 7, "color": "#eee8ff", "opacity": .6}, hovertemplate="%{text}<extra>User embedding</extra>"))
    if view_mode in ("Users only", "Both") and user is not None:
        fig.add_trace(go.Scattergl(x=[user["x"]], y=[user["y"]], mode="markers", name=user["user_id"],
            marker={"symbol": "diamond", "size": 14, "color": "#f4a9ff", "line": {"color": "white", "width": 2}},
            hovertemplate=f"{user['user_id']}<extra>User embedding</extra>"))
    if view_mode == "Users only":
        fig.update_traces(visible="legendonly", selector=lambda trace: trace.name not in {"Users", "Selected", user["user_id"] if user else ""})
    fig.update_layout(
        height=height, margin={"l": 10, "r": 12, "t": 18, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e9ecf5"}, legend_title_text=color.replace("_", " ").title(),
        hovermode="closest", clickmode="event+select", dragmode="pan",
        xaxis={"title": "Projection dimension 1", "gridcolor": "rgba(180,190,220,.13)", "zerolinecolor": "rgba(180,190,220,.2)", **({"range": axis_ranges["x"]} if axis_ranges else {})},
        yaxis={"title": "Projection dimension 2", "gridcolor": "rgba(180,190,220,.13)", "zerolinecolor": "rgba(180,190,220,.2)", **({"range": axis_ranges["y"]} if axis_ranges else {})},
        selectionrevision="nebula-selection",
    )
    return fig
