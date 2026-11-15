from __future__ import annotations

import json
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import available_models, load_data
from src.similarity import compute_item_neighbors, compute_user_item_ranking
from src.components.embedding_plot import make_clickable_plot, make_plot
from src.components.neighbor_table import results_frame
from src.components.eval_panel import render as render_eval
from src.components.linked_3d import linked_3d_maps

st.set_page_config(page_title="Nebula · Embedding Explorer", page_icon="✦", layout="wide")
st.markdown("""
<style>
  :root { --nebula-line: rgba(170,185,220,.13); --nebula-muted: #9ba6bf; }
  .block-container {padding: 1.35rem 2.2rem 2rem; max-width: 1900px;}
  [data-testid="stSidebar"] {display:none;}
  [data-testid="stMetric"] {background:rgba(97,113,160,.08);border:1px solid var(--nebula-line);padding:.65rem .8rem;border-radius:12px;}
  .nebula-header {display:flex;align-items:center;justify-content:space-between;margin-bottom:.45rem;}
  .nebula-brand {font-size:1.05rem;font-weight:700;letter-spacing:.03em;color:#f1f3fb;}
  .nebula-brand span {color:#8c83ff;margin-right:.55rem;font-size:1.2rem;}
  .nebula-kicker {color:#9ba6bf;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;}
  .nebula-title {font-size:2rem;font-weight:700;letter-spacing:-.04em;margin:.4rem 0 0;}
  .nebula-subtitle {color:var(--nebula-muted);font-size:.92rem;margin:.2rem 0 1.15rem;}
  .panel {border:1px solid var(--nebula-line);background:rgba(20,25,43,.55);border-radius:16px;padding:1rem 1.1rem;}
  .panel-title {font-size:1.05rem;font-weight:650;margin:0 0 .55rem;}
  .prompt {color:#aab4ca;font-size:.86rem;border-top:1px solid var(--nebula-line);padding-top:.75rem;margin-top:.75rem;}
  div[data-testid="stPlotlyChart"] {border:1px solid var(--nebula-line);background:rgba(13,17,31,.48);border-radius:16px;overflow:hidden;}
  div[data-testid="stRadio"] > label {display:none;}
  .muted {opacity:.68;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

MODELS = available_models()
if not MODELS:
    st.error("No complete model data found. Run `python generate_synthetic_data.py` from the embedding_explorer directory.")
    st.stop()


@st.cache_data(show_spinner="Loading embeddings and evaluation data…")
def cached_data(model_id: str) -> dict:
    return load_data(model_id)


def model_label(model_id: str) -> str:
    return model_id.replace("_", " ").title()


def color_frame(frame: pd.DataFrame, factor: str) -> tuple[pd.DataFrame, str]:
    data = frame.copy()
    if factor == "Popularity quartile":
        data["popularity_bin"] = pd.qcut(data.popularity_score, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)
        return data, "popularity_bin"
    if factor in {"Energy", "Valence"}:
        key = factor.lower() + "_bin"
        data[key] = pd.cut(data[factor.lower()], 5, duplicates="drop").astype(str)
        return data, key
    return data, {"Item type": "item_type", "Genre": "genre", "Release era": "release_era", "Language": "language"}.get(factor, "item_type")


def selected_item_id(event, frame: pd.DataFrame | None = None, color_column: str | None = None) -> str | None:
    try:
        points = event.selection.points
        if points:
            point = points[-1]
            custom = point.get("customdata")
            if custom is not None:
                return str(custom[0] if isinstance(custom, (list, tuple, np.ndarray)) else custom)
            # Plotly selection payloads can omit customdata. Resolve the trace
            # and point indexes against the deterministic per-category trace order.
            if frame is not None and color_column in frame.columns:
                trace_index = point.get("curve_number", point.get("curveNumber"))
                point_index = point.get("point_index", point.get("pointIndex", point.get("point_number", point.get("pointNumber"))))
                categories = frame[color_column].astype(str).drop_duplicates().tolist()
                if trace_index is not None and point_index is not None and 0 <= int(trace_index) < len(categories):
                    group = frame[frame[color_column].astype(str) == categories[int(trace_index)]]
                    order = np.random.default_rng(17).permutation(len(group))
                    point_index = int(point_index)
                    if 0 <= point_index < len(order):
                        return str(group.iloc[order[point_index]].item_id)
    except Exception:
        pass
    return None


def aligned_coordinates(left: dict, right: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    a = left["items"].set_index("item_id").loc[left["item_ids"], ["x", "y", "z"]].to_numpy(dtype=np.float64)
    b = right["items"].set_index("item_id").loc[left["item_ids"], ["x", "y", "z"]].to_numpy(dtype=np.float64)
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    aa, bb = a - ac, b - bc
    u, singular, vt = np.linalg.svd(bb.T @ aa, full_matrices=False)
    rotation = u @ vt
    scale = singular.sum() / max(np.square(bb).sum(), 1e-12)
    aligned_b = (bb @ rotation) * scale + ac
    left_frame = left["items"].copy()
    right_frame = right["items"].copy()
    aligned_lookup = pd.DataFrame(aligned_b, columns=["x", "y", "z"], index=left["item_ids"])
    for axis in ("x", "y", "z"):
        right_frame[axis] = right_frame.item_id.map(aligned_lookup[axis])
    movement_by_id = pd.Series(np.linalg.norm(a - aligned_b, axis=1), index=left["item_ids"])
    movement = left_frame.item_id.map(movement_by_id).to_numpy(dtype=np.float64)
    return left_frame, right_frame, movement


def radar_chart(values: dict[str, dict[str, float]]) -> go.Figure:
    axes = [
        ("Relevance", "ndcg_at_50"), ("Freshness", "freshness"), ("Diversity", "diversity"),
        ("Discovery", "discovery"), ("Language match", "localization"),
        ("Popularity calibration", "popularity_calibration"),
    ]
    fig = go.Figure()
    colors = ["#20c7df", "#ff7a69"]
    for (model, metrics), color in zip(values.items(), colors):
        names = [name for name, key in axes]
        vals = [float(np.clip(metrics.get(key, 0), 0, 1)) for _, key in axes]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], fill="toself", name=model,
                                      line={"width": 2, "color": color}, fillcolor=f"{color}55"))
    grid = "rgba(170,185,220,.24)"
    fig.update_layout(height=390, margin={"l": 32, "r": 32, "t": 18, "b": 28},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#edf1fa"},
                      polar={"bgcolor": "rgba(13,17,31,.45)",
                             "radialaxis": {"visible": True, "range": [0, 1], "tickvals": [.25, .5, .75, 1],
                                            "gridcolor": grid, "linecolor": grid, "tickfont": {"color": "#9ba6bf", "size": 10}},
                             "angularaxis": {"gridcolor": grid, "linecolor": grid, "tickfont": {"color": "#edf1fa", "size": 11}}},
                      legend={"orientation": "h", "y": -0.1}, showlegend=True)
    return fig


def make_item_slice(model_data: dict, evaluation: pd.DataFrame, users: pd.DataFrame, slice_field: str) -> pd.DataFrame:
    meta_cols = ["item_id", "item_type", "genre", "release_era", "language", "release_year", "popularity_score"]
    metadata = model_data["items"][meta_cols].drop_duplicates("item_id").set_index("item_id")
    if slice_field not in metadata.columns:
        slice_field = "item_type"
    rows = []
    merged = evaluation.merge(users[["user_id"]], on="user_id", how="inner")
    for row in merged.itertuples(index=False):
        truth = set(json.loads(row.ground_truth_items))
        for rank, item_id in enumerate(json.loads(row.top_k_predicted_items), start=1):
            if item_id not in metadata.index:
                continue
            item = metadata.loc[item_id]
            rows.append({"slice": str(item[slice_field]), "hit": int(item_id in truth),
                         "fresh": int(item.release_year >= 2022), "popularity": float(item.popularity_score),
                         "rank": rank, "user_id": row.user_id})
    recs = pd.DataFrame(rows)
    if recs.empty:
        return recs
    out = recs.groupby("slice", dropna=False).agg(recommendations=("item_id" if "item_id" in recs else "rank", "size"),
             users=("user_id", "nunique"), hit_rate=("hit", "mean"), freshness=("fresh", "mean"), mean_popularity=("popularity", "mean")).reset_index()
    out["recommendation_share"] = out.recommendations / max(1, out.recommendations.sum())
    return out


def render_explorer():
    title, model_picker = st.columns([4, 1.2], vertical_alignment="bottom")
    with title:
        st.markdown('<div class="nebula-title">Explore the embedding space</div>', unsafe_allow_html=True)
    chosen_model = model_picker.selectbox("Embedding model", MODELS, format_func=model_label, key="explore_model")
    data = cached_data(chosen_model)
    items, users = data["items"], data["users"]
    head, surprise = st.columns([5, 1])
    with head:
        st.markdown(f'<div class="nebula-subtitle">{len(items):,} items · music and shows in one shared space · 256D embeddings projected for exploration</div>', unsafe_allow_html=True)
    with surprise:
        st.write("")
        if st.button("✦ Surprise me", use_container_width=True):
            st.session_state["selected_item_id"] = str(np.random.choice(data["item_ids"]))
            st.session_state["item_id_input"] = st.session_state["selected_item_id"]

    color_choice, type_filter, popularity = st.columns([1, 1, 1.4])
    color_choice = color_choice.selectbox("Color by", ["Item type", "Genre", "Release era", "Popularity quartile", "Energy", "Valence", "Language"], key="color_by")
    type_filter = type_filter.multiselect("Media", ["music", "podcast"], default=["music", "podcast"], key="item_types")
    pmin, pmax = popularity.slider(
        "Popularity range", 0.0, 1.0, (0.0, 1.0), key="popularity_range",
        help="Focus the map on popular catalog items or the long tail. This filters visible points only; it does not change embeddings or similarity scores.")
    popularity.caption("Focus on mainstream ↔ niche catalog coverage")
    projection_control, view_control, user_control = st.columns([1, 1, 1.3])
    projection = projection_control.selectbox("Projection", ["2D · click to inspect", "3D · rotate"], key="projection_mode")
    view_mode = view_control.selectbox("Layer", ["Items only", "Users only", "Both"], key="view_mode")
    user_id = user_control.selectbox("Inspect user", [""] + users.user_id.tolist(), format_func=lambda x: "Choose user" if not x else x, key="inspect_user")
    focus_dense = False
    if projection.startswith("3D"):
        focus_dense = st.checkbox(
            "Focus on dense region", value=True, key="focus_dense_region",
            help="Frame the central 98% of each projection axis so the main neighborhoods use more of the canvas. Turn off to see the full tails.")
    filtered = items[items.item_type.isin(type_filter) & items.popularity_score.between(pmin, pmax)]
    plot_frame, plot_color = color_frame(filtered, color_choice)
    selected_user = None
    user_points = None
    if view_mode != "Items only":
        user_points = users
    if user_id:
        selected_user = users.loc[users.user_id == user_id].iloc[0]
        # Draw the selected user only in the dedicated highlighted trace below.
        # Keeping them in the background user layer makes the marker look doubled.
        user_points = users.loc[users.user_id != user_id] if view_mode != "Items only" else None
        user_marker = {"user_id": user_id, "x": float(selected_user.x), "y": float(selected_user.y), "z": float(selected_user.z)}
    else:
        user_marker = None

    left, right = st.columns([2.55, 1], gap="large")
    with left:
        click_capability = "CLICK A POINT TO INSPECT" if projection.startswith("2D") else "ROTATE / ZOOM · SWITCH TO 2D TO SELECT"
        title_col, helper_col = st.columns([3.2, 1])
        title_col.markdown(f'<div class="panel-title">{len(filtered):,} items <span class="nebula-kicker">· {click_capability}</span></div>', unsafe_allow_html=True)
        if projection.startswith("3D") and focus_dense:
            helper_col.markdown('<div class="nebula-kicker" style="text-align:right;padding-top:.2rem">DENSE VIEW · 1–99%</div>', unsafe_allow_html=True)
        if projection.startswith("2D"):
            fig = make_clickable_plot(plot_frame, plot_color, view_mode, user_marker, user_points,
                                      selected_ids=[st.session_state.get("selected_item_id", "")])
            event = st.plotly_chart(fig, use_container_width=True, key="explore_map_2d", on_select="rerun", selection_mode="points")
            clicked = selected_item_id(event)
        else:
            fig = make_plot(plot_frame, plot_color, view_mode, user_marker, user_points,
                            selected_ids=[st.session_state.get("selected_item_id", "")], focus_dense=focus_dense)
            st.plotly_chart(fig, use_container_width=True, key="explore_map_3d")
            clicked = None
        if clicked and clicked in set(data["item_ids"]):
            if clicked != st.session_state.get("selected_item_id"):
                st.session_state["selected_item_id"] = clicked
                st.session_state["item_id_input"] = clicked
                st.rerun()
        if projection.startswith("3D") and focus_dense:
            st.caption("Dense view frames the central 98% on each axis; turn it off to include the sparse tails. Similarity and neighbor rankings use the original 256D embeddings.")
        else:
            st.caption("The projection is for visual exploration. Similarity and neighbor rankings use the original 256D embeddings.")
        st.markdown('<div class="prompt">What stands out? &nbsp; · &nbsp; Which items are unexpectedly close? &nbsp; · &nbsp; What would you test next?</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel-title">Item inspector</div>', unsafe_allow_html=True)
        selected_item = st.text_input("Search by item ID", key="item_id_input", placeholder="Search an item…", label_visibility="collapsed")
        if not selected_item:
            with st.container(border=True):
                st.markdown('<div class="nebula-kicker">READY TO EXPLORE</div>', unsafe_allow_html=True)
                st.markdown("### Choose an item")
                st.caption("Click a point in the 2D map or search by ID to see its metadata and nearest neighbors.")
        elif selected_item not in set(data["item_ids"]):
            st.warning("That item ID is not in this model.")
        else:
            item = items.loc[items.item_id == selected_item].iloc[0]
            with st.container(border=True):
                st.markdown(f"### {item.item_name}")
                st.caption(f"{item.item_type.title()}  ·  {item.genre}  ·  {item.release_year}")
                st.metric("Popularity", f"{item.popularity_score:.2f}")
                st.caption(f"Language · {item.language}     Release year · {item.release_year}")
                artist = item.artist_id if pd.notna(item.artist_id) and item.artist_id else "—"
                show = item.podcast_show_id if pd.notna(item.podcast_show_id) and item.podcast_show_id else "—"
                st.write(f"**Artist:** {artist}  ·  **Podcast show:** {show}")
                st.caption(f"Duration · {int(item.duration_seconds):,} sec     Energy · {item.energy:.2f}     Valence · {item.valence:.2f}")
            neighbors = compute_item_neighbors(selected_item, data["item_embeddings"], data["item_ids"], 20)
            st.markdown("**Nearby items** <span class='nebula-kicker'>· ORIGINAL SPACE</span>", unsafe_allow_html=True)
            st.dataframe(results_frame(neighbors, items).drop(columns="distance"), hide_index=True, use_container_width=True, height=330)
    if selected_user is not None:
        st.divider()
        u1, u2 = st.columns([1, 2])
        with u1:
            st.subheader(f"User · {user_id}")
            st.write(f"{selected_user.user_segment.replace('_', ' ').title()} · {selected_user.primary_preference.title()} first")
            st.caption(f"Preferred genre: {selected_user.preferred_genre} · Language: {selected_user.preferred_language} · Region: {selected_user.region}")
        with u2:
            st.markdown("**Items ranked by embedding similarity for this user**")
            st.caption("Candidate generation ranking; it is not the final recommendation list.")
            ranking = compute_user_item_ranking(user_id, data["user_embeddings"], data["user_ids"], data["item_embeddings"], data["item_ids"], 20)
            st.dataframe(results_frame(ranking, items).drop(columns="distance"), hide_index=True, use_container_width=True, height=300)


def comparison_3d_figure(frame: pd.DataFrame, color_column: str, color_map: dict, selected: str,
                         connector_point: np.ndarray, show_connector: bool, height: int = 490) -> go.Figure:
    """Shared-style 3D item map used for side-by-side model comparison."""
    order = np.random.default_rng(17).permutation(len(frame))
    points = frame.iloc[order]
    categories = points[color_column].astype(str)
    colors = categories.map(color_map).fillna('#20c7df').tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=points.x, y=points.y, z=points.z, mode='markers', name='Items', showlegend=False,
        customdata=points[['item_id', 'item_name', color_column]].astype(str).to_numpy(),
        marker={'size': 2.4, 'opacity': .42, 'color': colors},
        hovertemplate='<b>%{customdata[1]}</b><br>ID: %{customdata[0]}<br>%{customdata[2]}<extra></extra>'))
    for category, color in color_map.items():
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers', name=category,
                                   marker={'size': 5, 'color': color}, hoverinfo='skip'))
    current = frame[frame.item_id.astype(str) == str(selected)]
    if not current.empty:
        point = current.iloc[0]
        fig.add_trace(go.Scatter3d(x=[point.x], y=[point.y], z=[point.z], mode='markers', name='Selected item',
                                   customdata=[[str(selected)]], marker={'symbol': 'diamond', 'size': 10,
                                   'color': '#ffffff', 'line': {'color': '#ffbf69', 'width': 3}},
                                   hovertemplate='Selected · %{customdata[0]}<extra></extra>'))
        if show_connector:
            fig.add_trace(go.Scatter3d(x=[point.x, float(connector_point[0])],
                                       y=[point.y, float(connector_point[1])],
                                       z=[point.z, float(connector_point[2])], mode='lines+markers',
                                       name='Aligned movement', showlegend=False,
                                       line={'color': '#ffd166', 'width': 8},
                                       marker={'size': [4, 8], 'color': '#ffd166'}, hoverinfo='skip'))
    axis = {'backgroundcolor': 'rgba(13,17,31,.5)', 'gridcolor': 'rgba(170,185,220,.18)',
            'zerolinecolor': 'rgba(170,185,220,.25)', 'showbackground': True, 'color': '#c5cee1'}
    fig.update_layout(height=height, margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'color': '#e9ecf5', 'size': 10}, hovermode='closest', clickmode='event+select',
                      dragmode='turntable', selectionrevision='nebula-compare-selection',
                      uirevision='nebula-compare-camera', showlegend=False,
                      scene={'xaxis': {**axis, 'title': 'x'}, 'yaxis': {**axis, 'title': 'y'},
                             'zaxis': {**axis, 'title': 'z'}, 'aspectmode': 'cube',
                             'camera': {'eye': {'x': 1.45, 'y': 1.45, 'z': 1.15}}})
    return fig


def render_comparison_item(selected: str, left_data: dict, right_data: dict,
                           left_frame: pd.DataFrame, aligned_right_frame: pd.DataFrame,
                           movement: np.ndarray, left_id: str, right_id: str,
                           include_neighbors: bool = True) -> None:
    row_a = left_frame.loc[left_frame.item_id.astype(str) == str(selected)].iloc[0]
    row_b = aligned_right_frame.loc[aligned_right_frame.item_id.astype(str) == str(selected)].iloc[0]
    item = left_data['items'].loc[left_data['items'].item_id.astype(str) == str(selected)].iloc[0]
    item_type = str(item.item_type).lower()
    glyph = '♫' if item_type == 'music' else '◉'
    color = '#20c7df' if item_type == 'music' else '#ff7a69'
    with st.container(border=True):
        st.markdown(f'<div style="display:flex;align-items:center;gap:.7rem"><span style="width:54px;height:54px;border-radius:12px;background:{color}25;color:{color};display:grid;place-items:center;font-size:1.6rem">{glyph}</span><div><b>{html.escape(str(item.item_name))}</b><br><span class="muted">{html.escape(str(item.item_type).title())} · {html.escape(str(item.genre))} · {int(item.release_year)}</span></div></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('**Embedding movement (aligned space)**')
        st.caption('Model 2 is rotated, centered, and scaled to Model 1 using shared item IDs. These are 3D display coordinates; displacement is not a raw 256D embedding distance.')
        st.write(f'**{model_label(left_id)} coordinates**  ({row_a.x:.3f}, {row_a.y:.3f}, {row_a.z:.3f})')
        st.write(f'**{model_label(right_id)} coordinates**  ({row_b.x:.3f}, {row_b.y:.3f}, {row_b.z:.3f})')
        distance = float(np.linalg.norm([row_a.x-row_b.x, row_a.y-row_b.y, row_a.z-row_b.z]))
        st.metric('Displacement (L2)', f'{distance:.3f}')
    with st.container(border=True):
        st.markdown('**Item metadata**')
        st.write(f'Type · {item.item_type.title()}  |  Genre · {item.genre}  |  Year · {item.release_year}')
        st.write(f'Language · {item.language}  |  Popularity · {item.popularity_score:.2f}')
    if include_neighbors:
        a = compute_item_neighbors(selected, left_data['item_embeddings'], left_data['item_ids'], 20)
        b = compute_item_neighbors(selected, right_data['item_embeddings'], right_data['item_ids'], 20)
        ids_a, ids_b = {x for x, _ in a}, {x for x, _ in b}
        st.markdown('**Top-20 neighbors ⓘ**')
        st.caption('Nearest items by cosine similarity in each model’s original 256D space.')
        st.metric('Retained (in both)', f'{len(ids_a & ids_b)} / 20')
        metadata = left_data['items'].set_index('item_id')
        groups = [('Retained in both', [x for x, _ in a if x in ids_a & ids_b], dict(a)),
                  (f'Lost (Model 1 only) · {len(ids_a - ids_b)}', [x for x, _ in a if x in ids_a - ids_b], dict(a)),
                  (f'New (Model 2 only) · {len(ids_b - ids_a)}', [x for x, _ in b if x in ids_b - ids_a], dict(b))]
        for title, ids, scores in groups:
            st.markdown(f'**{title}**')
            if not ids:
                st.caption('None')
            for neighbor_id in ids[:4]:
                neighbor = metadata.loc[neighbor_id]
                mark = '♫' if neighbor.item_type == 'music' else '◉'
                st.markdown(f'`{mark} {neighbor_id}` · {html.escape(str(neighbor.item_name))}  <span class="muted">cosine {scores[neighbor_id]:.3f}</span>', unsafe_allow_html=True)


def render_model_comparison():
    st.title('Model comparison')
    st.session_state.setdefault('compare_left', MODELS[0])
    st.session_state.setdefault('compare_right', MODELS[min(1, len(MODELS)-1)])
    if st.session_state.pop('compare_swap_requested', False):
        st.session_state['compare_left'], st.session_state['compare_right'] = st.session_state['compare_right'], st.session_state['compare_left']
    first, swap, second, align = st.columns([1.2, .28, 1.2, 1.25], vertical_alignment='bottom')
    left_id = first.selectbox('Model 1', MODELS, format_func=model_label, key='compare_left')
    if swap.button('⇄', help='Swap models', use_container_width=True):
        st.session_state['compare_swap_requested'] = True
        st.rerun()
    right_id = second.selectbox('Model 2', MODELS, format_func=model_label, key='compare_right')
    aligned_view = align.toggle('Aligned view', value=True, key='compare_aligned')
    if left_id == right_id:
        st.info('Choose two different models to compare their embeddings.')
        return
    left_data, right_data = cached_data(left_id), cached_data(right_id)
    if not np.array_equal(left_data['item_ids'], right_data['item_ids']):
        st.error('For this comparison, both models need the same item IDs.')
        return
    left_frame, aligned_right_frame, movement = aligned_coordinates(left_data, right_data)
    right_frame = aligned_right_frame if aligned_view else right_data['items'].copy()
    shared_ids = set(left_frame.item_id.astype(str)) & set(right_frame.item_id.astype(str))
    mover_frame = left_frame.assign(_movement=movement).nlargest(50, '_movement')
    mover_ids = mover_frame.item_id.astype(str).tolist()
    names = left_frame.set_index('item_id').item_name.to_dict()
    if st.session_state.get('comparison_selected_item_id') not in shared_ids:
        st.session_state['comparison_selected_item_id'] = mover_ids[0]
    if aligned_view:
        st.caption(f'Aligned using {len(shared_ids):,} shared items · same item positions can be compared visually.')
    else:
        st.caption('Raw projection coordinates · model spaces are not directly comparable. The movement summary remains based on aligned coordinates.')

    tabs = st.tabs(['Embedding space', 'Neighborhood changes', 'Population shift'])
    with tabs[0]:
        color_col, hint_col = st.columns([1, 3])
        factor = color_col.selectbox('Color by', ['Item type', 'Genre', 'Release era', 'Popularity quartile', 'Language'], key='compare_color_by')
        hint_col.caption('Click a point in either map to inspect the same item in both; rotate the aligned views together.')
        lf, lc = color_frame(left_frame, factor)
        rf, rc = color_frame(right_frame, factor)
        categories = lf[lc].astype(str).drop_duplicates().tolist()
        base_colors = {'music': '#20c7df', 'podcast': '#ff7a69'}
        cycle = ['#7c83ff', '#ffb454', '#57d6b2', '#e788d8', '#a6d77d', '#fb8f5b', '#69a8ff', '#e7d276']
        color_map = {v: base_colors.get(v, cycle[i % len(cycle)]) for i, v in enumerate(categories)}
        map_col, inspector = st.columns([2.45, 1], gap='small')
        with map_col:
            heads = st.columns(2)
            heads[0].markdown(f'**{model_label(left_id)} embeddings** · {len(left_frame):,} items')
            heads[1].markdown(f'**{model_label(right_id)} embeddings** · {len(right_frame):,} items' + (' · aligned' if aligned_view else ' · raw'))
            maps = st.columns(2, gap='small')
            selected = st.session_state['comparison_selected_item_id']
            point_a = aligned_right_frame.loc[aligned_right_frame.item_id.astype(str) == selected, ['x','y','z']].iloc[0].to_numpy()
            point_b = left_frame.loc[left_frame.item_id.astype(str) == selected, ['x','y','z']].iloc[0].to_numpy()
            fig_a = comparison_3d_figure(lf, lc, color_map, selected, point_a, aligned_view)
            fig_b = comparison_3d_figure(rf, rc, color_map, selected, point_b, aligned_view)
            map_selection = linked_3d_maps(
                fig_a, fig_b, link_camera=aligned_view, height=500,
                key=f'compare_linked_maps_{left_id}_{right_id}_v2', default=selected)
            clicked = str(map_selection) if map_selection else ''
            if clicked and clicked in shared_ids and clicked != selected:
                st.session_state['comparison_selected_item_id'] = clicked
                st.session_state['comparison_item_id'] = ''
                st.session_state['compare_mover_select'] = clicked
                st.rerun()
            st.caption('The connector marks the selected item’s aligned 3D movement. Camera rotation is linked while aligned view is on.')
        with inspector:
            st.markdown('**Selected item**')
            options = [''] + list(dict.fromkeys([st.session_state['comparison_selected_item_id']] + mover_ids))
            current = st.session_state['comparison_selected_item_id']
            if st.session_state.get('compare_mover_select') not in options:
                st.session_state['compare_mover_select'] = current
            mover = st.selectbox('High-movement items', options,
                                 format_func=lambda value: 'Choose an item…' if not value else f'{value} · {names[value]}',
                                 key='compare_mover_select')
            if mover and mover != st.session_state['comparison_selected_item_id']:
                st.session_state['comparison_selected_item_id'] = mover
                st.session_state['comparison_item_id'] = ''
            typed = st.text_input('Or search by item ID', key='comparison_item_id', placeholder='e.g. music_000001')
            selected = typed.strip() or st.session_state['comparison_selected_item_id']
            if selected in shared_ids:
                st.session_state['comparison_selected_item_id'] = selected
                render_comparison_item(selected, left_data, right_data, left_frame, aligned_right_frame, movement, left_id, right_id)
            else:
                st.warning('That item ID is not in both selected models.')

        upper = max(float(np.quantile(movement, .995)), 1.0)
        slider_col, value_col, note_col = st.columns([3, .55, 4])
        threshold = slider_col.slider('Displacement threshold (L2)', 0.0, upper, min(1.0, upper),
                                      max(upper / 100, .01), format='%.2f', key='compare_movement_threshold')
        value_col.markdown(f'**{threshold:.2f}**')
        note_col.caption('Aligned 3D positions · use this threshold to focus on larger or smaller moves.')
        hist = go.Figure(go.Histogram(x=movement, nbinsx=45, marker_color='#758ee8', opacity=.9))
        hist.add_vline(x=threshold, line_dash='dash', line_color='#ffca6b', line_width=2,
                       annotation_text=f'Threshold = {threshold:.2f}', annotation_position='top right')
        hist.update_layout(title='Item movement distribution', xaxis_title='Displacement (L2) · aligned projection',
                           yaxis_title='Number of items', height=270, margin={'t': 45, 'b': 20, 'l': 5, 'r': 5},
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,31,.35)', font={'color': '#e9ecf5'},
                           xaxis={'gridcolor': 'rgba(180,190,220,.12)'}, yaxis={'gridcolor': 'rgba(180,190,220,.12)'})
        chart, explanation = st.columns([2.6, 1])
        chart.plotly_chart(hist, use_container_width=True, key='compare_movement_histogram')
        with explanation:
            st.markdown('**How to read this**')
            st.caption('Model 2 is rotated, centered, and scaled to Model 1 using shared item IDs. Displacement is a map change, not a raw 256D distance or quality score. Neighbor rankings use each model’s original 256D vectors.')

    with tabs[1]:
        selected = st.session_state['comparison_selected_item_id']
        st.markdown('### Neighborhood changes for the selected item')
        st.caption('Nearest neighbors are ranked by cosine similarity in each model’s original 256D embedding space.')
        detail, neighbors = st.columns([1, 1.5], gap='medium')
        with detail:
            render_comparison_item(selected, left_data, right_data, left_frame, aligned_right_frame, movement, left_id, right_id, include_neighbors=False)
        with neighbors:
            a = compute_item_neighbors(selected, left_data['item_embeddings'], left_data['item_ids'], 20)
            b = compute_item_neighbors(selected, right_data['item_embeddings'], right_data['item_ids'], 20)
            ids_a, ids_b = {x for x, _ in a}, {x for x, _ in b}
            retained, lost, new = ids_a & ids_b, ids_a - ids_b, ids_b - ids_a
            count_cols = st.columns(3)
            count_cols[0].metric('Retained in both', f'{len(retained)} / 20')
            count_cols[1].metric('Lost · Model 1 only', len(lost))
            count_cols[2].metric('New · Model 2 only', len(new))
            metadata = left_data['items'].set_index('item_id')
            groups = [('Retained in both', [x for x, _ in a if x in retained], dict(a)),
                      ('Lost · Model 1 only', [x for x, _ in a if x in lost], dict(a)),
                      ('New · Model 2 only', [x for x, _ in b if x in new], dict(b))]
            for title, ids, scores in groups:
                with st.container(border=True):
                    st.markdown(f'**{title}**')
                    if not ids:
                        st.caption('None')
                    for item_id in ids[:8]:
                        neighbor = metadata.loc[item_id]
                        glyph = '♫' if neighbor.item_type == 'music' else '◉'
                        tint = '#20c7df' if neighbor.item_type == 'music' else '#ff7a69'
                        st.markdown(f'<div style="display:flex;align-items:center;gap:.55rem;padding:.22rem 0;border-bottom:1px solid rgba(170,185,220,.1)"><span style="width:30px;height:30px;border-radius:7px;background:{tint}30;color:{tint};display:grid;place-items:center">{glyph}</span><span><b>{html.escape(str(item_id))}</b> · {html.escape(str(neighbor.item_name))}<br><span class="muted">{html.escape(str(neighbor.item_type))} · {html.escape(str(neighbor.genre))} · cosine {scores[item_id]:.3f}</span></span></div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('### Where is movement concentrated?')
        st.caption('See which metadata groups account for larger map changes. This is descriptive; it does not say which model is better.')
        factor = st.selectbox('Group items by', ['Item type', 'Genre', 'Release era', 'Language'], key='compare_population_factor')
        column = {'Item type': 'item_type', 'Genre': 'genre', 'Release era': 'release_era', 'Language': 'language'}[factor]
        threshold = st.session_state.get('compare_movement_threshold', 1.0)
        pop = left_frame[['item_id', 'item_type', 'genre', 'release_era', 'language']].copy()
        pop['movement'] = movement
        pop['above'] = pop.movement >= threshold
        summary = pop.groupby(column, dropna=False).agg(items=('item_id', 'size'), median=('movement', 'median'),
                    p90=('movement', lambda values: float(np.quantile(values, .9))), above_share=('above', 'mean')).reset_index()
        summary = summary.rename(columns={column: 'Group'}).sort_values('median', ascending=False)
        if factor == 'Genre':
            summary = summary.head(12)
        left_chart, right_chart = st.columns(2, gap='medium')
        fig_median = px.bar(summary.sort_values('median'), x='median', y='Group', orientation='h',
                            color='median', color_continuous_scale=['#20c7df', '#ff7a69'], hover_data=['items', 'p90'])
        fig_median.update_layout(title='Median aligned movement by group', xaxis_title='Median displacement (L2)', yaxis_title=None,
                                 height=390, coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)',
                                 plot_bgcolor='rgba(13,17,31,.35)', font={'color': '#e9ecf5'})
        fig_share = px.bar(summary.sort_values('above_share'), x='above_share', y='Group', orientation='h',
                           color='above_share', color_continuous_scale=['#20c7df', '#ff7a69'], hover_data=['items'])
        fig_share.update_layout(title=f'Share above {threshold:.2f} threshold', xaxis={'tickformat': '.0%'},
                                xaxis_title='Items above threshold', yaxis_title=None, height=390,
                                coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(13,17,31,.35)', font={'color': '#e9ecf5'})
        left_chart.plotly_chart(fig_median, use_container_width=True, key='compare_population_median')
        right_chart.plotly_chart(fig_share, use_container_width=True, key='compare_population_share')
        st.dataframe(summary.rename(columns={'items': 'Item count', 'median': 'Median movement', 'p90': '90th percentile', 'above_share': 'Share above threshold'}), hide_index=True, use_container_width=True)


QUALITY_TARGETS = {
    "ndcg_at_50": .85, "freshness": .23, "diversity": .95,
    "discovery": .15, "localization": .45, "popularity_calibration": .95,
}
QUALITY_LABELS = {
    "ndcg_at_50": "Relevance", "freshness": "Freshness", "diversity": "Diversity",
    "discovery": "Discovery", "localization": "Language match",
    "popularity_calibration": "Popularity calibration",
}


def mean_ci(values, bounded: bool = True) -> tuple[float, float, float, int]:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
    if values.empty:
        return 0.0, 0.0, 0.0, 0
    mean = float(values.mean())
    margin = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    low, high = mean - margin, mean + margin
    if bounded:
        low, high = max(0.0, low), min(1.0, high)
    return mean, low, high, len(values)


def rec_item_rows(data: dict, evaluation: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Expand top-50 recommendations so metadata-level exposure can be explored."""
    user_columns = ["user_id", "preferred_genre", "preferred_language"]
    joined = evaluation[["user_id", "top_k_predicted_items"]].merge(users[user_columns], on="user_id", how="inner")
    item_meta = data["items"].set_index("item_id")
    rows = []
    for row in joined.itertuples(index=False):
        for item_id in json.loads(row.top_k_predicted_items):
            if item_id not in item_meta.index:
                continue
            item = item_meta.loc[item_id]
            rows.append({"user_id": row.user_id, "item_id": item_id, "item_type": item.item_type,
                         "genre": item.genre, "release_era": item.release_era, "language": item.language,
                         "release_year": int(item.release_year), "popularity_score": float(item.popularity_score),
                         "genre_match": str(item.genre) == str(row.preferred_genre),
                         "language_match": str(item.language) == str(row.preferred_language)})
    return pd.DataFrame(rows)


def apply_item_filters(frame: pd.DataFrame, item_type: str, genre: str, release_era: str, language: str) -> pd.DataFrame:
    filtered = frame
    for column, value in (("item_type", item_type), ("genre", genre), ("release_era", release_era), ("language", language)):
        if value != "All" and column in filtered:
            filtered = filtered[filtered[column].astype(str) == value]
    return filtered


def render_quality():
    st.title("Model quality")
    st.caption("Evaluation analytics for recommendation models")
    st.info("Illustrative demo data · All scores and intervals are synthetic examples.", icon="ℹ️")

    datasets = {model_id: cached_data(model_id) for model_id in MODELS[:2]}
    if len(datasets) < 2:
        st.warning("Select two available models to compare model quality.")
        return
    model_a, model_b = MODELS[:2]
    users = datasets[model_a]["users"].drop(columns=["x", "y", "z", "projection_method"], errors="ignore")
    segment_filter_values = ["All segments"] + sorted(users.user_segment.dropna().astype(str).unique().tolist())
    preference_filter_values = ["All preferences"] + sorted(users.primary_preference.dropna().astype(str).unique().tolist())
    item_meta = datasets[model_a]["items"]
    item_type_values = ["All"] + sorted(item_meta.item_type.dropna().astype(str).unique().tolist())
    genre_values = ["All"] + sorted(item_meta.genre.dropna().astype(str).unique().tolist())
    era_values = ["All"] + sorted(item_meta.release_era.dropna().astype(str).unique().tolist())
    language_values = ["All"] + sorted(item_meta.language.dropna().astype(str).unique().tolist())

    cohort_pairs = users[["user_segment", "primary_preference"]].drop_duplicates().sort_values(["primary_preference", "user_segment"])
    cohort_labels = ["All users"] + [
        f"{row.primary_preference.title()} listeners · {row.user_segment.title()} users"
        for row in cohort_pairs.itertuples(index=False)
    ]
    cohort_lookup = {label: (row.user_segment, row.primary_preference)
                     for label, row in zip(cohort_labels[1:], cohort_pairs.itertuples(index=False))}
    default_cohort = next((label for label in cohort_labels if "Podcast listeners · Regular users" == label), "All users")

    top_a, vs, top_b, cohort_col = st.columns([1.05, .15, 1.05, 1.8], vertical_alignment="bottom")
    with top_a:
        st.selectbox("Compare models", MODELS, index=0, format_func=model_label, key="quality_a")
    with vs:
        st.markdown('<div style="text-align:center;padding-bottom:.55rem;color:#9ba6bf">vs</div>', unsafe_allow_html=True)
    with top_b:
        st.selectbox("Model 2", MODELS, index=1, format_func=model_label, key="quality_b")
    focus_cohort = cohort_col.selectbox("User cohort", cohort_labels, index=cohort_labels.index(default_cohort), key="quality_focus_cohort")
    model_a = st.session_state["quality_a"]
    model_b = st.session_state["quality_b"]
    if model_a == model_b:
        st.info("Choose two different models to compare performance.")
        return
    if set((model_a, model_b)) != set(datasets):
        datasets = {model_id: cached_data(model_id) for model_id in (model_a, model_b)}
        users = datasets[model_a]["users"].drop(columns=["x", "y", "z", "projection_method"], errors="ignore")
        item_meta = datasets[model_a]["items"]

    filters = st.columns(6, gap="small")
    segment_filter = filters[0].selectbox("User segment", segment_filter_values, key="quality_segment_filter")
    preference_filter = filters[1].selectbox("Primary preference", preference_filter_values, key="quality_preference_filter")
    selected_type = filters[2].selectbox("Item type", item_type_values, key="quality_item_type_filter")
    selected_genre = filters[3].selectbox("Genre", genre_values, key="quality_genre_filter")
    selected_era = filters[4].selectbox("Release era", era_values, key="quality_era_filter")
    selected_language = filters[5].selectbox("Language", language_values, key="quality_language_filter")

    selected_users = users.copy()
    if focus_cohort != "All users":
        segment, preference = cohort_lookup[focus_cohort]
        selected_users = selected_users[(selected_users.user_segment == segment) & (selected_users.primary_preference == preference)]
    if segment_filter != "All segments":
        selected_users = selected_users[selected_users.user_segment.astype(str) == segment_filter]
    if preference_filter != "All preferences":
        selected_users = selected_users[selected_users.primary_preference.astype(str) == preference_filter]
    selected_user_ids = set(selected_users.user_id.astype(str))
    if not selected_user_ids:
        st.warning("These cohort filters do not match any users.")
        return

    selected_evaluations = {}
    for model_id in (model_a, model_b):
        evaluation = datasets[model_id]["evaluation"]
        selected_evaluations[model_id] = evaluation[evaluation.user_id.astype(str).isin(selected_user_ids)].copy()

    summaries = {}
    for model_id, evaluation in selected_evaluations.items():
        summary = {key: mean_ci(evaluation[key], bounded=(key != "diversity"))
                   for key in (*QUALITY_TARGETS, "recall_at_50")}
        predicted = [item_id for value in evaluation.top_k_predicted_items for item_id in json.loads(value)]
        coverage = len(set(predicted)) / max(1, len(datasets[model_id]["items"]))
        summaries[model_id] = {"metrics": summary, "coverage": coverage, "users": int(evaluation.user_id.nunique())}

    def metric_card(title: str, key: str):
        with st.container(border=True):
            st.markdown(f"**{title} ⓘ**")
            if key == "coverage":
                for model_id, color in ((model_a, "#20c7df"), (model_b, "#ff7a69")):
                    st.markdown(f'<span style="color:{color}">●</span> {html.escape(model_label(model_id))} '
                                f'<b>{summaries[model_id]["coverage"]:.1%}</b>', unsafe_allow_html=True)
            elif key == "users":
                st.markdown(f'**n = {summaries[model_a]["users"]:,}**')
                st.caption(f'{model_label(model_a)}: {summaries[model_a]["users"]:,} · {model_label(model_b)}: {summaries[model_b]["users"]:,}')
            else:
                for model_id, color in ((model_a, "#20c7df"), (model_b, "#ff7a69")):
                    mean, low, high, _ = summaries[model_id]["metrics"][key]
                    st.markdown(f'<span style="color:{color}">●</span> {html.escape(model_label(model_id))} '
                                f'<b>{mean:.3f}</b> <span class="muted">({low:.3f}–{high:.3f})</span>', unsafe_allow_html=True)

    kpis = st.columns(4, gap="small")
    for col, title, key in zip(kpis, ("NDCG@50", "Recall@50", "Catalog coverage", "Cohort size"),
                               ("ndcg_at_50", "recall_at_50", "coverage", "users")):
        with col:
            metric_card(title, key)

    metric_labels = {"ndcg_at_50": "NDCG@50", "recall_at_50": "Recall@50", **QUALITY_LABELS}
    model_raw = {model_label(model_a): {key: summaries[model_a]["metrics"][key][0] for key in QUALITY_TARGETS},
                 model_label(model_b): {key: summaries[model_b]["metrics"][key][0] for key in QUALITY_TARGETS}}
    normalized = {label: {key: min(1.0, value / QUALITY_TARGETS[key]) for key, value in metrics.items()}
                  for label, metrics in model_raw.items()}

    profile_col, definition_col = st.columns([1.65, .95], gap="small")
    with profile_col:
        st.markdown("### Model quality (normalized per dimension)")
        st.caption("Normalized to illustrative targets; 1.0 means the target is met. These are separate dimensions, not a single winner score.")
        st.plotly_chart(radar_chart(normalized), use_container_width=True, key="quality_radar")
        score_columns = st.columns(3, gap="small")
        for index, (key, label) in enumerate(QUALITY_LABELS.items()):
            with score_columns[index % 3]:
                st.markdown(f"**{label}**<br><span style='color:#20c7df'>{normalized[model_label(model_a)][key]:.2f}</span> · "
                            f"<span style='color:#ff7a69'>{normalized[model_label(model_b)][key]:.2f}</span>", unsafe_allow_html=True)
    with definition_col:
        with st.container(border=True):
            st.markdown("### Metric definitions / key")
            st.markdown("""
            **Freshness** · share of recommendations released recently (2022+).

            **Diversity** · spread across genres and artists/shows.

            **Discovery** · items from genres absent in recent history.

            **Language match** · share in the user's preferred language.

            **Popularity calibration** · head/mid/tail mix vs. the user's target.

            **Catalog coverage** · unique catalog items surfaced in top-50 lists.
            """)
            st.caption("Demo targets: NDCG .85 · Freshness .23 · Diversity .95 · Discovery .15 · Language .45 · Popularity calibration .95. Normalized values are capped at 1.0.")
    exposure_col, slice_col = st.columns(2, gap="small")
    with exposure_col:
        st.markdown("### Popularity exposure ⓘ")
        exposure_rows = []
        for model_id in (model_a, model_b):
            data = datasets[model_id]
            recs = rec_item_rows(data, selected_evaluations[model_id], users)
            recs = apply_item_filters(recs, selected_type, selected_genre, selected_era, selected_language)
            if recs.empty:
                continue
            q20, q80 = item_meta.popularity_score.quantile([.2, .8]).tolist()
            recs["Popularity bucket"] = np.where(recs.popularity_score >= q80, "Head (top 20%)",
                                                  np.where(recs.popularity_score >= q20, "Mid (next 60%)", "Long-tail (last 20%)"))
            shares = recs["Popularity bucket"].value_counts(normalize=True)
            for bucket in ("Head (top 20%)", "Mid (next 60%)", "Long-tail (last 20%)"):
                exposure_rows.append({"Model": model_label(model_id), "Bucket": bucket, "Share": float(shares.get(bucket, 0))})
        if exposure_rows:
            exposure_fig = px.bar(pd.DataFrame(exposure_rows), x="Share", y="Model", color="Bucket", text_auto=".0%",
                                  orientation="h", barmode="stack", category_orders={"Bucket": ["Head (top 20%)", "Mid (next 60%)", "Long-tail (last 20%)"]},
                                  color_discrete_map={"Head (top 20%)": "#20c7df", "Mid (next 60%)": "#b8c1d6", "Long-tail (last 20%)": "#ff7a69"})
            exposure_fig.update_layout(height=210, margin={"l": 8, "r": 8, "t": 8, "b": 8}, yaxis_title=None, xaxis={"range": [0, 1], "tickformat": ".0%"}, legend={"orientation": "h", "y": -0.24, "font": {"size": 9}})
            st.plotly_chart(exposure_fig, use_container_width=True, key="quality_popularity_exposure")
            st.caption("Share of recommendation exposure by catalog popularity bucket. Context only; a larger head or tail share is not inherently better.")
        else:
            st.info("No recommended items match these metadata filters.")

    with slice_col:
        st.markdown("### Item slice (share of recommendations) ⓘ")
        if exposure_rows:
            slices = []
            for model_id in (model_a, model_b):
                # recs still contains the last model's data; build the filtered list for each model explicitly.
                model_recs = rec_item_rows(datasets[model_id], selected_evaluations[model_id], users)
                model_recs = apply_item_filters(model_recs, selected_type, selected_genre, selected_era, selected_language)
                if model_recs.empty:
                    continue
                denominator = len(model_recs)
                shares = {
                    "Music": float((model_recs.item_type == "music").mean()),
                    "Podcasts": float((model_recs.item_type == "podcast").mean()),
                    "Genre match": float(model_recs.genre_match.mean()),
                    "Release era\n(2010+)": float((model_recs.release_year >= 2010).mean()),
                }
                for slice_name, share in shares.items():
                    slices.append({"Slice": slice_name, "Share": share, "Model": model_label(model_id), "Recommendations": denominator})
            mix_fig = px.bar(pd.DataFrame(slices), x="Slice", y="Share", color="Model", barmode="group", text_auto=".0%",
                             color_discrete_map={model_label(model_a): "#20c7df", model_label(model_b): "#ff7a69"}, hover_data=["Recommendations"])
            mix_fig.update_layout(height=225, margin={"l": 8, "r": 8, "t": 8, "b": 10}, yaxis={"range": [0, 1], "tickformat": ".0%"}, xaxis_title=None, legend={"orientation": "h", "y": 1.12})
            st.plotly_chart(mix_fig, use_container_width=True, key="quality_item_slice")
        else:
            st.info("No recommended items match these metadata filters.")

    st.divider()
    st.markdown("### Cohort comparison <span class='muted'>· illustrative demo data</span>", unsafe_allow_html=True)
    st.caption("Model means show approximate 95% confidence intervals. Delta intervals compare the same users in both models. Metadata filters affect the exposure charts above; user filters affect this table.")
    cohort_specs = []
    for segment in sorted(users.user_segment.dropna().astype(str).unique()):
        pretty_segment = segment.replace("_", " ").title()
        cohort_specs.append(("Power users" if "power" in segment.lower() else f"{pretty_segment} listeners", users.user_segment.astype(str) == segment))
    for preference in sorted(users.primary_preference.dropna().astype(str).unique()):
        cohort_specs.append((f"{preference.title()}-first", users.primary_preference.astype(str) == preference))
    cohort_output = []
    for cohort_name, cohort_mask in cohort_specs:
        cohort_users = users[cohort_mask]
        if segment_filter != "All segments":
            cohort_users = cohort_users[cohort_users.user_segment.astype(str) == segment_filter]
        if preference_filter != "All preferences":
            cohort_users = cohort_users[cohort_users.primary_preference.astype(str) == preference_filter]
        ids = set(cohort_users.user_id.astype(str))
        # The top-line cohort selector scopes the KPI cards. Keep the comparison
        # table independent so it remains useful for scanning across cohorts.
        cohort_eval_a = datasets[model_a]["evaluation"]
        cohort_eval_b = datasets[model_b]["evaluation"]
        ids_a = cohort_eval_a[cohort_eval_a.user_id.astype(str).isin(ids)].set_index("user_id")
        ids_b = cohort_eval_b[cohort_eval_b.user_id.astype(str).isin(ids)].set_index("user_id")
        common_ids = ids_a.index.intersection(ids_b.index)
        if len(common_ids) == 0:
            continue
        ndcg_a = mean_ci(ids_a.loc[common_ids, "ndcg_at_50"])
        ndcg_b = mean_ci(ids_b.loc[common_ids, "ndcg_at_50"])
        deltas = {}
        for key in ("ndcg_at_50", "freshness", "diversity"):
            delta_values = ids_b.loc[common_ids, key].astype(float).to_numpy() - ids_a.loc[common_ids, key].astype(float).to_numpy()
            delta_mean = float(np.mean(delta_values))
            margin = 1.96 * float(np.std(delta_values, ddof=1)) / np.sqrt(len(delta_values)) if len(delta_values) > 1 else 0.0
            deltas[key] = (delta_mean, delta_mean - margin, delta_mean + margin)
        cohort_output.append({
            "Cohort": cohort_name,
            f"{model_label(model_a)} NDCG@50": f"{ndcg_a[0]:.3f} ({ndcg_a[1]:.3f}–{ndcg_a[2]:.3f})",
            f"{model_label(model_b)} NDCG@50": f"{ndcg_b[0]:.3f} ({ndcg_b[1]:.3f}–{ndcg_b[2]:.3f})",
            "Δ NDCG": f"{deltas['ndcg_at_50'][0]:+.3f} ({deltas['ndcg_at_50'][1]:+.3f}–{deltas['ndcg_at_50'][2]:+.3f})",
            "Δ Freshness": f"{deltas['freshness'][0]:+.3f} ({deltas['freshness'][1]:+.3f}–{deltas['freshness'][2]:+.3f})",
            "Δ Diversity": f"{deltas['diversity'][0]:+.3f} ({deltas['diversity'][1]:+.3f}–{deltas['diversity'][2]:+.3f})",
            "Sample n": f"n = {len(common_ids):,}",
        })
    if cohort_output:
        cohort_df = pd.DataFrame(cohort_output)
        table_headers = list(cohort_df.columns)
        widths = ["12%", "15%", "15%", "17%", "16%", "16%", "9%"]
        header_cells = "".join(
            f'<th style="width:{width};text-align:left;padding:.6rem .55rem;border-bottom:1px solid rgba(170,185,220,.24);">{html.escape(header)}</th>'
            for header, width in zip(table_headers, widths)
        )
        table_rows = []
        for row_index, row in cohort_df.iterrows():
            cells = []
            for header, width in zip(table_headers, widths):
                value = str(row[header])
                style = f"width:{width};padding:.58rem .55rem;border-bottom:1px solid rgba(170,185,220,.14);vertical-align:top;"
                if header in {"Δ NDCG", "Δ Freshness", "Δ Diversity"}:
                    try:
                        delta = float(value.split(" ")[0])
                        style += "background:rgba(32,199,223,.18);" if delta > 0 else "background:rgba(255,122,105,.18);" if delta < 0 else ""
                    except (TypeError, ValueError):
                        pass
                if header == "Cohort":
                    style += "font-weight:600;"
                cells.append(f'<td style="{style}">{html.escape(value)}</td>')
            row_bg = "rgba(97,113,160,.08)" if row_index % 2 else "transparent"
            table_rows.append(f'<tr style="background:{row_bg}">{"".join(cells)}</tr>')
        st.markdown(
            '<div style="border:1px solid rgba(170,185,220,.18);border-radius:12px;overflow:hidden;">'
            '<table style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:.76rem;line-height:1.35;">'
            f'<thead><tr style="background:rgba(97,113,160,.12);">{header_cells}</tr></thead>'
            f'<tbody>{"".join(table_rows)}</tbody></table></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No users match the selected cohort filters.")
    st.caption("All values are synthetic demonstration metrics. Confidence intervals summarize variation across the included users and are not a substitute for a time-based offline evaluation.")


st.markdown('<div class="nebula-header"><div class="nebula-brand"><span>✦</span>NEBULA <span class="nebula-kicker">/ RECOMMENDATION LAB</span></div><div class="nebula-kicker">LOCAL EXPLORATION WORKSPACE</div></div>', unsafe_allow_html=True)
page = st.radio("Workspace", ["Item space", "Compare models", "Model quality"], horizontal=True, label_visibility="collapsed", key="workspace_page")
st.markdown('<div style="border-bottom:1px solid rgba(170,185,220,.13);margin:-.25rem 0 1.2rem"></div>', unsafe_allow_html=True)

if page == "Item space":
    render_explorer()
elif page == "Compare models":
    render_model_comparison()
else:
    render_quality()
