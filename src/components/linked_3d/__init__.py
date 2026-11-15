from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT_DIR = Path(__file__).resolve().parent
_linked_maps = components.declare_component("nebula_linked_3d_maps", path=str(_COMPONENT_DIR))


def linked_3d_maps(figure_a, figure_b, link_camera=True, height=510, key=None, default=""):
    """Render two Plotly 3D maps with click-to-select and synchronized cameras."""
    return _linked_maps(
        figure_a=figure_a.to_json(),
        figure_b=figure_b.to_json(),
        link_camera=bool(link_camera),
        height=int(height),
        key=key,
        default=default,
    )
