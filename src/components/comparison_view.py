import streamlit as st


def render(first, second, metadata, similarity, first_neighbors, second_neighbors):
    left, right = st.columns(2)
    for col, item_id in ((left, first), (right, second)):
        row = metadata.loc[metadata.item_id == item_id].iloc[0]
        with col:
            st.markdown(f"**{row.item_name}** · {row.item_type} · {row.genre}")
            st.caption(f"{row.release_year} · popularity {row.popularity_score:.2f}")
    st.metric("Cosine similarity", f"{similarity:.3f}")
    common = set(x for x, _ in first_neighbors) & set(x for x, _ in second_neighbors)
    st.markdown(f"Common items among the two top-20 neighbor lists: **{len(common)}**")
