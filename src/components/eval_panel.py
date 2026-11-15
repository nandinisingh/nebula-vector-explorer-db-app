import plotly.express as px
import streamlit as st


def render(aggregate, evaluation, selected_user=None):
    metrics = aggregate["metrics"]
    st.subheader("Evaluation results")
    if selected_user:
        row = evaluation[evaluation.user_id == selected_user]
        if row.empty:
            st.info("No evaluation record for this user.")
            return
        row = row.iloc[0]
        metric_rows = [{"metric_name": m["metric_name"], "aggregate": m["metric_value"], "user": float(row[m["metric_name"]])} for m in metrics]
        st.plotly_chart(px.bar(__import__("pandas").DataFrame(metric_rows).melt("metric_name"), x="metric_name", y="value", color="variable", barmode="group"), use_container_width=True)
        st.dataframe(__import__("pandas").DataFrame(metric_rows), hide_index=True, use_container_width=True)
        truth = __import__("json").loads(row.ground_truth_items)
        pred = __import__("json").loads(row.top_k_predicted_items)
        hits = set(truth) & set(pred)
        st.markdown(f"**Reference items:** {len(truth)} · **Top-50 predictions:** {len(pred)} · **Overlap:** {len(hits)}")
        st.caption("Reference items: " + ", ".join(truth))
        st.caption("Predicted items: " + ", ".join(pred[:15]) + (" …" if len(pred) > 15 else ""))
    else:
        df = __import__("pandas").DataFrame(metrics)
        st.plotly_chart(px.bar(df, x="metric_name", y="metric_value", color="metric_name", hover_data=["description"]), use_container_width=True)
        st.dataframe(df.rename(columns={"metric_name": "Metric", "metric_value": "Value", "description": "Description"}), hide_index=True, use_container_width=True)
