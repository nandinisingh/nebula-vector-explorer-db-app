import pandas as pd


def results_frame(results, metadata):
    scores = pd.DataFrame(results, columns=["item_id", "similarity"])
    frame = scores.merge(metadata[["item_id", "item_name", "item_type", "genre"]], on="item_id", how="left")
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame["distance"] = 1 - frame.similarity
    return frame
