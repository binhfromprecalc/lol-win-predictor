import json
import pandas as pd
from src.features.features_engineering import compute_team_diffs

def build_dataset(timeline_path, winner):
    with open(timeline_path, "r") as f:
        timeline = json.load(f)

    rows = []

    for frame in timeline["info"]["frames"]:
        features = compute_team_diffs(frame)

        row = {
            **features,
            "minute": frame["timestamp"] // 60000,
            "win": 1 if winner == "blue" else 0
        }

        rows.append(row)

    return pd.DataFrame(rows)
