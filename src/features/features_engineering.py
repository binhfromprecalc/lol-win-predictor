import pandas as pd

def compute_team_diffs(frame):

    blue_gold = sum(p["totalGold"] for p in frame["participantFrames"].values() if int(p["participantId"]) <= 5)
    red_gold = sum(p["totalGold"] for p in frame["participantFrames"].values() if int(p["participantId"]) > 5)

    gold_diff = blue_gold - red_gold

    return {
        "gold_diff": gold_diff,
    }
