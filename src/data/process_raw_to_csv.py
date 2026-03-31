import os
import json
import pandas as pd

RAW_TIMELINE_DIR = "data/raw/timelines"
RAW_MATCH_DIR = "data/raw/matches"
OUTPUT_PATH = "data/processed/dataset.csv"


def get_team_gold(frame):
    blue_gold = 0
    red_gold = 0

    for pid, p in frame["participantFrames"].items():
        pid = int(pid)
        if pid <= 5:
            blue_gold += p["totalGold"]
        else:
            red_gold += p["totalGold"]

    return blue_gold, red_gold


def get_match_winner(match_data):
    for team in match_data["info"]["teams"]:
        if team["win"]:
            return team["teamId"]  # 100 = blue, 200 = red
    return None


def process_game(match_path, timeline_path):
    with open(match_path) as f:
        match_data = json.load(f)

    with open(timeline_path) as f:
        timeline_data = json.load(f)

    winner = get_match_winner(match_data)

    blue_kills = 0
    red_kills = 0
    blue_towers = 0
    red_towers = 0
    blue_dragons = 0
    red_dragons = 0
    # blue_grubs = 0
    # red_grubs = 0
    # blue_heralds = 0
    # red_heralds = 0
    # blue_barons = 0
    # red_barons = 0

    rows = []

    for frame in timeline_data["info"]["frames"]:
        timestamp = frame["timestamp"]
        minute = timestamp // 60000

        
        for event in frame.get("events", []):

            if event["type"] == "CHAMPION_KILL":
                killer = event.get("killerId", 0)

                if 1 <= killer <= 5:
                    blue_kills += 1
                elif 6 <= killer <= 10:
                    red_kills += 1

            elif event["type"] == "BUILDING_KILL":
                if event.get("buildingType") == "TOWER_BUILDING":
                    team = event.get("teamId")

                    if team == 200:
                        blue_towers += 1
                    elif team == 100:
                        red_towers += 1

            elif event["type"] == "ELITE_MONSTER_KILL":
                killer_team = event.get("killerTeamId")
                monster_type = event.get("monsterType")
                
                if monster_type == "DRAGON":
                    if killer_team == 100:
                        blue_dragons += 1
                    elif killer_team == 200:
                        red_dragons += 1
                
                # elif monster_type == "HORDE":
                #     if killer_team == 100:
                #         blue_grubs += 1
                #     elif killer_team == 200:
                #         red_grubs += 1
                
                # elif monster_type == "RIFTHERALD":
                #     if killer_team == 100:
                #         blue_heralds += 1
                #     elif killer_team == 200:
                #         red_heralds += 1
                
                # elif monster_type == "BARON_NASHOR":
                #     if killer_team == 100:
                #         blue_barons += 1
                #     elif killer_team == 200:
                #         red_barons += 1

        blue_gold, red_gold = get_team_gold(frame)


        row = {
            "minute": minute,
            "game_id": match_data["metadata"]["matchId"],
            "gold_diff": blue_gold - red_gold,
            "kills_diff": blue_kills - red_kills,
            "towers_diff": blue_towers - red_towers,
            "dragons_diff": blue_dragons - red_dragons,
            # "grubs_diff": blue_grubs - red_grubs,
            # "herald_diff": blue_heralds - red_heralds,
            # "baron_diff": blue_barons - red_barons,

            "win": 1 if winner == 100 else 0
        }

        rows.append(row)

    return rows




def build_dataset():
    all_rows = []

    match_files = os.listdir(RAW_MATCH_DIR)

    for file in match_files:
        match_path = os.path.join(RAW_MATCH_DIR, file)
        timeline_path = os.path.join(RAW_TIMELINE_DIR, file)

        if not os.path.exists(timeline_path):
            continue

        try:
            rows = process_game(match_path, timeline_path)
            all_rows.extend(rows)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    df = pd.DataFrame(all_rows)
    df["gold_diff_per_min"] = df["gold_diff"] / (df["minute"] + 1)
    df = df.drop(columns=["gold_diff"])
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dataset with {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_dataset()
