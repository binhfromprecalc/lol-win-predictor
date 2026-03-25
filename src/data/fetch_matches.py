import os
import time
import requests
from typing import List
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")

REGION = "americas"  
PLATFORM = "na1"     

BASE_URL = f"https://{REGION}.api.riotgames.com"

HEADERS = {
    "X-Riot-Token": API_KEY
}

RAW_DATA_DIR = "data/raw"

def riot_request(url: str, params=None, retries=3):
    for attempt in range(retries):
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code == 200:
            return response.json()

        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 1))
            print(f"Rate limited. Sleeping {retry_after}s...")
            time.sleep(retry_after)

        else:
            print(f"Error {response.status_code}: {response.text}")
            time.sleep(1)

    raise Exception(f"Failed request: {url}")


def get_puuid(game_name: str, tag_line: str) -> str:
    url = f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    data = riot_request(url)
    return data["puuid"]


def get_match_ids(puuid: str, start=0, count=20) -> List[str]:
    url = f"{BASE_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {
        "queue": 420,
        "start": start,
        "count": count
    }
    return riot_request(url, params)


def get_match(match_id: str):
    url = f"{BASE_URL}/lol/match/v5/matches/{match_id}"
    return riot_request(url)


def get_timeline(match_id: str):
    url = f"{BASE_URL}/lol/match/v5/matches/{match_id}/timeline"
    return riot_request(url)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        import json
        json.dump(data, f)


def fetch_and_store_matches(game_name: str, tag_line: str, num_matches=20):
    puuid = get_puuid(game_name, tag_line)
    print(f"PUUID: {puuid}")

    match_ids = get_match_ids(puuid, count=num_matches)

    for match_id in match_ids:
        print(f"Fetching {match_id}...")

        match_path = f"{RAW_DATA_DIR}/matches/{match_id}.json"
        timeline_path = f"{RAW_DATA_DIR}/timelines/{match_id}.json"

        if os.path.exists(match_path) and os.path.exists(timeline_path):
            print(f"Skipping {match_id}, already exists.")
            continue

        try:
            match_data = get_match(match_id)
            timeline_data = get_timeline(match_id)

            save_json(match_data, match_path)
            save_json(timeline_data, timeline_path)


        except Exception as e:
            print(f"Failed {match_id}: {e}")


if __name__ == "__main__":
    fetch_and_store_matches("binh", "NA1", num_matches=20)
