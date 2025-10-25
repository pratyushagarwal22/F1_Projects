from pathlib import Path
import fastf1 as f1
import pandas as pd

# Since existing cache is as F1_Projects/f1_cache
# This script lives in F1_Projects/race_weekend_data_pipeline/scripts/
# So we need to go up two levels to set the cache correctly

CACHE_DIR = Path(__file__).resolve().parents[2] / "f1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True) # no operation if directory exists.

f1.Cache.enable_cache(str(CACHE_DIR))

def main():
    season = 2024
    schedule = f1.get_event_schedule(season, include_testing=False)
    cols = ['RoundNumber', 'EventName', 'Country', 'Location', 'EventDate']
    print(schedule[cols].to_string(index=False))
    print(f"Cache in use: {CACHE_DIR}")

if __name__ == "__main__":
    main()
