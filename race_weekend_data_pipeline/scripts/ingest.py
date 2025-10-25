from pathlib import Path
import pandas as pd
import fastf1 as f1
import argparse
import json
from tqdm import tqdm

# Since existing cache is as F1_Projects/f1_cache. Using the shared cache (../..) 2 levels up.
CACHE_DIR = Path(__file__).resolve().parents[2] / "f1_cache"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DEFAULT_SESSIONS = ("FP1", "FP2", "FP3", "SQ", "S", "Q", "R")

# Normalizing the session names to match FastF1's expected inputs.
def normalize_session_name(name: str) -> str:
    # Cannonicalize session names and common aliases for FastF1.
    n = name.strip().upper()
    if n in ("S", "SPRINT"): return "S"
    if n in ("SQ", "SS", "SPRINT SHOOTOUT", "SPRINT QUALIFYING"): return "SQ"
    if n in ("Q", "QUALI", "QUALIFYING"): return "Q"
    if n in ("R", "RACE"): return "R"
    if n in ("FP1", "FREE PRACTICE 1", "PRACTICE 1", "P1"): return "FP1"
    if n in ("FP2", "FREE PRACTICE 2", "PRACTICE 2", "P2"): return "FP2"
    if n in ("FP3", "FREE PRACTICE 3", "PRACTICE 3", "P3"): return "FP3"
    return n

# Creating the session keys for easy lookup. 
def session_key(season: int, round_num: int, session: str) -> str:
    s = normalize_session_name(session)
    code_map = {
        "FP1" : "F1",
        "FP2" : "F2",
        "FP3" : "F3",
        "SQ"  : "SQ",
        "S"   : "S",
        "Q"   : "Q",
        "R"   : "R"
    }
    code = code_map.get(s, s)
    return f"{season}_{round_num:02d}_{code}"

# Saving json output.
def save_json(path: Path, obj: dict): 
    path.write_text(json.dumps(obj, indent=2))

def ingest_season(season: int=2024, sessions=DEFAULT_SESSIONS, include_telemetry=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    f1.Cache.enable_cache(str(CACHE_DIR))

    schedule = f1.get_event_schedule(season, include_testing=False)
    for _, event in schedule.iterrows():
        rnd = int(event['RoundNumber'])

        for s in sessions:
            s_norm = normalize_session_name(s)
            try: 
                ses = f1.get_session(season, rnd, s_norm)
            except Exception:
                # Invalid session type for this round, skip.
                continue

            try:
                ses.load(telemetry=False if include_telemetry else False, laps=True, weather=True, messages=False)
            except Exception as e:
                print(f"Failed to load session: {season} R{rnd} {s_norm}: {e}")
                continue
                
            skey = session_key(season, rnd, s_norm)
            OUTDIR = RAW_DIR / skey
            OUTDIR.mkdir(parents=True, exist_ok=True)

            # Getting Laps Data.
            if hasattr(ses, 'laps') and ses.laps is not None and len(ses.laps) > 0:
                ses.laps.to_parquet(OUTDIR / "laps.parquet")
            
            # Getting Weather Data.
            weather_df = None
            if hasattr(ses, 'weather_data') and ses.weather_data is not None:
                try:
                    weather_df = pd.DataFrame(ses.weather_data).copy()
                    if weather_df.index.name:
                        weather_df = weather_df.reset_index().rename(columns={weather_df.index.name: 'Time'})
                    if len(weather_df) > 0:
                        weather_df.to_parquet(OUTDIR / 'weather.parquet')
                except Exception:
                    weather_df = None 
            
            # Saving Session Results Data.
            results_df = None
            if hasattr(ses, 'results') and ses.results is not None:
                try:
                    if len(ses.results) > 0:
                        results_df = ses.results
                        results_df.to_parquet(OUTDIR / 'results.parquet')
                except Exception: 
                    results_df = None

            laps_n = len(ses.laps) if hasattr(ses, 'laps') and ses.laps is not None else 0
            weather_n = len(weather_df) if weather_df is not None else 0
            results_n = len(results_df) if results_df is not None else 0
            print(f" [ok] Loaded Session: {skey} -> Laps: {laps_n}, Weather: {weather_n}, Results: {results_n}")

            # Creating the Drivers dim. 
            drv_rows = []
            for drv in ses.drivers:
                drv_info = ses.get_driver(drv)
                drv_rows.append({
                    'driver_number' : drv_info.get('DriverNumber', None),
                    'driver_id' : drv_info.get('Abbreviation', None),
                    'broadcast_name' : drv_info.get('BroadcastName', None),
                    'full_name' : drv_info.get('FullName', None),
                    'team' : drv_info.get('TeamName', None)
                })
            
            if drv_rows:
                pd.DataFrame(drv_rows).drop_duplicates().to_parquet(OUTDIR / 'drivers.parquet')
            
            # Telemetry Data [Optional] can be large. 
            if include_telemetry:
                for drv in tqdm(ses.drivers, desc=f"{skey} Telemetry", leave=False):
                    laps_d = ses.laps.pick_driver(drv)
                    if laps_d is None or laps_d.empty:
                        continue
                    parts = []
                    for _, lap in laps_d.iterlaps():
                        try:
                            tel = lap.get_telemetry()
                            tel['LapNumber'] = lap['LapNumber']
                            parts.append(tel)
                        except Exception: 
                            continue
                    if parts:
                        pd.concat(parts, ignore_index=True).to_parquet(OUTDIR / f"telemetry_{drv}.parquet")
            
            # Saving Metadata about the session.
            meta = {
                'session_key' : skey,
                'season' : season,
                'round_number' : rnd,
                'event_name' : event.get('EventName', None),
                'country' : event.get('Country', None),
                'circuit' : event.get('Location', None),
                'session_name' : s_norm,
                'event_date' : str(event.get('EventDate', None).date()) if pd.notna(event.get('EventDate', None)) else None,
                'include_telemetry' : include_telemetry
            }

            save_json(OUTDIR / 'session_meta.json', meta)
            
            print(f"[ok] Ingested Raw Data: {skey} -> {OUTDIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw FastF1 data to parquet files.")
    parser.add_argument('--season', type=int, default=2024)
    parser.add_argument('--sessions', nargs='+', default = list(DEFAULT_SESSIONS))
    parser.add_argument('--telemetry', action='store_true', help='Include per lap telemetry data (can be large).')
    args = parser.parse_args()

    ingest_season(season=args.season, sessions=tuple(args.sessions), include_telemetry=args.telemetry)