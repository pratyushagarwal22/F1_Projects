from pathlib import Path
import pandas as pd 
import numpy as np
from datetime import datetime

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

def to_ms(s: pd.Series) -> pd.Series:
    # Converting a pandas Series of timedelta to milliseconds.
    return s.dt.total_seconds().mul(1000)

def derive_stints(laps: pd.DataFrame) -> pd.Series:
    # Sorting per driver. Incrementing when compound changes. (NaN -> UNKNOWN)
    x = laps.copy()
    x['Compund'] = x['Compound'].fillna('UNKNOWN')
    changed = x['Compound'].ne(x['Compound'].shift(1)) | x['DriverNumber'].ne(x['DriverNumber'].shift())
    return changed.groupby(x['DriverNumber']).cumsum().astype('int64')

def build_fact_laps(laps: pd.DataFrame, session_key: str) -> pd.DataFrame:
    laps = laps.sort_values(['DriverNumber', 'LapNumber']).reset_index(drop=True).copy()
    
    # Converting timedelta columns to ms -> create even if source columns missing to keep schema stable.
    def maybe_ms(col_src, col_dst): 
        laps[col_dst] = to_ms(laps[col_src]) if col_src in laps.columns and pd.api.types.is_timedelta64_dtype(laps[col_src]) else np.nan
    
    maybe_ms('LapTime', 'lap_time_ms')
    maybe_ms('Sector1Time', 'sector1_time_ms')
    maybe_ms('Sector2Time', 'sector2_time_ms')
    maybe_ms('Sector3Time', 'sector3_time_ms')

    # Flags for pit stops.
    laps['is_inlap'] = laps.get('PitInTime').notna() if 'PitInTime' in laps.columns else False
    laps['is_outlap'] = laps.get('PitOutTime').notna() if 'PitOutTime' in laps.columns else False
    laps['is_pit'] = laps['is_inlap'] | laps['is_outlap']

    # Deriving stint numbers.
    laps['stint'] = derive_stints(laps)

    # select / rename columns for fact table.
    out = laps.rename(columns = {
        'DriverNumber' : 'driver_number', 
        'LapNumber' : 'lap_number',
        'Compound' : 'tyre_compound', 
        'TrackStatus' : 'track_status',
        'TyreLife' : 'tyre_life',
        'IsAccurate' : 'is_accurate',
        'SpeedST' : 'speed_trap_kph'
    })[[
        'driver_number', 'lap_number', 'tyre_compound', 'stint', 'track_status', 'tyre_life', 'is_accurate', 'speed_trap_kph', 'lap_time_ms', 'sector1_time_ms', 'sector2_time_ms', 'sector3_time_ms', 'is_inlap', 'is_outlap', 'is_pit'
    ]]
    out['session_key'] = session_key
    return out

def build_fact_pitstops(laps: pd.DataFrame, session_key: str) -> pd.DataFrame:
    """ Pairing in-lap PitInTime with the correct PitOutTime per driver.
        - prefer the same row PitOutTime if present.
        - else use the NEXT rows PitOutTime (out-lap) for that driver.
    Also fetch the out-lap tyre compound ("compound_out") from the next row. 
    """

    out_cols = ['driver_number', 'lap_number', 'pit_time_ms', 'compound_out', 'session_key']
    required = {'DriverNumber', 'LapNumber', 'PitInTime', 'PitOutTime', 'Compound'}

    # Guard -> must have all required columns.
    if not required.issubset(laps.columns):
        return pd.DataFrame(columns=out_cols)
    
    df = (laps[list(required)].sort_values(['DriverNumber', 'LapNumber'], kind='mergesort').copy())

    # Ensure timedeltas (FastF1 usually gives timedeltas already.)
    for c in ['PitInTime', 'PitOutTime']:
        if not pd.api.types.is_datetime64_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors='coerce')
    
    # For each driver capture the NEXT rows PitOutTime and Compound (out-lap info)
    df['next_out_time'] = df.groupby('DriverNumber')['PitOutTime'].shift(-1)
    df['next_compound'] = df.groupby('DriverNumber')['Compound'].shift(-1)

    in_mask = df['PitInTime'].notna()

    # Choose out time: prefer same row PitOutTime if present, else NEXT rows PitOutTime
    out_time = df['PitOutTime'].where(in_mask) # same row if present
    out_time = out_time.fillna(df['next_out_time'].where(in_mask)) # else next rows out time

    # Duration
    pit_dt = out_time - df['PitInTime']
    pit_ms = to_ms(pit_dt)

    # Setting reasonable bounds for pit durations (0-5 minutes)
    valid = in_mask & out_time.notna() & (pit_ms > 0) & (pit_ms < 5 * 60 * 1000)

    result = pd.DataFrame({
        'driver_number' : df.loc[valid, 'DriverNumber'].values,
        'lap_number' : df.loc[valid, 'LapNumber'].values,
        'pit_time_ms' : pit_ms.loc[valid].values,
        # prefer next row compound else fallback to same row if needed.
        'compound_out' : df.loc[valid, 'next_compound'].fillna(df.loc[valid, 'Compound']).values
    })
    result['session_key'] = session_key

    return result[out_cols]

def build_fact_weather(session_dir: Path, session_key: str, session_start_utc: datetime | None = None) -> pd.DataFrame | None:
    w_path = session_dir / 'weather.parquet'
    if not w_path.exists():
        return None
    
    w = pd.read_parquet(w_path).copy()

    # Normalizing Column Names present in FastF1 weather data.
    rename = {
        'AirTemp' : 'air_temp_c', 
        'TrackTemp' : 'track_temp_c',
        'Humidity' : 'humidity_pct', 
        'Pressure' : 'pressure_hPa', 
        'WindSpeed' : 'wind_speed_kph', 
        'WindDirection' : 'wind_direction_deg', 
        'Rainfall' : 'rainfall',
        'Time' : 'session_time'
    }

    w.rename(columns={k: v for k, v in rename.items() if k in w.columns}, inplace=True)

    # If time is index, materialize it as 'session_time' (timedelta or datetime)
    if 'session_time' not in w.columns and isinstance(w.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        idx_name = w.index.name or 'session_time'
        w = w.reset_index().rename(columns={idx_name: 'session_time'})

    # Ensure that session_time exists.
    if 'session_time' not in w.columns:
        # Nothing to align by, return empty with stable schema.
        out = pd.DataFrame(columns=[
            'session_key', 'session_time', 'time_utc', 'air_temp_c', 'track_temp_c', 'humidity_pct', 'pressure_hPa', 'wind_speed_mps', 'wind_direction_deg', 'rainfall'
        ])

        return out
    
    # If user wants absolute UTC and we have a timedelta column + a start time, compute it.
    time_utc = pd.NaT
    if session_start_utc is not None:
        # try to add timedeltas; if session_time is already datetime, just use it.
        if pd.api.types.is_timedelta64_dtype(w['session_time']):
            time_utc = pd.to_datetime(session_start_utc) + w['session_time']
        elif pd.api.types.is_datetime64_any_dtype(w['session_time']):
            time_utc = pd.to_datetime(w['session_time'], utc=True)
    w['time_utc'] = time_utc

    # Wind speed -> FastF1 ususally in kph, convert to m/s if present.
    if 'wind_speed_kph' in w.columns:
        w['wind_speed_mps'] = pd.to_numeric(w['wind_speed_kph'], errors='coerce').div(3.6)
    elif 'wind_speed_mps' not in w.columns and 'WindSpeed' in w.columns:
        w['wind_speed_mps'] = pd.to_numeric(w['WindSpeed'], errors='coerce')
    # else: maybe missing, leave as NaN columns below.

    # Making sure dtypes are correct.
    for c in ['air_temp_c', 'track_temp_c', 'humidity_pct', 'pressure_hPa', 'wind_direction_deg', 'rainfall']:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors='coerce')
    
    # Order and dedupe by time.
    w = w.sort_values('session_time', kind='mergesort').drop_duplicates(subset=['session_time'], keep='last')

    # Returning stable output schema (same columns in all cases)
    w['session_key'] = session_key
    out_cols = [
        'session_key',
        'session_time', 
        'time_utc',
        'air_temp_c',
        'track_temp_c', 
        'humidity_pct',
        'pressure_hPa',
        'wind_speed_mps',
        'wind_direction_deg',
        'rainfall'
    ]

    # Ensure all columns exist.
    for c in out_cols:
        if c not in w.columns:
            w[c] = np.nan if c != 'session_key' else session_key

    return w[out_cols]

def build_dim_drivers(session_dir: Path, session_key: str) -> pd.DataFrame | None:
    d_path = session_dir / 'drivers.parquet'
    if not d_path.exists():
        return None
    
    d = pd.read_parquet(d_path).copy()

    cols = ['driver_number', 'driver_id', 'broadcast_name', 'full_name', 'team']

    # Accept either already normalized or raw names.
    rename = {
        'DriverNumber' : 'driver_number',
        'Abbreviation' : 'driver_id', 
        'BroadcastName' : 'broadcast_name',
        'FullName' : 'full_name',
        'TeamName' : 'team'
    }

    for k, v in rename.items():
        if k in d.columns and v not in d.columns:
            d.rename(columns={k: v}, inplace=True)
    
    d = d[[c for c in cols if c in d.columns]].drop_duplicates()
    d['session_key'] = session_key
    return d

# [Optional] Telemetry data is huge. Transformation could be added here later.

def transform_one_session(session_key: str):
    in_dir = RAW_DIR / session_key
    out_dir = PROCESSED_DIR / session_key

    if not in_dir.exists():
        print(f" [skip] {session_key} : Raw Directory Missing.")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # LAPS (required) -> fact_laps, fact_pitstops
    laps_path = in_dir / 'laps.parquet' 
    if not laps_path.exists():
        print(f" [skip] {session_key} : Laps Data Missing.")
        return
    laps = pd.read_parquet(laps_path)

    # Building Fact and Dim tables.
    fact_laps = build_fact_laps(laps, session_key)
    fact_pits = build_fact_pitstops(laps, session_key)
    fact_weather = build_fact_weather(in_dir, session_key)
    dim_drivers = build_dim_drivers(in_dir, session_key)

    # Saving the outputs as parquet in the processed directory.
    fact_laps.to_parquet(out_dir / 'fact_laps.parquet', index=False)
    fact_pits.to_parquet(out_dir / 'fact_pitstops.parquet', index=False)
    if fact_weather is not None and len(fact_weather):
        fact_weather.to_parquet(out_dir / 'fact_weather.parquet', index=False)
    if dim_drivers is not None and len(dim_drivers):
        dim_drivers.to_parquet(out_dir / 'dim_drivers.parquet', index=False)

    print(f" [ok] Transformed Session: {session_key} -> Laps: {len(fact_laps)}, Pits: {len(fact_pits)} "
          f" Weather: {0 if fact_weather is None else len(fact_weather)} "
          f" Drivers: {0 if dim_drivers is None else len(dim_drivers)}")

def transform_season():
    if not RAW_DIR.exists():
        print(" [warn] Raw Directory not found. Nothing to transform.")
        return
    
    for session_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        transform_one_session(session_dir.name)

if __name__ == "__main__":
    transform_season()