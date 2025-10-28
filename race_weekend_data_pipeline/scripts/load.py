from pathlib import Path
import pandas as pd
import argparse
import glob
import sqlalchemy as sa

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TABLES = [
    'fact_laps',
    'fact_pitstops',
    'fact_weather',
    'dim_drivers'
]

SCHEMA_COLS = {
    'fact_laps' : [
        'session_key', 'driver_number', 'lap_number', 'stint', 'tyre_compound', 'lap_time_ms', 'sector_1_ms', 'sector_2_ms', 'sector_3_ms', 'tyre_life', 'is_inlap', 'is_outlap', 'is_pit', 'track_status', 'speed_trap_kph', 'is_accurate'
    ],
    'fact_pitstops': [
        'session_key', 'driver_number', 'lap_number', 'pit_time_ms', 'compound_out'
    ],
    'fact_weather': [
        'session_key', 'time_utc', 'air_temp_c', 'track_temp_c', 'humidity_pct',
        'pressure_hPa', 'wind_speed_mps', 'wind_dir_deg', 'rainfall'
    ],
    'dim_drivers': [
        'session_key', 'driver_number', 'driver_id', 'broadcast_name', 'full_name', 'team'
    ]
}

def sqlite_safe_df(df: pd.DataFrame, table: str) -> pd.DataFrame:
    # Coerce the df so that SQLite bindings counts/types are stable.
    if df.empty:
        return df
    
    df = df.copy()
    
    # Standardizing common key types.
    for c in ['session_key', 'driver_number']:
        if c in df.columns:
            df[c] = df[c].astype(str)

    if 'lap_number' in df.columns:
        df['lap_number'] = pd.to_numeric(df['lap_number'], errors='coerce').astype('Int64')

    # Booleans -> Int8 (0, 1)
    boolish = [c for c in ['is_inlap', 'is_outlap', 'is_pit', 'is_accurate'] if c in df.columns]
    for c in boolish:
        # cast via Int8 so NULLs remain possible; final cast to object avoids pandas NA gotchas on sqlite
        df[c] = df[c].astype('Int8').astype('object') # keep None possible.
    
    # Ensuring all expected columns exist in order.
    cols = SCHEMA_COLS.get(table)
    if cols: 
        missing = [c for c in cols if c not in df.columns]
        for c in missing:
            df[c] = None
        df = df[cols]
    
    # Replacing pandas NA/NaN with real None (DB Null).
    df = df.where(pd.notna(df), None)

    return df


def concat_parquets(table: str) -> pd.DataFrame:
    # Reading all data/processed/{session_key}/*.parquet files, union for outer columns.
    paths = glob.glob(str(PROCESSED_DATA_DIR / "*" / f"{table}.parquet"))
    if not paths: 
        return pd.DataFrame()
    dfs = []
    for path in paths: 
        try:
            dfs.append(pd.read_parquet(path))
        except Exception as e: 
            # keep going even if one file is corrupted.
            print(f"Error reading {path}: {e}")
            pass
    
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True, sort=False)

    # Normalizing a few common dtypes for portability. 
    if 'session_key' in df.columns: 
        df['session_key'] = df['session_key'].astype(str)
    if 'driver_number' in df.columns:
        df['driver_number'] = df['driver_number'].astype(str)
    if 'lap_number' in df.columns:
        df['lap_number'] = pd.to_numeric(df['lap_number'], errors='coerce').astype('Int64')

    return df

def write_table(df: pd.DataFrame, engine: sa.Engine, name: str, if_exists='replace'):
    if df.empty:
        print(f" [skip] No data for table {name}")
        return 0
    
    df = sqlite_safe_df(df, name)

    # Dialect-aware insert method.
    # - SQLite can be finicky with SQLAlchemy's multi-row expansion, when mixed nullables.
    #   method=None is safest 
    # - For Postgres, method='multi' is faster.

    dialect = engine.dialect.name
    is_sqlite = dialect == 'sqlite'
    method = None if is_sqlite else 'multi'
    
    df.to_sql(name, con=engine, if_exists=if_exists, index=False, method=method, chunksize=50000)
    print(f" [ok] Wrote table {name} with {len(df)} : rows")
    return len(df)

def create_indexes(engine: sa.Engine, only: list[str] | None = None):
    # Creating portable indexes for SQLite / Postgres.
    # Creating indexes only for tables that exist.
    # If `only` is provided, restrict index creation to that subset.

    insp = sa.inspect(engine)
    present = set(insp.get_table_names())
    if only: 
        target = present.intersection(set(only))
    else:
        target = present # all existing tables.
    
    with engine.begin() as conn:
        
        if 'fact_laps' in target:
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_fact_laps_key ON fact_laps (session_key, driver_number, lap_number);")
        if 'fact_pitstops' in target:
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_fact_pitstops_key ON fact_pitstops (session_key, driver_number, lap_number);")
        if 'fact_weather' in target:
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_fact_weather_key ON fact_weather (session_key);")
        if 'dim_drivers' in target:
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_dim_drivers_key ON dim_drivers (session_key, driver_number);")

    print(f" [ok] Indexes created for : {', '.join(sorted(target)) or '(none)'}")

def main():
    parser = argparse.ArgumentParser(description="Load processed parquets into database.")
    parser.add_argument(
        '--db', 
        default='sqlite:///data/f1.db', 
        help='SQLAlchemy URL, e.g. sqlite:///data/f1.db or postgresql+psycopg2://user:pass@host:5432/dbname'
    )
    parser.add_argument('--index-tables', nargs='*', default=None, help='Restrict index creation to these tables only. Default all existing tables.')
    parser.add_argument('--tables', nargs='+', default=TABLES, help='Tables to load, default all.')
    parser.add_argument('--append', action='store_true', help='Append rows instead of replacing tables.')

    args = parser.parse_args()

    engine = sa.create_engine(args.db)

    if_exists = 'append' if args.append else 'replace'
    for table in args.tables:
        df = concat_parquets(table)
        write_table(df, engine, table, if_exists=if_exists)
    
    create_indexes(engine, only=args.index_tables)

if __name__ == '__main__':
    main()