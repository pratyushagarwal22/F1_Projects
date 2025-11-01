from pathlib import Path
import pandas as pd
import json
import argparse
from datetime import datetime
import sqlalchemy as sa

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "reports" / "quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_QUERIES = {
    # row counts - per table
    'row_counts' : """
            SELECT 'fact_laps' as tbl, COUNT(*) as rows FROM fact_laps
            UNION ALL
            SELECT 'fact_weather', COUNT(*) as rows from fact_weather
            UNION ALL
            SELECT 'dim_drivers', COUNT(*) as rows from dim_drivers
    """,
    # --- per session, fact_laps rows
    'laps_per_session' : """
            SELECT session_key, COUNT(*) as laps 
            FROM fact_laps
            WHERE (:pattern IS NULL or session_key LIKE :pattern)
            GROUP BY session_key
            ORDER BY session_key ASC
    """,

    # --- duplicate key check for fact_laps
    'dup_keys_fact_laps' : """
            SELECT session_key, driver_number, lap_number, COUNT(*) as dup_count
            FROM fact_laps
            WHERE (:pattern IS NULL or session_key LIKE :pattern)
            GROUP BY session_key, driver_number, lap_number
            HAVING COUNT(*) > 1
            ORDER BY session_key, driver_number, lap_number
    """,

    # --- negative or unrealistic times for laps
    'bad_lap_times' : """
            SELECT session_key, driver_number, lap_number, lap_time_ms
            FROM fact_laps
            WHERE (:pattern is NULL or session_key LIKE :pattern)
                AND (lap_time_ms is NULL or lap_time_ms <= 0 or lap_time_ms > 3*60*1000)
            ORDER BY session_key, driver_number, lap_number
    """,

    # --- stint monotonicity violations (stint should never decrease within a driver)
    'stint_monotonic_violations' : """
        WITH ordered AS (
            SELECT session_key, driver_number, lap_number, stint, 
                LAG(stint) OVER (PARTITION BY session_key, driver_number ORDER BY lap_number) as prev_stint
            FROM fact_laps
            WHERE (:pattern is NULL or session_key LIKE :pattern)
        )

        SELECT session_key, driver_number, lap_number, prev_stint, stint
        FROM ordered
        WHERE prev_stint IS NOT NULL AND stint < prev_stint
        ORDER BY session_key, driver_number, lap_number
    """,

    # --- driver dimension coverage: do all laps have a matching dim driver?
    'dim_coverage' : """
        SELECT fl.session_key, fl.driver_number, COUNT(*) as laps, 
            SUM(
                CASE WHEN dd.driver_number is NULL THEN 1 ELSE 0 END
            ) as laps_without_dim_driver
        FROM fact_laps fl
        LEFT JOIN dim_drivers dd ON fl.session_key = dd.session_key AND fl.driver_number = dd.driver_number
        WHERE (:pattern is NULL or fl.session_key LIKE :pattern)
        GROUP BY fl.session_key, fl.driver_number
        HAVING SUM(CASE WHEN dd.driver_number is NULL THEN 1 ELSE 0 END) > 0
        ORDER BY fl.session_key, fl.driver_number
    """,

    # --- weather coverage per session (simple completeness signal)
    'weather_per_session' : """
        SELECT session_key, COUNT(*) as weather_points
        FROM fact_weather
        WHERE (:pattern is NULL or session_key LIKE :pattern)
        GROUP BY session_key
        ORDER BY session_key ASC
    """
}

def tables_present(engine: sa.Engine) -> set[str]:
    return set(sa.inspect(engine).get_table_names())

def run_query(engine: sa.Engine, sql: str, pattern: str | None = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(sa.text(sql), conn, params={"pattern" : pattern})
    
def write_artifacts(tag: str, df: pd.DataFrame):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = OUT_DIR / f"{tag}__{ts}"
    # CSV for eyeballing the data
    df.to_csv(f"{base}.csv", index=False)
    # JSON summary for programmatic ingestion.
    with open(f"{base}.json", "w") as f:
        json.dump(json.loads(df.to_json(orient='records')), f, indent=2)
    print (f" [ok] Wrote quality report artifacts for {tag}: {base}.(csv|json)")


def soft_assert(name: str, df: pd.DataFrame, should_be_empty: bool):
    ok = df.empty if should_be_empty else not df.empty
    status = "PASS" if ok else "FAIL"

    print(f" [{status}] quality check -> {name} : {len(df)} rows")
    return ok

def compute_summary(
        row_counts: pd.DataFrame, 
        dup_keys: pd.DataFrame, 
        bad_times: pd.DataFrame,
        stint_viol : pd.DataFrame,
        dim_cov : pd.DataFrame, 
        laps_per_sess : pd.DataFrame,
        weather_per_sess : pd.DataFrame
) -> dict:
    return {
        'totals' : {r['tbl'] : int(r['rows']) for _, r in row_counts.iterrows()},
        'dup_keys_fact_laps' : len(dup_keys),
        'bad_lap_times' : len(bad_times),
        'stint_monotonic_violations' : len(stint_viol),
        'dim_coverage_violations' : len(dim_cov),
        "sessions_with_laps": int(laps_per_sess["session_key"].nunique()) if not laps_per_sess.empty else 0,
        "sessions_with_weather": int(weather_per_sess["session_key"].nunique()) if not weather_per_sess.empty else 0
    }

def main():
    parser = argparse.ArgumentParser(description='Run data quality checks and generate reports.')
    parser.add_argument('--db', default='sqlite:///data/f1.db', help='Database connection string.')
    parser.add_argument('--session-like', default=None, help='Optional SQL LIKE pattern to filter session_key, ex: 2024-%-R, 2024-20-%')
    parser.add_argument(
        '--fail-on', 
        nargs='*', 
        default=['dup_keys', 'stint_mono'], 
        choices=['dup_keys', 'bad_times', 'stint_mono', 'dim_cov'],
        help='Which checks should cause non zero exit code if they have rows'
    )

    args = parser.parse_args()

    engine = sa.create_engine(args.db)
    pattern = args.session_like

    # 1) Running all queries
    row_counts = run_query(engine, DEFAULT_QUERIES['row_counts'], pattern=None)
    laps_per_sess = run_query(engine, DEFAULT_QUERIES['laps_per_session'], pattern)
    dup_keys = run_query(engine, DEFAULT_QUERIES['dup_keys_fact_laps'], pattern)
    bad_times = run_query(engine, DEFAULT_QUERIES['bad_lap_times'], pattern)
    stint_viol = run_query(engine, DEFAULT_QUERIES['stint_monotonic_violations'], pattern)
    dim_cov = run_query(engine, DEFAULT_QUERIES['dim_coverage'], pattern)
    weather_per_sess = run_query(engine, DEFAULT_QUERIES['weather_per_session'], pattern)

    # 2) Writing artifacts
    write_artifacts('row_counts', row_counts)
    write_artifacts('laps_per_session', laps_per_sess)
    write_artifacts('dup_keys_fact_laps', dup_keys)
    write_artifacts('bad_lap_times', bad_times)
    write_artifacts('stint_monotonic_violations', stint_viol)
    write_artifacts('dim_coverage_violations', dim_cov)
    write_artifacts('weather_per_session', weather_per_sess)

    # 3) Console PASS / FAIL summaries
    pass_dup = soft_assert('Duplicate Keys (fact_laps)', dup_keys, should_be_empty=True)
    pass_time = soft_assert('Bad lap_time_ms (<=0 or >3min)', bad_times, should_be_empty=True)
    pass_stint = soft_assert("Stint monotonicity", stint_viol, should_be_empty=True)
    pass_dim   = soft_assert("Dim coverage (missing driver rows)", dim_cov, should_be_empty=True)

    # 4) Writing a single summary JSON
    summary = compute_summary(row_counts, dup_keys, bad_times, stint_viol, dim_cov, laps_per_sess, weather_per_sess)
    OUT_DIR.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    print(f" [ok] Wrote overall quality summary : {OUT_DIR / 'summary.json'}")

    # 5) Exit code for CI/CD 
    fail = False
    if 'dup_keys' in args.fail_on and not pass_dup: fail = True
    if 'bad_times' in args.fail_on and not pass_time: fail = True
    if 'stint_mono' in args.fail_on and not pass_stint: fail = True
    if 'dim_cov' in args.fail_on and not pass_dim: fail = True
    raise SystemExit(1 if fail else 0)

if __name__ == "__main__":
    main()