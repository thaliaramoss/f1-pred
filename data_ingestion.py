import os
import warnings

import fastf1
import numpy as np
import pandas as pd
from fastf1.core import DataNotLoadedError

warnings.filterwarnings("ignore")


def setup_fastf1_cache(cache_dir: str = "f1_cache"):
    """
    Create (if needed) and enable a local cache directory for FastF1.
    """
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    print(f"[INFO] FastF1 cache enabled at: {cache_dir}")


def _timedelta_to_seconds(series: pd.Series) -> pd.Series:
    """
    Convert a Series of pandas Timedelta to float seconds.
    """
    if series.isna().all():
        return pd.Series(np.nan, index=series.index)
    return series.dt.total_seconds()


def collect_f1_race_dataset(
    seasons=None,
    save_path: str | None = "f1_race_dataset.csv",
) -> pd.DataFrame:
    """
    Collect a rich race-level dataset for multiple seasons.

    Each row = one driver in one race.

    This version does NOT use get_event_schedule (which is unstable),
    it loads races by round number instead.
    """
    if seasons is None:
        seasons = [2021, 2022, 2023, 2024, 2025]

    # Number of rounds per season (you can adjust 2025 if needed)
    N_ROUNDS = {
        2021: 22,
        2022: 22,
        2023: 22,
        2024: 24,
        2025: 24,
    }

    all_rows = []

    for season in seasons:
        print(f"\n=== Season {season} ===")
        n_rounds = N_ROUNDS.get(season, 0)

        if n_rounds == 0:
            print(f"[WARN] No round information for season {season}, skipping.")
            continue

        # Iterate over all planned rounds for that season
        for round_number in range(1, n_rounds + 1):
            print(f" -> Loading race: Season {season}, Round {round_number}")
            try:
                session = fastf1.get_session(season, round_number, "R")
                session.load()
            except Exception as e:
                print(f"   [ERROR] Could not load session S{season} R{round_number}: {e}")
                continue

            # Try to get race results
            results = session.results
            if results is None or len(results) == 0:
                print(f"   [WARN] No results for S{season} R{round_number}, skipping session")
                continue

            # Try to load lap data – if it fails, we'll proceed with empty laps
            try:
                laps = session.laps
            except DataNotLoadedError:
                print(f"   [WARN] No lap data for S{season} R{round_number}, using empty laps")
                laps = pd.DataFrame()

            # Event metadata (if available)
            event = getattr(session, "event", None)
            if event is not None:
                event_name = event.get("EventName", None)
                location = event.get("Location", None)
                event_date = event.get("EventDate", None)
            else:
                event_name = None
                location = None
                event_date = None

            # Pit stops table (per stop)
            try:
                pits = session.get_pit_stops()
            except Exception:
                pits = pd.DataFrame()

            # Iterate driver by driver
            for _, res in results.iterrows():
                driver_code = res["Abbreviation"]
                driver_number = res.get("DriverNumber", None)
                driver_name = res.get("FullName", None)
                team_name = res.get("TeamName", None)

                # Laps for this driver
                if not laps.empty:
                    try:
                        drv_laps = laps.pick_driver(driver_code).copy()
                    except Exception:
                        drv_laps = pd.DataFrame()
                else:
                    drv_laps = pd.DataFrame()

                # Basic lap stats
                if not drv_laps.empty and "LapTime" in drv_laps.columns:
                    lap_times_sec = _timedelta_to_seconds(drv_laps["LapTime"])
                    avg_lap_time = lap_times_sec.mean()
                    best_lap_time = lap_times_sec.min()
                    std_lap_time = lap_times_sec.std()
                    n_laps = len(drv_laps)

                    # Stints / tyre info
                    if "Stint" in drv_laps.columns:
                        n_stints = drv_laps["Stint"].nunique()
                    else:
                        n_stints = 0

                    compounds_used = (
                        drv_laps["Compound"].dropna().unique().tolist()
                        if "Compound" in drv_laps.columns
                        else []
                    )

                    # Simple race pace proxy: average of "clean" laps
                    # (remove in/out laps and laps with pit time)
                    if {"PitInTime", "PitOutTime", "LapTime"}.issubset(drv_laps.columns):
                        clean_laps = drv_laps[
                            (~drv_laps["PitInTime"].notna())
                            & (~drv_laps["PitOutTime"].notna())
                        ]
                        clean_lap_times_sec = _timedelta_to_seconds(clean_laps["LapTime"])
                        clean_avg_lap_time = clean_lap_times_sec.mean()
                    else:
                        clean_avg_lap_time = np.nan
                else:
                    avg_lap_time = np.nan
                    best_lap_time = np.nan
                    std_lap_time = np.nan
                    n_laps = 0
                    n_stints = 0
                    compounds_used = []
                    clean_avg_lap_time = np.nan

                # Pit stop stats for this driver
                if not pits.empty and "Driver" in pits.columns:
                    drv_pits = pits[pits["Driver"] == driver_code]
                    n_pitstops = len(drv_pits)
                    if "PitTime" in drv_pits.columns:
                        total_pit_time = _timedelta_to_seconds(drv_pits["PitTime"]).sum()
                    else:
                        total_pit_time = np.nan
                else:
                    n_pitstops = 0
                    total_pit_time = np.nan

                # Result-level info
                grid_pos = res.get("GridPosition", np.nan)
                final_pos = res.get("Position", np.nan)
                status = res.get("Status", None)
                points = res.get("Points", np.nan)

                # Total race time
                total_time = res.get("Time", pd.NaT)
                total_time_sec = (
                    total_time.total_seconds()
                    if (pd.notna(total_time) and hasattr(total_time, "total_seconds"))
                    else np.nan
                )

                # Fastest lap time
                fastest_lap_time = res.get("FastestLapTime", pd.NaT)
                fastest_lap_time_sec = (
                    fastest_lap_time.total_seconds()
                    if (pd.notna(fastest_lap_time) and hasattr(fastest_lap_time, "total_seconds"))
                    else np.nan
                )

                row = {
                    # Meta
                    "season": season,
                    "round": round_number,
                    "event_name": event_name,
                    "location": location,
                    "event_date": event_date,

                    # Driver / team
                    "driver_code": driver_code,
                    "driver_number": driver_number,
                    "driver_name": driver_name,
                    "team_name": team_name,

                    # Result
                    "grid_position": grid_pos,
                    "final_position": final_pos,
                    "status": status,
                    "points": points,
                    "total_time_sec": total_time_sec,
                    "fastest_lap_time_sec": fastest_lap_time_sec,

                    # Race distance / laps
                    "laps_completed": n_laps,

                    # Lap time statistics
                    "avg_lap_time_sec": avg_lap_time,
                    "best_lap_time_sec": best_lap_time,
                    "std_lap_time_sec": std_lap_time,
                    "clean_avg_lap_time_sec": clean_avg_lap_time,  # race pace proxy

                    # Pit stops
                    "n_pitstops": n_pitstops,
                    "total_pit_time_sec": total_pit_time,

                    # Tyres / strategy
                    "n_stints": n_stints,
                    "compounds_used": ", ".join(compounds_used) if compounds_used else None,
                }

                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"\n[OK] Dataset saved with {len(df)} rows to: {save_path}")

    return df


if __name__ == "__main__":
    print("=== BUILDING F1 RACE DATASET ===")
    setup_fastf1_cache("f1_cache")
    df = collect_f1_race_dataset(seasons=[2021, 2022, 2023, 2024, 2025])
    print("\n[DONE] Shape:", df.shape)
