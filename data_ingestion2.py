import fastf1
import pandas as pd
import numpy as np
import time

# Cache Configuration
fastf1.Cache.enable_cache('f1_cache') 

def get_seconds(timedelta_val):
    """Converts Pandas/FastF1 Timedelta object to seconds (float)."""
    if pd.isna(timedelta_val):
        return np.nan
    return timedelta_val.total_seconds()

def process_session(session, season_year):
    """
    Extracts advanced features from a specific race session.
    Includes: Race Pace, Sector Ratios, Weather, and Qualifying Delta.
    """
    session_rows = []
    
    # 1. EVENT METADATA
    try:
        event_name = session.event.get('EventName', 'Unknown Event')
        round_number = session.event.get('RoundNumber', 0)
        location = session.event.get('Location', 'Unknown Location')
        season = season_year
        
        if hasattr(session.event, 'EventDate'):
             event_date = session.event.EventDate.strftime('%Y-%m-%d')
        else:
             event_date = str(session.event.get('EventDate', 'Unknown'))
    except Exception as e:
        print(f"  [Metadata Error] {e}")
        return []

    laps = session.laps
    results = session.results

    if results.empty: return []

    # =========================================================
    # 2. CALCULATE SESSION BENCHMARKS (GRID AVERAGES & WEATHER)
    # =========================================================
    # We need these benchmarks BEFORE iterating through drivers for comparison
    try:
        # A. Weather (Did it rain?)
        weather = session.weather_data
        rain_prob = 0.0
        track_temp = 30.0
        if not weather.empty:
            if 'Rainfall' in weather.columns:
                # If Rainfall > 0 on average, we consider it a wet/mixed race
                rain_prob = 1.0 if weather['Rainfall'].mean() > 0 else 0.0
            if 'TrackTemp' in weather.columns:
                track_temp = weather['TrackTemp'].mean()

        # B. Grid Sector Averages (For Sector Ratios)
        # Filter for representative quick laps from the whole grid
        all_quick_laps = laps.pick_quicklaps()
        grid_s1_avg = get_seconds(all_quick_laps['Sector1Time'].mean())
        grid_s2_avg = get_seconds(all_quick_laps['Sector2Time'].mean())
        grid_s3_avg = get_seconds(all_quick_laps['Sector3Time'].mean())

        # C. Qualifying Data (For Quali Delta)
        # We attempt to load the corresponding 'Q' session
        pole_seconds = None
        try:
            session_q = fastf1.get_session(season, round_number, 'Q')
            session_q.load(telemetry=False, weather=False, messages=False)
            # Get the absolute fastest time recorded in the session (Pole Position)
            pole_time = session_q.laps.pick_quicklaps()['LapTime'].min()
            pole_seconds = get_seconds(pole_time)
        except:
            pole_seconds = None # If loading fails, we will use a grid-based fallback later

    except Exception as e:
        print(f"  [Warning] Benchmark calculation failed: {e}")
        grid_s1_avg, grid_s2_avg, grid_s3_avg = np.nan, np.nan, np.nan
        rain_prob, track_temp = 0.0, 30.0

    # =========================================================
    # 3. ITERATE PER DRIVER
    # =========================================================
    for driver_code, res in results.iterrows():
        # Basic Info
        driver_name = res.get("FullName", "Unknown")
        driver_number = res.get("DriverNumber", "Unknown")
        team_name = res.get("TeamName", "Unknown")
        grid_position = res.get("GridPosition", np.nan)
        final_pos = res.get("Position", np.nan)
        points = res.get("Points", 0.0)
        status = res.get("Status", "Unknown")
        total_time_sec = get_seconds(res.get("Time", np.nan))
        
        # Collect Driver Laps
        try:
            drv_laps = laps.pick_driver(driver_code)
        except:
            drv_laps = pd.DataFrame() 
        
        # Initialize Calculated Features
        clean_air_pace = np.nan
        s1_ratio, s2_ratio, s3_ratio = 1.0, 1.0, 1.0
        quali_delta = np.nan
        avg_lap_time = np.nan
        best_lap_time = np.nan
        std_lap_time = np.nan
        clean_avg_lap_time = np.nan
        laps_completed = 0
        n_pitstops = 0
        n_stints = 0
        compounds_used = ""

        if not drv_laps.empty:
            laps_completed = len(drv_laps)
            
            # --- Basic Timing Metrics ---
            avg_lap_time = get_seconds(drv_laps["LapTime"].mean())
            best_lap_time = get_seconds(drv_laps["LapTime"].min())
            
            # Filter for valid racing laps
            quick_laps = drv_laps.pick_quicklaps()
            
            if not quick_laps.empty:
                clean_avg_lap_time = get_seconds(quick_laps["LapTime"].mean())
                std_lap_time = get_seconds(quick_laps["LapTime"].std())
                
                # --- NEW FEATURE: Clean Air Pace (Top 5 Laps) ---
                # Average of the 5 fastest laps (Potential Pace)
                best_5 = quick_laps.nsmallest(5, 'LapTime')
                clean_air_pace = get_seconds(best_5['LapTime'].mean())
                
                # --- NEW FEATURE: Sector Ratios ---
                # Comparison: Driver Sector / Grid Average (< 1.0 is better)
                s1 = get_seconds(quick_laps['Sector1Time'].mean())
                s2 = get_seconds(quick_laps['Sector2Time'].mean())
                s3 = get_seconds(quick_laps['Sector3Time'].mean())
                
                if grid_s1_avg > 0 and s1 > 0: s1_ratio = s1 / grid_s1_avg
                if grid_s2_avg > 0 and s2 > 0: s2_ratio = s2 / grid_s2_avg
                if grid_s3_avg > 0 and s3 > 0: s3_ratio = s3 / grid_s3_avg

            # --- NEW FEATURE: Quali Delta ---
            if pole_seconds:
                try:
                    # Look for driver's best lap in the 'Q' session loaded above
                    q_laps = session_q.laps.pick_driver(driver_code).pick_quicklaps()
                    if not q_laps.empty:
                        best_q = get_seconds(q_laps['LapTime'].min())
                        quali_delta = best_q - pole_seconds
                except:
                    pass
            
            # Fallback for Quali Delta if real time is missing
            # (Grid 1 = 0s, roughly +0.15s per grid position)
            if pd.isna(quali_delta) and not pd.isna(grid_position):
                quali_delta = max(0, (grid_position - 1) * 0.15)

            # Pit Stops & Stints
            pit_stops_data = drv_laps[drv_laps['PitInTime'].notna()]
            n_pitstops = len(pit_stops_data)
            
            # Calculate total pit time (Approximate)
            total_pit_time = np.nan
            if not pit_stops_data.empty and 'PitOutTime' in drv_laps.columns:
                 durations = drv_laps.loc[drv_laps['PitInTime'].notna(), 'PitOutTime'] - \
                             drv_laps.loc[drv_laps['PitInTime'].notna(), 'PitInTime']
                 total_pit_time = get_seconds(durations.sum())
            elif n_pitstops == 0:
                 total_pit_time = 0.0

            if 'Stint' in drv_laps.columns: n_stints = drv_laps['Stint'].nunique()
            if 'Compound' in drv_laps.columns:
                compounds = drv_laps['Compound'].dropna().unique()
                compounds_used = ", ".join(compounds)

        # Assemble Final Row
        row = {
            "season": season,
            "round": round_number,
            "event_name": event_name,
            "location": location,
            "event_date": event_date,
            "driver_code": driver_code,
            "driver_number": driver_number,
            "driver_name": driver_name,
            "team_name": team_name,
            "grid_position": grid_position,
            "final_position": final_pos,
            "status": status,
            "points": points,
            "total_time_sec": total_time_sec,
            
            # Basic Metrics
            "laps_completed": laps_completed,
            "avg_lap_time_sec": avg_lap_time,
            "best_lap_time_sec": best_lap_time,
            "std_lap_time_sec": std_lap_time,
            "clean_avg_lap_time_sec": clean_avg_lap_time,
            "n_pitstops": n_pitstops,
            "total_pit_time_sec": total_pit_time,
            "n_stints": n_stints,
            "compounds_used": compounds_used,
            
            # --- ADVANCED NEW FEATURES ---
            "RainProbability": rain_prob,
            "TrackTemp": track_temp,
            "CleanAirPace": clean_air_pace,
            "Sector1_Ratio": s1_ratio,
            "Sector2_Ratio": s2_ratio,
            "Sector3_Ratio": s3_ratio,
            "Quali_Delta_Pole": quali_delta
        }

        session_rows.append(row)
    
    return session_rows

def build_full_dataset(start_year, end_year):
    """Iterates through all years and rounds to build the dataset."""
    all_data = []

    for year in range(start_year, end_year + 1):
        print(f"\n=== Starting Season {year} ===")
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Error getting schedule for {year}: {e}")
            continue

        race_events = schedule[schedule['RoundNumber'] > 0]

        for _, event in race_events.iterrows():
            round_num = event['RoundNumber']
            event_name = event['EventName']
            
            print(f"  -> Round {round_num}: {event_name}...", end=" ")
            time.sleep(2) # Friendly pause for API
            
            try:
                session = fastf1.get_session(year, round_num, 'R')
                # Load with weather=True to get real rain data
                session.load(telemetry=False, weather=True, messages=False)
                
                rows = process_session(session, year)
                all_data.extend(rows)
                print(f"OK ({len(rows)} drivers)")
                
            except Exception as e:
                print(f"FAILED. Reason: {e}")
                
    return pd.DataFrame(all_data)

# --- Main Execution ---
if __name__ == "__main__":
    START_YEAR = 2021
    END_YEAR = 2025 # Adjust as needed

    print(f"Building full dataset ({START_YEAR}-{END_YEAR}) with advanced features...")
    
    full_df = build_full_dataset(START_YEAR, END_YEAR)
    
    if not full_df.empty:
        # Sort and Save
        full_df = full_df.sort_values(by=['season', 'round', 'final_position'])
        filename = f'f1_data_{START_YEAR}_{END_YEAR}.csv'
        full_df.to_csv(filename, index=False)
        print(f"\nSuccess! File saved to: {filename}")
        print(f"Columns: {full_df.columns.tolist()}")
    else:
        print("No data generated.")