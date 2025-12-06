import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)

RACE_DATA_PATH = "f1_race_dataset.csv"

df_races = pd.read_csv(RACE_DATA_PATH)

print(f"Loaded {len(df_races)} rows")

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build driver-race level features based on past performance.

    Each output row = one driver in one race,
    using ONLY information from races BEFORE that race (no data leakage).
    """
    df = df.copy()
    
    # --- 1. Basic column alignment ---
    # We assume the input has at least these columns:
    # 'season', 'round', 'event_name', 'location',
    # 'driver_code', 'driver_name', 'team_name',
    # 'final_position', 'grid_position', 'status', 'points'
    
    # Ensure numeric types where needed
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    
    # For safety, ensure positions are numeric
    df["final_position_num"] = pd.to_numeric(df["final_position"], errors="coerce")
    df["grid_position_num"] = pd.to_numeric(df["grid_position"], errors="coerce")
    
    # Some datasets may have NaNs (e.g., DNS/DNF); we'll handle later
    
    # --- 2. Create a chronological ordering key ---
    # If 'event_date' exists, we can use it; otherwise, rely on (season, round)
    if "event_date" in df.columns:
        # convert to datetime if needed
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df = df.sort_values(["season", "event_date", "driver_code"])
    else:
        # fallback: sort by season and round
        df = df.sort_values(["season", "round", "driver_code"])
    
    # A helper column for "time order" (just to make it explicit)
    df["time_order"] = (
        df["season"].astype(int) * 100
        + df.groupby("season")["round"].rank(method="dense").astype(int)
    )
    
    print("Unique seasons in input:", sorted(df["season"].dropna().unique()))
    print("Total distinct races:", df.groupby(["season", "round"]).ngroups)
    
    # --- 3. Iterate drivers and build features ---
    features = []
    processed_drivers = 0
    rows_per_season = {}
    
    # we'll use driver_code as unique driver ID (you can switch to driver_name if you prefer)
    for driver in df["driver_code"].unique():
        df_driver = (
            df[df["driver_code"] == driver]
            .sort_values("time_order")
            .reset_index(drop=True)
        )
        
        for idx in range(len(df_driver)):
            # History up to BEFORE this race
            history = df_driver.iloc[:idx]
            
            # require at least 3 previous races
            if len(history) < 3:
                continue
            
            current_race = df_driver.iloc[idx]
            current_season = int(current_race["season"])
            
            if current_season not in rows_per_season:
                rows_per_season[current_season] = 0
            rows_per_season[current_season] += 1
            
            # ---------- EXPERIENCE ----------
            total_races = len(history)
            
            # ---------- EXPONENTIAL WEIGHTS ----------
            # more weight to recent races
            weights = np.exp(np.linspace(-2, 0, len(history)))
            weights = weights / weights.sum()
            
            # ---------- POSITION (WEIGHTED) ----------
            positions = history["final_position_num"].fillna(20)
            weighted_avg_position = np.average(positions, weights=weights)
            
            # ---------- POINTS (WEIGHTED) ----------
            points_hist = history["points"].fillna(0)
            weighted_avg_points = np.average(points_hist, weights=weights)
            
            # ---------- RECENT FORM ----------
            last_3 = history.tail(3)
            recent_form_pos = last_3["final_position_num"].fillna(20).mean()
            recent_points = last_3["points"].fillna(0).sum()
            
            last_5 = history.tail(5)
            podiums_last_5 = (last_5["final_position_num"] <= 3).sum()
            
            # ---------- CONSISTENCY ----------
            std_positions = history["final_position_num"].fillna(20).std()
            
            # ---------- FINISH RATE ----------
            finished = (history["status"] == "Finished").sum()
            finish_rate = finished / len(history) if len(history) > 0 else 0.0
            
            # ---------- QUALIFYING PERFORMANCE ----------
            avg_grid = history["grid_position_num"].fillna(20).mean()
            
            # improvement from grid to race:
            # positive = usually gains positions, negative = usually loses
            avg_grid_to_race_gain = (
                history["grid_position_num"].fillna(20)
                - history["final_position_num"].fillna(20)
            ).mean()
            
            # ---------- CIRCUIT-SPECIFIC PERFORMANCE ----------
            # We'll use 'event_name' as circuit identifier (or 'location' if you prefer)
            circuit_col = "event_name"  # or "location"
            current_circuit = current_race[circuit_col]
            
            history_circuit = history[history[circuit_col] == current_circuit]
            
            if len(history_circuit) > 0:
                avg_pos_circuit = (
                    history_circuit["final_position_num"].fillna(20).mean()
                )
                races_on_circuit = len(history_circuit)
            else:
                avg_pos_circuit = weighted_avg_position
                races_on_circuit = 0
            
            # ---------- CURRENT RACE RESULT ----------
            current_final_pos = current_race["final_position_num"]
            current_points = current_race["points"]
            
            # ---------- BUILD FEATURE ROW ----------
            row = {
                # meta
                "season": current_season,
                "round": int(current_race["round"]),
                "event_name": current_race["event_name"],
                "location": current_race.get("location", None),
                "driver_code": current_race["driver_code"],
                "driver_name": current_race.get("driver_name", None),
                "team_name": current_race.get("team_name", None),
                
                # target-style info (results of current race)
                "final_position": current_final_pos,
                "won": 1 if current_final_pos == 1 else 0,
                "podium": 1 if current_final_pos <= 3 else 0,
                "points_scored": current_points,
                "grid_position": current_race["grid_position_num"],
                
                # experience
                "race_experience": total_races,
                
                # weighted performance
                "weighted_avg_position": weighted_avg_position,
                "weighted_avg_points": weighted_avg_points,
                
                # recent form (short window)
                "recent_form_3races": recent_form_pos,
                "recent_points_3races": recent_points,
                "podiums_last_5races": int(podiums_last_5),
                
                # stability / reliability
                "consistency_std_position": std_positions,
                "finish_rate": finish_rate,
                
                # qualifying skills
                "avg_grid_position_history": avg_grid,
                "avg_grid_to_race_gain": avg_grid_to_race_gain,
                
                # track-specific performance
                "avg_position_on_circuit": avg_pos_circuit,
                "circuit_experience": races_on_circuit,
            }
            
            features.append(row)
        
        processed_drivers += 1
    
    print(f"\n✓ Processed {processed_drivers} drivers")
    print("\nRows generated per season:")
    for s in sorted(rows_per_season.keys()):
        print(f"  {s}: {rows_per_season[s]} rows")
    
    return pd.DataFrame(features)
# Cell 5: generate the features

print("=== BUILDING FEATURE DATASET ===\n")

df_features = create_advanced_features(df_races)

print("\nShape of feature dataset:", df_features.shape)
df_features.head()


FEATURES_PATH = "f1_race_features.csv"

df_features.to_csv(FEATURES_PATH, index=False)
print(f"Features saved to: {FEATURES_PATH}")