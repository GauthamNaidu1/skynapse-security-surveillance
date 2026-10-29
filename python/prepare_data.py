import os
import sys
import glob
import json
import numpy as np
import pandas as pd

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def find_dataset_file(data_dir="data"):
    """
    Search for a crime CSV file in the data/ directory.
    Returns the file path or None if not found.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return None

    # Preferred filename patterns
    patterns = [
        os.path.join(data_dir, "crime_data.csv"),
        os.path.join(data_dir, "hyderabad_crime_data*.csv"),
        os.path.join(data_dir, "*.csv")
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def compute_trailing_30d_density_no_leakage(df, area_col="Area", time_col="timestamp"):
    """
    Computes trailing 30-day incident density for each incident's area strictly WITHOUT future leakage.
    For an incident at timestamp T in area A, counts all incidents in area A in window [T - 30 days, T).
    Uses binary search (np.searchsorted) over sorted timestamps for O(N log N) speed and zero leakage.
    """
    density = np.zeros(len(df), dtype=int)
    thirty_days = pd.Timedelta(days=30)
    
    # Process each area independently
    for area, group in df.groupby(area_col):
        area_indices = group.index.values
        area_times = group[time_col].values
        
        # area_times are sorted because df is chronologically sorted
        for i, (idx, t) in enumerate(zip(area_indices, area_times)):
            window_start = t - thirty_days
            # Find earliest index in area_times where timestamp >= window_start
            left_pos = np.searchsorted(area_times, window_start, side='left')
            # Events strictly before index i in [t - 30d, t) count as prior density
            density[idx] = i - left_pos
            
    return density

def load_and_prepare_data(csv_path):
    print("=" * 60)
    print(f"LOADING DATASET: {csv_path}")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    print(f"\n[Raw Data Shape] {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[Columns] {list(df.columns)}")
    print("\n[Data Types & Non-Null Counts]")
    print(df.info())
    print("\n[First 3 Rows Preview]")
    print(df.head(3))
    
    # -------------------------------------------------------------
    # 1. PARSE TIMESTAMP
    # -------------------------------------------------------------
    # Check possible timestamp column names
    timestamp_candidates = ['Timestamp', 'timestamp', 'DateTime', 'datetime', 'Date', 'date']
    found_time_col = None
    for col in timestamp_candidates:
        if col in df.columns:
            found_time_col = col
            break
            
    if found_time_col:
        # If separate Time column exists and found_time_col only has date
        if 'Time' in df.columns and found_time_col in ['Date', 'date']:
            df['timestamp'] = pd.to_datetime(df[found_time_col].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        else:
            df['timestamp'] = pd.to_datetime(df[found_time_col], errors='coerce')
    elif 'Time' in df.columns:
        # Fallback if only Time is given: synthesize date sequence or parse time
        df['timestamp'] = pd.to_datetime(df['Time'], errors='coerce')
    else:
        raise ValueError(f"No timestamp or date column found in dataset. Columns: {list(df.columns)}")

    # Drop any records with unparseable timestamps
    initial_len = len(df)
    df = df.dropna(subset=['timestamp']).copy()
    if len(df) < initial_len:
        print(f"[Warning] Dropped {initial_len - len(df)} rows with invalid timestamps.")

    # Sort strictly chronologically
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"\n[Time Span] {df['timestamp'].min()} to {df['timestamp'].max()}")

    # -------------------------------------------------------------
    # 2. DEFINE PREDICTION TARGET: Crime Severity (Low, Medium, High)
    # -------------------------------------------------------------
    # MAPPING RULE:
    # If explicit 'Severity' column exists:
    #   'Low'    -> 0 (Minor infractions, low-risk property or nuisance)
    #   'Medium' -> 1 (Moderate incidents, theft, vandalism, disputes)
    #   'High' / 'Critical' -> 2 (Violent crimes, assaults, armed robbery, arson)
    # If only 'Crime_Type' or 'Category' exists:
    #   Low: 'Traffic Violation', 'Suspicious Activity', 'Trespassing', 'Noise'
    #   Medium: 'Theft', 'Burglary', 'Vandalism', 'Fraud', 'Pickpocketing'
    #   High: 'Robbery', 'Fighting', 'Assault', 'Arson', 'Homicide', 'Weapon'
    # -------------------------------------------------------------
    severity_map = {
        'Low': 0, 'low': 0, 0: 0,
        'Medium': 1, 'medium': 1, 1: 1,
        'High': 2, 'high': 2, 2: 2,
        'Critical': 2, 'critical': 2, 3: 2
    }
    
    if 'Severity' in df.columns:
        df['target_severity'] = df['Severity'].map(severity_map)
    elif 'Severity_Level' in df.columns:
        df['target_severity'] = df['Severity_Level'].map(severity_map)
    elif 'Crime_Type' in df.columns:
        # Heuristic mapping based on standard crime categorization
        crime_to_severity = {
            'Traffic Violation': 0, 'Suspicious Activity': 0, 'Normal': 0,
            'Theft': 1, 'Vandalism': 1, 'Burglary': 1,
            'Fighting': 2, 'Arson': 2, 'Robbery': 2, 'Assault': 2, 'Chain Snatching': 2
        }
        df['target_severity'] = df['Crime_Type'].map(crime_to_severity)
    else:
        raise ValueError("Neither 'Severity' nor 'Crime_Type' column found for target definition.")

    # Fill any unmapped targets with modal value and report
    unmapped = df['target_severity'].isna().sum()
    if unmapped > 0:
        print(f"[Warning] {unmapped} rows had unknown severity; defaulting to 1 (Medium).")
        df['target_severity'] = df['target_severity'].fillna(1).astype(int)
    else:
        df['target_severity'] = df['target_severity'].astype(int)

    # -------------------------------------------------------------
    # 3. FEATURE ENGINEERING
    # -------------------------------------------------------------
    # 3.1 Temporal cyclical features: hour sin & cos (preserves 23:00 -> 00:00 continuity)
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

    # 3.2 Day of week and weekend indicator
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # 3.3 Identify location / area column
    area_col = None
    for col in ['Area', 'area', 'Location', 'location', 'District', 'district', 'PdDistrict']:
        if col in df.columns:
            area_col = col
            break
    if not area_col:
        # If no area column, create a default
        df['Area'] = 'Sector-1'
        area_col = 'Area'

    # 3.4 Trailing 30-day incident density per area (WITHOUT FUTURE LEAKAGE)
    print(f"\n[Computing 30-day trailing density per area strictly on past data...]")
    df['trailing_30d_density'] = compute_trailing_30d_density_no_leakage(df, area_col=area_col, time_col='timestamp')

    # -------------------------------------------------------------
    # 4. CHRONOLOGICAL TEMPORAL SPLIT (70% Train, 15% Val, 15% Test)
    # -------------------------------------------------------------
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # One-hot encode area using categories discovered ONLY from the training set (prevent data leakage)
    train_areas = train_df[area_col].dropna().unique()
    for area in train_areas:
        col_name = f"area_{area}"
        train_df[col_name] = (train_df[area_col] == area).astype(int)
        val_df[col_name] = (val_df[area_col] == area).astype(int)
        test_df[col_name] = (test_df[area_col] == area).astype(int)

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL SPLIT SUMMARY (70 / 15 / 15)")
    print("=" * 60)
    print(f"Train split : {len(train_df):>6} samples | {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Val split   : {len(val_df):>6} samples | {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    print(f"Test split  : {len(test_df):>6} samples | {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    # -------------------------------------------------------------
    # 5. CLASS DISTRIBUTION & IMBALANCE AUDIT
    # -------------------------------------------------------------
    class_names = {0: "Low (0)", 1: "Medium (1)", 2: "High (2)"}
    
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION & IMBALANCE AUDIT")
    print("=" * 60)
    for split_name, split_data in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split_data['target_severity'].value_counts().sort_index()
        total = len(split_data)
        dist_str = " | ".join([f"{class_names.get(k, k)}: {v} ({v/total*100:.1f}%)" for k, v in counts.items()])
        print(f"[{split_name:5}] Total: {total:>5} | {dist_str}")
        
        # Check imbalance ratio
        if len(counts) > 1:
            imbalance_ratio = counts.max() / counts.min()
            if imbalance_ratio > 3.0:
                print(f"        * IMBALANCE FLAGGED in {split_name}: Max/Min class ratio = {imbalance_ratio:.2f}:1")
            else:
                print(f"        * Balanced distribution: Max/Min class ratio = {imbalance_ratio:.2f}:1")

    # Save prepared splits
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    summary = {
        "dataset_source": csv_path,
        "total_samples": n,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_date_range": [str(train_df['timestamp'].min()), str(train_df['timestamp'].max())],
        "val_date_range": [str(val_df['timestamp'].min()), str(val_df['timestamp'].max())],
        "test_date_range": [str(test_df['timestamp'].min()), str(test_df['timestamp'].max())],
        "features": ['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'trailing_30d_density'] + [f"area_{a}" for a in train_areas],
        "target": "target_severity"
    }
    with open("data/processed/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nProcessed splits saved successfully to data/processed/")
    return summary

if __name__ == "__main__":
    dataset_file = find_dataset_file()
    if not dataset_file:
        print("\n" + "!" * 60)
        print("ERROR: No CSV dataset file found in 'data/' directory.")
        print("Please place the Hyderabad Crime CSV file at:")
        print("  data/crime_data.csv")
        print("(or any .csv in the data/ folder)")
        print("!" * 60)
        sys.exit(1)
        
    load_and_prepare_data(dataset_file)
