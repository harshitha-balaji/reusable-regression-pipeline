import os
import json
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Globally ignore specific DtypeWarnings that pandas throws on messy, mixed CSV chunks
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# =====================================================================
# CONFIGURATION LOADER
# =====================================================================
def load_config(config_path: str = "pipeline_settings.json") -> dict:
    """
    Safely loads pipeline settings from an external config file.
    Provides a robust fallback dictionary if the file is missing.
    """
    if not os.path.exists(config_path):
        print(f"[!] Config file '{config_path}' not found. Using built-in defaults.")
        return {
            "pipeline_settings": {
                "test_split_ratio": 0.2,
                "random_state": 42,
                "numeric_threshold": 0.50,
                "cross_validation_folds": 5,
                "scaling_enabled": True
            }
        }
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"[+] Configuration successfully loaded from '{config_path}'.")
    return config


# =====================================================================
# PHASE 1: UNIVERSAL DATA INGESTION & HEURISTIC CLEANING
# =====================================================================
def ingest_and_clean_csv(csv_path: str, numeric_threshold: float = 0.50) -> pd.DataFrame:
    """
    Phase 1: Loads a raw CSV, filters out high-cardinality text junk, 
    detects and filters out ID/Primary Key columns using name tokens + uniqueness ratios, 
    imputes missing data dynamically (Mean for numeric, Mode for text), 
    and outputs a beautiful console tracking dashboard.
    """
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] File not found at: '{csv_path}'")
        return None

    try:
        # Latin-1 encoding handles special symbols and characters gracefully
        raw_df = pd.read_csv(csv_path, encoding="latin1")
        total_rows, total_cols = raw_df.shape
        print(f"\n[+] Successfully loaded: '{os.path.basename(csv_path)}' ({total_rows:,} rows, {total_cols} columns)")

        clean_columns = {}
        dropped_non_numeric = []
        id_cols = []
        total_nans_patched = 0

        # High-Fidelity ID Detection Logic from rrp
        def is_id_column(series: pd.Series, col_name: str) -> bool:
            name_signal = col_name.lower() == "id" or any(
                tok in col_name.lower() for tok in ["_id", "id_", " id", "id ", "index"]
            )
            if not name_signal:
                return False
            
            try:
                clean_series = series.dropna()
                # Must be 100% unique values and contain whole integers (not float metrics)
                if clean_series.nunique() == len(clean_series):
                    if (pd.to_numeric(clean_series, errors='coerce') % 1 == 0).all():
                        return True
            except:
                pass
            return False

        # Evaluate and process every column in the dataset
        for col in raw_df.columns:
            if is_id_column(raw_df[col], col):
                id_cols.append(col)
                continue

            # Check if column is predominantly numeric
            numeric_s = pd.to_numeric(raw_df[col], errors="coerce")
            valid_numeric_count = numeric_s.notnull().sum()
            numeric_ratio = valid_numeric_count / total_rows

            if numeric_ratio >= numeric_threshold:
                # Column is Numeric: Patch missing data using the column Mean
                nan_count = numeric_s.isnull().sum()
                if nan_count > 0:
                    numeric_s = numeric_s.fillna(numeric_s.mean())
                    total_nans_patched += nan_count
                clean_columns[col] = numeric_s
            else:
                # Column is Text: Check for low-cardinality categorical data (e.g. unique categories < 20% of rows)
                unique_values = raw_df[col].dropna().nunique()
                if unique_values > 0 and (unique_values / total_rows) < 0.20:
                    # Column is Categorical: Patch missing text entries using the column Mode
                    nan_count = raw_df[col].isnull().sum()
                    if nan_count > 0:
                        mode_value = raw_df[col].mode()[0]
                        raw_df[col] = raw_df[col].fillna(mode_value)
                        total_nans_patched += nan_count
                    clean_columns[col] = raw_df[col]
                else:
                    dropped_non_numeric.append(col)

        processed_df = pd.DataFrame(clean_columns)

        # Print the high-fidelity console dashboard summary
        print("\n" + "=" * 70)
        print(" PHASE 1: UNIVERSAL DATA INGESTION & PARSING SUMMARY")
        print("=" * 70)
        print(f"  Total Rows Processed:       {len(processed_df):,}")
        print(f"  Valid Processed Columns:    {len(processed_df.columns)} columns passed checks")
        print(f"  Dropped (High-Risk Text):   {len(dropped_non_numeric)} columns")
        print(f"  Dropped (ID / Primary Keys): {len(id_cols)} columns {id_cols if id_cols else ''}")
        print(f"  Missing Cells Imputed:      {total_nans_patched} values patched (Mean/Mode)")

        print("\n  Cleaned columns available for your ML workflow:")
        for idx, col in enumerate(processed_df.columns, 1):
            dtype_label = "Numeric" if pd.api.types.is_numeric_dtype(processed_df[col]) else "Categorical"
            print(f"   [{idx}] {col:<25} ({dtype_label})")
        print("=" * 70)

        return processed_df

    except Exception as e:
        print(f"\n[ERROR] Could not parse CSV file. Details: {e}")
        return None


# =====================================================================
# PHASE 2: TARGET SELECTION & AUTO-ENCODING
# =====================================================================
def select_target_and_features(df: pd.DataFrame):
    """
    Phase 2: Prompts the operator to select a target variable (y).
    Automatically encodes text targets into numerical formats (0, 1) for ccp,
    and converts text features into numerical bits via One-Hot encoding.
    """
    columns_list = list(df.columns)

    print("\n" + "=" * 70)
    print(" PHASE 2: INTERACTIVE TARGET VARIABLE SELECTION")
    print("=" * 70)

    target_column = None
    while target_column is None:
        user_choice = input("\n[?] Enter the name or number of the column you want to predict: ").strip()

        if user_choice.isdigit():
            idx = int(user_choice) - 1
            if 0 <= idx < len(columns_list):
                target_column = columns_list[idx]
            else:
                print(f"[ERROR] Number out of range. Choose between 1 and {len(columns_list)}.")
        elif user_choice in columns_list:
            target_column = user_choice
        else:
            print(f"[ERROR] '{user_choice}' is not a valid column. Match the list exactly.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    # Dynamic Classification Target Handler: Encode text classes into 0s and 1s safely
    if not pd.api.types.is_numeric_dtype(y):
        print(f"\n[!] Detected Categorical Target '{target_column}'. Encoding classes to integers...")
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=target_column)
        
        # Attach encoder to Series so the calling pipeline can save it for predict mode
        y._label_encoder = encoder

        # Display translation map to the console operator
        mapping = {class_label: int(code) for code, class_label in enumerate(encoder.classes_)}
        print(f"    🎯 Translation Key Map: {mapping}")

    # One-Hot Encode remaining text columns in feature matrix X (Forces 0/1 integers instead of booleans)
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    feature_columns = list(X.columns)

    print("\n" + "-" * 70)
    print(f"  Target Extracted  : {target_column} (y)")
    print(f"  Features Configured: {', '.join(feature_columns)} (X)")
    print("-" * 70)

    return X, y, feature_columns