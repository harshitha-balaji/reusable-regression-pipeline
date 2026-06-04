import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Globally ignore the specific DtypeWarning that pandas throws on messy CSV chunks
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# =====================================================================
# CONFIG LOADER
# =====================================================================
def load_config(config_path: str = "pipeline_settings.json") -> dict:
    """Loads pipeline settings from external config file."""
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
    print(f"[+] Configuration loaded from '{config_path}'.")
    return config


# =====================================================================
# PHASE 1: DATA INGESTION
# =====================================================================
def ingest_and_clean_csv(csv_path: str, numeric_threshold: float = 0.50):
    """
    Phase 1: Loads a CSV, filters out columns that are not mostly numeric,
    drops true ID columns by uniqueness ratio, patches missing values,
    and returns a pristine DataFrame.
    """
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] File not found at: '{csv_path}'")
        return None

    try:
        raw_df = pd.read_csv(csv_path, encoding="latin1")
        total_rows, total_cols = raw_df.shape
        print(f"\n[+] Successfully loaded: '{os.path.basename(csv_path)}' ({total_rows:,} rows, {total_cols} columns)")

        clean_columns = {}

        for col in raw_df.columns:
            numeric_s = pd.to_numeric(raw_df[col], errors="coerce")
            valid_numeric_count = numeric_s.notnull().sum()
            numeric_ratio = valid_numeric_count / total_rows

            if numeric_ratio >= numeric_threshold:
                clean_columns[col] = numeric_s

        numeric_df = pd.DataFrame(clean_columns)

        def is_id_column(series, col_name):
            # 1. Name heuristic check
            name_signal = col_name.lower() == "id" or any(tok in col_name.lower() for tok in ["_id", "id_", " id", "id ", "index"])
            if not name_signal:
                return False
            
            # 2. Check if it acts like a primary key (100% unique integers)
            try:
                clean_series = series.dropna()
                if clean_series.nunique() == len(clean_series):
                    # Ensure they are clean, whole numbers (no float prices)
                    if (clean_series % 1 == 0).all():
                        return True
            except:
                pass
                
            return False

        id_cols = [c for c in numeric_df.columns if is_id_column(numeric_df[c], c)]
        numeric_df = numeric_df.drop(columns=id_cols)

        dropped_non_numeric = [c for c in raw_df.columns if c not in numeric_df.columns and c not in id_cols]
        total_nans = numeric_df.isnull().sum().sum()
        if total_nans > 0:
            numeric_df = numeric_df.fillna(numeric_df.mean())

        print("\n" + "=" * 70)
        print(" PHASE 1: DATA INGESTION & PARSING SUMMARY")
        print("=" * 70)
        print(f"  Total Rows Processed:     {len(numeric_df):,}")
        print(f"  Valid Numeric Features:   {len(numeric_df.columns)} columns passed threshold")
        print(f"  Dropped (Text/Non-numeric): {len(dropped_non_numeric)} columns")
        print(f"  Dropped (ID Columns):     {len(id_cols)} columns {id_cols if id_cols else ''}")
        print(f"  Missing Values Patched:   {total_nans} cells filled with column means")

        print("\n  Cleaned columns available for your ML model:")
        for idx, col in enumerate(numeric_df.columns, 1):
            print(f"   [{idx}] {col}")
        print("=" * 70)

        return numeric_df

    except Exception as e:
        print(f"\n[ERROR] Could not parse CSV file. Details: {e}")
        return None


# =====================================================================
# PHASE 2: TARGET SELECTION
# =====================================================================
def select_target_and_features(df: pd.DataFrame):
    """
    Phase 2: Prompts the user to select a target column (y).
    Automatically assigns all other columns as input features (X).
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
            print(f"[ERROR] '{user_choice}' is not a valid column. Match the list above exactly.")

    feature_columns = [col for col in columns_list if col != target_column]

    print("\n" + "-" * 70)
    print(f"  Target Selected  : {target_column} (y)")
    print(f"  Features Assigned: {', '.join(feature_columns)} (X)")
    print("-" * 70)

    X = df[feature_columns]
    y = df[target_column]

    return X, y, feature_columns


# =====================================================================
# PHASE 3: TRAINING
# =====================================================================
def train_regression_pipeline(X: pd.DataFrame, y: pd.Series, config: dict):
    """
    Phase 3: Splits the data, builds a scaling + regression pipeline,
    trains the model, and runs cross-validation for trustworthy evaluation.
    """
    settings = config["pipeline_settings"]
    test_size     = settings.get("test_split_ratio", 0.2)
    random_state  = settings.get("random_state", 42)
    cv_folds      = settings.get("cross_validation_folds", 5)
    scaling       = settings.get("scaling_enabled", True)

    print("\n" + "=" * 70)
    print(" PHASE 3: TRAINING THE REGRESSION PIPELINE")
    print("=" * 70)
    print(f"[+] Splitting data: {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test (random_state={random_state})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    steps = []
    if scaling:
        steps.append(('scaler', StandardScaler()))
        print("[+] StandardScaler enabled — features will be normalized before training.")
    steps.append(('regressor', LinearRegression()))

    pipeline = Pipeline(steps)

    print("[+] Fitting model on training data (Ordinary Least Squares)...")
    pipeline.fit(X_train, y_train)

    # Shuffled KFold ensures stable CV estimates even on sorted datasets
    print(f"[+] Running {cv_folds}-fold shuffled cross-validation...")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")

    print(f"[Success] Training complete.")
    print(f"  CV R² Scores : {[round(s, 4) for s in cv_scores]}")
    print(f"  CV R² Mean   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return pipeline, X_train, X_test, y_train, y_test, cv_scores


# =====================================================================
# PHASE 4: ANALYTICS REPORT
# =====================================================================
def generate_terminal_report(pipeline, X_test, y_test, feature_names: list, target_name: str, cv_scores: np.ndarray):
    """
    Phase 4: Evaluates the model on held-out test data, displays CV scores,
    extracts feature coefficients, and renders a CLI analytics dashboard.
    """
    y_pred = pipeline.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    regressor_step = pipeline.named_steps['regressor']
    weights = regressor_step.coef_.flatten()
    intercept = regressor_step.intercept_

    feature_impacts = sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)
    max_abs_weight = max(abs(weights)) if len(weights) > 0 else 1
    if max_abs_weight == 0:
        max_abs_weight = 1

    print("\n" + "=" * 75)
    print(" REUSABLE REGRESSION PIPELINE (RRP) — PERFORMANCE DASHBOARD")
    print("=" * 75)
    print(f"  Target Variable (y) : {target_name}")
    print(f"  Input Features (X)  : {', '.join(feature_names)}")
    print(f"  Test Dataset Size   : {len(y_test):,} rows evaluated")

    print("\n" + "-" * 75)
    print(" MODEL ACCURACY METRICS")
    print("-" * 75)
    print(f"  R-squared (R²) Score  : {r2:.4f}  ← held-out test set")
    print(f"  CV R² Mean            : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})  ← {len(cv_scores)}-fold average")
    print(f"  Mean Absolute Error   : {mae:.2f}")
    print(f"  Root Mean Sq. Error   : {rmse:.2f}")

    print("\n" + "-" * 75)
    print(" FEATURE IMPACT RANKING (SCALED COEFFICIENTS)")
    print("-" * 75)
    print(f"   {'Rank':<6}{'Feature Name':<25}{'Weight':>10}    Visual Power Graph")
    print(f"   {'----':<6}{'------------':<25}{'------':>10}    ------------------")

    for idx, (name, weight) in enumerate(feature_impacts, 1):
        bar_size = int(abs(weight) / max_abs_weight * 20)
        bar_visual = "█" * bar_size
        direction = "(+)" if weight >= 0 else "(-)"
        print(f"   [{idx}]   {name:<22} {weight:>10.2f}    {bar_visual:<20} {direction}")

    print(f"\n  Base Intercept (b): {float(intercept):.2f}")
    print("=" * 75)


# =====================================================================
# PREDICT MODE — load a saved pipeline and run predictions
# =====================================================================
def run_predict_mode(model_path: str):
    """
    Predict Mode: Loads a saved .pkl pipeline and accepts new CSV input
    or manual feature values to produce live predictions.
    """
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file not found: '{model_path}'")
        return

    pipeline = joblib.load(model_path)
    print(f"\n[+] Pipeline loaded from '{model_path}'.")

    feature_names = pipeline.named_steps['regressor'].feature_names_in_ if hasattr(pipeline.named_steps['regressor'], 'feature_names_in_') else None

    print("\n" + "=" * 70)
    print(" PREDICT MODE — Live Inference Interface")
    print("=" * 70)
    print("\nOptions:")
    print("  [1] Predict from a new CSV file")
    print("  [2] Manually enter feature values")
    mode = input("\nSelect option (1-2): ").strip()

    if mode == "1":
        csv_path = input("[?] Enter path to new CSV file: ").strip().strip("'\"")
        if not os.path.exists(csv_path):
            print(f"[ERROR] File not found: '{csv_path}'")
            return

        raw_new_df = pd.read_csv(csv_path, encoding="latin1")

        if feature_names is not None:
            try:
                # Isolate target headers FIRST before parsing numeric types
                new_df = raw_new_df[list(feature_names)].copy()
            except KeyError as e:
                print(f"[ERROR] CSV is missing expected feature columns: {e}")
                return
        else:
            new_df = raw_new_df

        # Cleanly convert only the required feature headers to numbers
        for col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
        new_df = new_df.dropna()

        if len(new_df) == 0:
            print("[ERROR] No valid numeric rows found matching features after cleaning.")
            return

        predictions = pipeline.predict(new_df)
        print(f"\n[+] Predictions for {len(predictions):,} rows:")
        for i, pred in enumerate(predictions[:20], 1):
            print(f"   Row {i:>4}: {pred:.4f}")
        if len(predictions) > 20:
            print(f"   ... and {len(predictions) - 20} more rows.")

    elif mode == "2":
        if feature_names is None:
            print("[ERROR] Cannot determine feature names from this pipeline.")
            return

        print(f"\n[?] Enter values for each feature:")
        values = []
        for fname in feature_names:
            while True:
                try:
                    val = float(input(f"   {fname}: ").strip())
                    values.append(val)
                    break
                except ValueError:
                    print("   [ERROR] Please enter a valid number.")

        # Wrap in a labeled DataFrame to prevent feature name UserWarnings
        input_df = pd.DataFrame([values], columns=list(feature_names))
        prediction = pipeline.predict(input_df)[0]
        print(f"\n  Predicted Value: {prediction:.4f}")

    print("\n" + "=" * 70)
    print("  [Status] Predict mode complete.")
    print("=" * 70)


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  🤖 WELCOME TO THE REUSABLE REGRESSION PIPELINE (RRP)")
    print("=" * 70)

    # Point directly to the uploaded settings file
    config = load_config("pipeline_settings.json")
    settings = config["pipeline_settings"]

    print("\nSelect Mode:")
    print("  [1] Train a new Regression Model on a CSV dataset")
    print("  [2] Load an existing saved pipeline (.pkl) for Live Predictions")
    
    engine_mode = ""
    while engine_mode not in ["1", "2"]:
        engine_mode = input("\nEnter choice (1-2): ").strip()
        if engine_mode not in ["1", "2"]:
            print("[ERROR] Invalid choice. Please enter 1 or 2.")

    # MODE 2: PREDICT MODE
    if engine_mode == "2":
        model_path = input("\n[?] Enter the path to your saved .pkl pipeline file: ").strip().strip("'\"")
        run_predict_mode(model_path)
        print("\n" + "=" * 70)
        print("  [Status] RRP execution finalized.")
        print("=" * 70 + "\n")
        sys.exit(0)

    # MODE 1: TRAINING MODE
    cleaned_dataframe = None

    while cleaned_dataframe is None:
        user_path = input("\n[?] Enter the path to your CSV file (or 'exit' to quit): ").strip()
        if user_path.lower() == "exit":
            print("\nExiting RRP. Goodbye!")
            sys.exit(0)
        user_path = user_path.strip("'\"")
        cleaned_dataframe = ingest_and_clean_csv(user_path, numeric_threshold=settings.get("numeric_threshold", 0.50))

    print("\n[Success] Phase 1 complete. DataFrame locked and loaded.")

    X, y, selected_features = select_target_and_features(cleaned_dataframe)
    print("\n[Success] Phase 2 complete. X and y arrays ready for training.")

    trained_pipeline, X_train, X_test, y_train, y_test, cv_scores = train_regression_pipeline(X, y, config)
    print("\n[Success] Phase 3 complete. Model trained and cross-validated.")

    generate_terminal_report(trained_pipeline, X_test, y_test, selected_features, y.name, cv_scores)

    save_choice = input("\n[?] Save this trained pipeline to disk? (yes/no): ").strip().lower()
    if save_choice in ["yes", "y"]:
        model_filename = f"rrp_pipeline_{y.name}.pkl"
        joblib.dump(trained_pipeline, model_filename)
        print(f"\n[SUCCESS] Pipeline saved to: '{os.path.abspath(model_filename)}'")
        print(f"  To make live predictions later, rerun this script and select Option [2].")
    else:
        print("\n[!] Pipeline discarded from memory. Engine closing cleanly.")

    print("\n" + "=" * 70)
    print("  [Status] RRP execution finalized.")
    print("=" * 70 + "\n")