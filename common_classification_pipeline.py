# common_classification_pipeline.py

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# IMPORT THE SHARED MODULAR DATA CORE HELPERS
from core.data_cleaner import load_config, ingest_and_clean_csv, select_target_and_features

# Globally ignore specific DtypeWarnings that pandas throws on messy CSV chunks
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# =====================================================================
# PHASE 3: CLASSIFICATION TRAINING ENGINE
# =====================================================================
def train_classification_pipeline(X: pd.DataFrame, y: pd.Series, config: dict):
    """
    Phase 3: Splits classification data, builds a scaling + logistic pipeline,
    and runs Stratified Cross-Validation to guarantee stable class distributions.
    """
    settings = config["pipeline_settings"]
    test_size     = settings.get("test_split_ratio", 0.2)
    random_state  = settings.get("random_state", 42)
    cv_folds      = settings.get("cross_validation_folds", 5)
    scaling       = settings.get("scaling_enabled", True)

    print("\n" + "=" * 70)
    print(" PHASE 3: TRAINING THE COMMON CLASSIFICATION PIPELINE (CCP)")
    print("=" * 70)
    print(f"[+] Splitting data: {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test (Stratified Split)")

    # stratify=y ensures training and testing splits reflect perfectly matching label distributions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    steps = []
    if scaling:
        steps.append(('scaler', StandardScaler()))
        print("[+] StandardScaler enabled — normalizing feature scales.")
    
    # max_iter=1000 protects the optimization matrix from early convergence failure warnings
    steps.append(('classifier', LogisticRegression(max_iter=1000, random_state=random_state)))
    
    pipeline = Pipeline(steps)

    print("[+] Fitting Logistic Regression model on training metrics...")
    pipeline.fit(X_train, y_train)

    # StratifiedKFold is the gold standard for tracking discrete classifications safely
    print(f"[+] Running {cv_folds}-fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")

    print(f"[Success] Training complete.")
    print(f"  CV Accuracy Scores : {[round(s, 4) for s in cv_scores]}")
    print(f"  CV Accuracy Mean   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return pipeline, X_train, X_test, y_train, y_test, cv_scores


# =====================================================================
# PHASE 4: ANALYTICS DASHBOARD REPORT
# =====================================================================
def generate_classification_dashboard(
    pipeline, X_test, y_test, feature_names: list, target_name: str, cv_scores: np.ndarray
):
    """
    Phase 4: Evaluates discrete model quality markers (Accuracy, Precision, Recall, F1)
    on held-out data sets and maps out normalized categorical log-odds impact levels.
    """
    y_pred = pipeline.predict(X_test)

    # Automatically adapt calculations to fit binary vs multi-class targets safely
    unique_classes = np.unique(y_test)
    is_binary = len(unique_classes) <= 2
    avg_method = 'binary' if is_binary else 'macro'

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
    recall    = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
    f1        = f1_score(y_test, y_pred, average=avg_method, zero_division=0)

    classifier_step = pipeline.named_steps['classifier']
    
    # Extract weights securely depending on feature dimensionality boundaries
    if classifier_step.coef_.ndim > 1 and classifier_step.coef_.shape[0] > 1:
        # Multi-class context: Take average absolute weight scale across target vectors
        weights = np.mean(np.abs(classifier_step.coef_), axis=0)
        is_multiclass = True
    else:
        weights = classifier_step.coef_.flatten()
        is_multiclass = False
        intercept = classifier_step.intercept_[0]

    feature_impacts = sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)
    max_abs_weight = max(abs(weights)) if len(weights) > 0 else 1
    if max_abs_weight == 0:
        max_abs_weight = 1

    print("\n" + "=" * 75)
    print(" COMMON CLASSIFICATION PIPELINE (CCP) — PERFORMANCE DASHBOARD")
    print("=" * 75)
    print(f"  Target Variable (y) : {target_name} ({'Binary' if is_binary else 'Multi-Class'})")
    print(f"  Total Class Labels  : {[int(c) for c in unique_classes]}")
    print(f"  Test Dataset Size   : {len(y_test):,} rows evaluated")

    print("\n" + "-" * 75)
    print(" CLASSIFICATION ACCURACY & LOGISTIC METRICS")
    print("-" * 75)
    print(f"  Accuracy Score       : {accuracy:.4f}  ← Held-out test set accuracy")
    print(f"  CV Accuracy Mean     : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})  ← {len(cv_scores)}-fold stable score")
    print(f"  Precision Score      : {precision:.4f}  ← Model precision ({avg_method} average)")
    print(f"  Recall (Sensitivity) : {recall:.4f}  ← Model capture rate ({avg_method} average)")
    print(f"  F1-Score (Balance)   : {f1:.4f}  ← Balanced harmonic score calculation")

    print("\n" + "-" * 75)
    print(" FEATURE ODDS INFLUENCE RANKING (LOG-ODDS COEFFICIENTS)")
    print("-" * 75)
    print(f"   {'Rank':<6}{'Feature Name':<25}{'Weight':>10}    Visual Power Graph")
    print(f"   {'----':<6}{'------------':<25}{'------':>10}    ------------------")

    for idx, (name, weight) in enumerate(feature_impacts, 1):
        bar_size = int(abs(weight) / max_abs_weight * 20)
        bar_visual = "█" * bar_size
        
        if is_multiclass:
            direction = "(Mean Abs Importance)"
        else:
            direction = "(+ Drives Class 1)" if weight >= 0 else "(- Drives Class 0)"
            
        print(f"   [{idx}]   {name:<22} {weight:>10.2f}    {bar_visual:<20} {direction}")

    if not is_multiclass:
        print(f"\n  Log-Odds Intercept (b): {float(intercept):.2f}")
    print("=" * 75)


# =====================================================================
# LIVE PREDICT MODE
# =====================================================================
def run_predict_mode(model_path: str):
    """ Loads saved classification pipeline to run inference on live vector blocks. """
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model asset not found at: '{model_path}'")
        return

    pipeline = joblib.load(model_path)
    print(f"\n[+] Classification pipeline successfully un-serialized from '{model_path}'.")

    # Load label encoder if it exists alongside the pipeline file
    encoder = None
    encoder_path = model_path.replace("ccp_pipeline_", "ccp_encoder_")
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
        print(f"[+] Label encoder loaded from '{encoder_path}' — predictions will show original class names.")

    # feature_names_in_ lives on the first fitted step — scaler if enabled, classifier if not.
    feature_names = None
    for step_name, step_obj in pipeline.steps:
        if hasattr(step_obj, 'feature_names_in_'):
            feature_names = step_obj.feature_names_in_
            break

    print("\n" + "=" * 70)
    print(" CCP PREDICT MODE — Live Classification Interface")
    print("=" * 70)
    print("\nOptions:")
    print("  [1] Batch predict classes from a new verification CSV file")
    print("  [2] Manually input discrete feature values for single-row inference")
    mode = input("\nSelect option (1-2): ").strip()

    if mode == "1":
        csv_path = input("[?] Enter path to verification CSV file: ").strip().strip("'\"")
        if not os.path.exists(csv_path):
            print(f"[ERROR] File not found: '{csv_path}'")
            return

        raw_new_df = pd.read_csv(csv_path, encoding="latin1")

        if feature_names is not None:
            try:
                # Isolate matching training features first before performing numeric cast filters
                new_df = pd.get_dummies(raw_new_df, drop_first=True, dtype=int)
                # Reindex ensures columns line up perfectly with what the model learned
                new_df = new_df.reindex(columns=list(feature_names), fill_value=0)
            except Exception as e:
                print(f"[ERROR] Could not align feature parameters: {e}")
                return
        else:
            new_df = raw_new_df

        # Cleanly convert required parameters to numeric forms
        for col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
        new_df = new_df.dropna()

        if len(new_df) == 0:
            print("[ERROR] Matrix is completely empty after cleaning operations.")
            return

        predictions = pipeline.predict(new_df)
        probabilities = pipeline.predict_proba(new_df)

        # Decode integer predictions back to original class labels if encoder is available
        display_preds = encoder.inverse_transform(predictions) if encoder else predictions

        print(f"\n[+] Output Prediction Log Matrix (First 20 rows printed):")
        for i, (pred, prob) in enumerate(zip(display_preds[:20], probabilities[:20]), 1):
            max_prob = np.max(prob) * 100
            print(f"   Row {i:>4}: Predicted Class = {pred} | Confidence Level = {max_prob:.1f}%")
        if len(predictions) > 20:
            print(f"   ... and {len(predictions) - 20} more rows.")

    elif mode == "2":
        if feature_names is None:
            print("[ERROR] Cannot parse feature shapes from this pipeline binary file.")
            return

        print(f"\n[?] Enter individual field parameter items:")
        values = []
        for fname in feature_names:
            while True:
                try:
                    val = float(input(f"   {fname}: ").strip())
                    values.append(val)
                    break
                except ValueError:
                    print("   [ERROR] Invalid parameter entry. Input raw number fields.")

        input_df = pd.DataFrame([values], columns=list(feature_names))
        prediction = pipeline.predict(input_df)[0]
        prob_dist = pipeline.predict_proba(input_df)[0]

        # Decode integer prediction back to original label if encoder available
        display_pred = encoder.inverse_transform([prediction])[0] if encoder else int(prediction)

        print(f"\n  🎯 Model Target Output Class: {display_pred}")
        print(f"     Confidence Mapping Breakdown: ")
        class_labels = encoder.classes_ if encoder else range(len(prob_dist))
        for cls_label, p_val in zip(class_labels, prob_dist):
            print(f"       Class [{cls_label}]: {p_val*100:.2f}%")

    print("\n" + "=" * 70)
    print("  [Status] CCP Predict mode finalized.")
    print("=" * 70)


# =====================================================================
# MAIN RUN ENGINE CONTROLLER
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  🤖 WELCOME TO THE COMMON CLASSIFICATION PIPELINE (CCP)")
    print("=" * 70)

    # Ingest settings via shared data engine architecture
    config = load_config("pipeline_settings.json")
    settings = config["pipeline_settings"]

    print("\nSelect Executive Workflow Mode:")
    print("  [1] Train a new Logistic Classification Model")
    print("  [2] Spin up Live Predict Engine from a saved (.pkl) asset")
    
    engine_mode = ""
    while engine_mode not in ["1", "2"]:
        engine_mode = input("\nEnter choice (1-2): ").strip()
        if engine_mode not in ["1", "2"]:
            print("[ERROR] Selection entry mismatch. Enter 1 or 2.")

    # WORKFLOW MODE 2: DESERIALIZED ASSET RUNTIME INFERENCE
    if engine_mode == "2":
        model_path = input("\n[?] Enter path to saved classification pipeline (.pkl): ").strip().strip("'\"")
        run_predict_mode(model_path)
        print("\n" + "=" * 70)
        print("  [Status] CCP execution finalized.")
        print("=" * 70 + "\n")
        sys.exit(0)

    # WORKFLOW MODE 1: ENGINE TRAINING INGESTION
    cleaned_dataframe = None
    while cleaned_dataframe is None:
        user_path = input("\n[?] Enter the path to your CSV file (or 'exit' to quit): ").strip()
        if user_path.lower() == "exit":
            print("\nExiting CCP Pipeline Workspace. Goodbye!")
            sys.exit(0)
        user_path = user_path.strip("'\"")
        cleaned_dataframe = ingest_and_clean_csv(user_path, numeric_threshold=settings.get("numeric_threshold", 0.50))

    print("\n[Success] Phase 1 universal data engine routines finalized.")

    # Map out features and execute automated encoding structures via the shared cleaner
    X, y, selected_features = select_target_and_features(cleaned_dataframe)
    print("\n[Success] Phase 2 target and scaling parameters configured.")

    # Train structural matrix values
    trained_pipeline, X_train, X_test, y_train, y_test, cv_scores = train_classification_pipeline(X, y, config)
    print("\n[Success] Phase 3 training routines completed successfully.")

    # Compile the final terminal dashboard tracking report
    generate_classification_dashboard(trained_pipeline, X_test, y_test, selected_features, y.name, cv_scores)

    # Binary Serialized Asset Export Handling
    save_choice = input("\n[?] Serialize and save classification pipeline to disk? (yes/no): ").strip().lower()
    if save_choice in ["yes", "y"]:
        model_filename = f"ccp_pipeline_{y.name}.pkl"
        joblib.dump(trained_pipeline, model_filename)
        print(f"\n[SUCCESS] Production classification model exported to: '{os.path.abspath(model_filename)}'")

        # If a LabelEncoder was used, save it alongside the pipeline so predict mode
        # can reverse-translate integer predictions back to original class labels.
        if hasattr(y, '_label_encoder'):
            encoder_filename = f"ccp_encoder_{y.name}.pkl"
            joblib.dump(y._label_encoder, encoder_filename)
            print(f"[SUCCESS] Label encoder exported to: '{os.path.abspath(encoder_filename)}'")
            print(f"  Place both .pkl files in the same directory for full label decoding in predict mode.")

        print(f"  To execute verification inference later, run this script and select Option [2].")
    else:
        print("\n[!] Production asset discarded from runtime memory. Engine closing cleanly.")

    print("\n" + "=" * 70)
    print("  [Status] CCP execution finalized.")
    print("=" * 70 + "\n")