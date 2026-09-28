import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from research.preprocessing.cleaner import clean_dataframe, strip_leakage_features, TabularPreprocessor, DEFAULT_TARGET_COL, BENIGN_LABEL
from research.preprocessing.dataset import load_cicids2017_csv_data
from research.baseline.classical_models import RandomForestIDS, LogisticRegressionIDS, XGBoostIDS


def run_holdout_experiment(
    attack_name: str,
    data_dir: str = "data/raw",
    model_type: str = "random_forest",
    test_size: float = 0.25,
    random_state: int = 42,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Executes a strict held-out attack evaluation:
    1. Loads full dataset.
    2. Identifies all instances of `attack_name` (held-out attack).
    3. Completely removes ALL instances of `attack_name` from the training pool.
    4. Trains the model strictly on (Benign + Other Known Attacks).
    5. Evaluates the model on a test set containing Benign, Known Attacks, AND the Held-Out Attack.
    6. Calculates Held-Out Attack Recall (out-of-distribution detection rate).

    Returns detailed metric dictionary.
    """
    df_raw, is_demo, status_msg = load_cicids2017_csv_data(data_dir=data_dir)
    df_clean = clean_dataframe(df_raw, target_col=DEFAULT_TARGET_COL)

    # Standardize label strings
    labels_series = df_clean[DEFAULT_TARGET_COL].astype(str).str.strip()
    unique_labels = labels_series.unique().tolist()

    # Identify matching attack class name flexibly (case-insensitive substring match if needed)
    matching_attack = None
    for l in unique_labels:
        if l.lower() == attack_name.lower() or attack_name.lower() in l.lower():
            matching_attack = l
            break

    if matching_attack is None:
        raise ValueError(f"Attack class '{attack_name}' not found in dataset labels: {unique_labels}")

    held_out_label = matching_attack

    # Separate into Pool A (Non-Held-Out) and Pool B (Held-Out Attack)
    non_heldout_mask = (labels_series != held_out_label)
    df_pool_train_test = df_clean[non_heldout_mask].copy()
    df_heldout_all = df_clean[~non_heldout_mask].copy()

    # Split non-heldout pool into train and test
    df_train_raw, df_test_known_raw = train_test_split(
        df_pool_train_test,
        test_size=test_size,
        random_state=random_state,
        stratify=df_pool_train_test[DEFAULT_TARGET_COL]
    )

    # Combine test set: (Known test subset + Held-out attack samples)
    df_test_full_raw = pd.concat([df_test_known_raw, df_heldout_all], ignore_index=True)

    # CRITICAL RESEARCH SANITY CHECK: Ensure ZERO samples of held-out attack in train set!
    train_heldout_count = (df_train_raw[DEFAULT_TARGET_COL].astype(str).str.strip() == held_out_label).sum()
    if train_heldout_count != 0:
        raise AssertionError(f"CRITICAL LEAKAGE ERROR: Training set contains {train_heldout_count} samples of held-out attack '{held_out_label}'!")

    # Fit preprocessor strictly on training data
    preprocessor = TabularPreprocessor(target_col=DEFAULT_TARGET_COL, benign_label=BENIGN_LABEL)
    X_train, y_train_binary, _, _ = preprocessor.fit_transform_train(df_train_raw)

    # Transform test set using fitted preprocessor (no test data leakage)
    X_test, y_test_binary, _ = preprocessor.transform(df_test_full_raw)

    # Initialize requested baseline model
    if model_type.lower() in ["rf", "random_forest"]:
        model = RandomForestIDS(random_state=random_state)
    elif model_type.lower() in ["lr", "logistic_regression"]:
        model = LogisticRegressionIDS(random_state=random_state)
    elif model_type.lower() in ["xgb", "xgboost"]:
        model = XGBoostIDS(random_state=random_state)
    else:
        model = RandomForestIDS(random_state=random_state)

    # Train model
    model.train(X_train, y_train_binary)

    # General evaluation on entire test set
    overall_metrics = model.evaluate(X_test, y_test_binary)

    # Compute breakdown recall metrics
    y_test_pred = model.predict(X_test)
    test_labels_str = df_test_full_raw[DEFAULT_TARGET_COL].astype(str).str.strip().values

    # 1. Benign Recall
    benign_mask = (test_labels_str == BENIGN_LABEL)
    benign_recall = float((y_test_pred[benign_mask] == 0).sum() / benign_mask.sum()) if benign_mask.sum() > 0 else 0.0

    # 2. Known Attack Recall
    known_attack_mask = (~benign_mask) & (test_labels_str != held_out_label)
    known_attack_recall = float((y_test_pred[known_attack_mask] == 1).sum() / known_attack_mask.sum()) if known_attack_mask.sum() > 0 else 0.0

    # 3. Held-Out Attack Recall (Our primary Out-Of-Distribution metric!)
    heldout_mask = (test_labels_str == held_out_label)
    heldout_recall = float((y_test_pred[heldout_mask] == 1).sum() / heldout_mask.sum()) if heldout_mask.sum() > 0 else 0.0

    result = {
        "held_out_attack": held_out_label,
        "model_name": model.name,
        "is_demo_data": is_demo,
        "status_note": status_msg,
        "train_samples_count": len(df_train_raw),
        "held_out_train_samples_count": int(train_heldout_count),
        "test_samples_count": len(df_test_full_raw),
        "held_out_test_samples_count": int(heldout_mask.sum()),
        "overall_precision": overall_metrics["precision"],
        "overall_recall": overall_metrics["recall"],
        "overall_f1": overall_metrics["f1_score"],
        "overall_accuracy": overall_metrics["accuracy"],
        "overall_roc_auc": overall_metrics["roc_auc"],
        "false_positive_rate": overall_metrics["false_positive_rate"],
        "benign_recall": round(benign_recall, 4),
        "known_attack_recall": round(known_attack_recall, 4),
        "held_out_attack_recall": round(heldout_recall, 4),  # <--- CRITICAL RESULT
        "confusion_matrix": overall_metrics["confusion_matrix"]
    }

    if save_results:
        os.makedirs("results/holdout_experiments", exist_ok=True)
        filename = f"holdout_{held_out_label.replace(' ', '_')}_{model_type}.json"
        with open(os.path.join("results/holdout_experiments", filename), "w") as f:
            json.dump(result, f, indent=2)

    return result
