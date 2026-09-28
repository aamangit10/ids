import pytest
import numpy as np
import pandas as pd
from research.preprocessing.cleaner import clean_dataframe, strip_leakage_features, TabularPreprocessor, DEFAULT_TARGET_COL
from research.data.synthetic_generator import generate_synthetic_tabular_dataset
from research.experiments.holdout_experiment import run_holdout_experiment


def test_clean_dataframe_and_nan_handling():
    """Verify clean_dataframe removes infinity, NaNs, and strips whitespace."""
    raw_data = {
        " Destination Port ": [80, 443, 8080],
        "Flow Duration": ["1000", "Infinity", "-inf"],
        "Flow Bytes/s": [500.0, np.nan, 200.0],
        "Label": ["BENIGN ", "DDoS", "BENIGN"]
    }
    df = pd.DataFrame(raw_data)
    cleaned = clean_dataframe(df, target_col="Label")
    
    assert len(cleaned) == 1  # 2 invalid/infinity rows dropped
    assert cleaned.iloc[0]["Label"] == "BENIGN"
    assert "Destination Port" in cleaned.columns  # Header whitespace stripped


def test_leakage_feature_stripping():
    """Verify IP addresses, Timestamps, and Flow IDs are strictly stripped."""
    df_syn = generate_synthetic_tabular_dataset(n_samples=50, seed=42)
    assert "Source IP" in df_syn.columns
    assert "Flow ID" in df_syn.columns
    
    df_stripped, removed_cols = strip_leakage_features(df_syn, target_col=DEFAULT_TARGET_COL)
    
    assert "Source IP" not in df_stripped.columns
    assert "Flow ID" not in df_stripped.columns
    assert "Destination IP" not in df_stripped.columns
    assert "Timestamp" not in df_stripped.columns
    assert len(removed_cols) > 0


def test_preprocessor_fitting_and_encoding():
    """Verify TabularPreprocessor fits strictly on train and encodes targets properly."""
    df_syn = generate_synthetic_tabular_dataset(n_samples=100, seed=42)
    preprocessor = TabularPreprocessor()
    
    X_train, y_binary, y_multi, df_proc = preprocessor.fit_transform_train(df_syn)
    
    assert preprocessor.is_fitted
    assert X_train.shape[0] == len(df_proc)
    assert set(np.unique(y_binary)).issubset({0, 1})
    
    # Test transform on new data (no refitting)
    df_syn_test = generate_synthetic_tabular_dataset(n_samples=20, seed=99)
    X_test, y_test_bin, y_test_multi = preprocessor.transform(df_syn_test)
    
    assert X_test.shape[1] == X_train.shape[1]
    assert X_test.shape[0] == len(df_syn_test)


def test_holdout_zero_samples_in_train():
    """
    CRITICAL RESEARCH VALIDITY TEST:
    Proves that when an attack class (e.g. 'Web Attack' or 'PortScan') is selected as held-out,
    there are EXACTLY ZERO samples of that attack class in the training partition.
    """
    attack_to_holdout = "Web Attack"
    res = run_holdout_experiment(
        attack_name=attack_to_holdout,
        data_dir="demo_data",
        model_type="random_forest",
        save_results=False
    )
    
    # Check strict zero sample requirement
    assert res["held_out_train_samples_count"] == 0, \
        f"CRITICAL TEST FAILED: Training set contained {res['held_out_train_samples_count']} samples of held-out attack '{attack_to_holdout}'!"
    
    assert res["held_out_test_samples_count"] > 0, "Test set must contain samples of the held-out attack for evaluation."
    assert 0.0 <= res["held_out_attack_recall"] <= 1.0
