import os
import json
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from research.preprocessing.cleaner import clean_dataframe, TabularPreprocessor, DEFAULT_TARGET_COL, BENIGN_LABEL
from research.preprocessing.dataset import load_cicids2017_csv_data
from research.baseline.classical_models import LogisticRegressionIDS, RandomForestIDS, XGBoostIDS


def run_all_baselines(data_dir: str = "data/raw") -> Dict[str, Any]:
    """
    Trains and evaluates all classical baseline models on full dataset (in-distribution split).
    Saves trained preprocessor and model artifacts to `./models/`.
    Returns comparison dictionary.
    """
    df_raw, is_demo, status_msg = load_cicids2017_csv_data(data_dir=data_dir)
    print(f"Dataset Status: {status_msg}")

    preprocessor = TabularPreprocessor(target_col=DEFAULT_TARGET_COL, benign_label=BENIGN_LABEL)
    
    # Stratified train/test split
    df_clean = clean_dataframe(df_raw, target_col=DEFAULT_TARGET_COL)
    df_train, df_test = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=42,
        stratify=df_clean[DEFAULT_TARGET_COL]
    )

    X_train, y_train, y_train_multi, _ = preprocessor.fit_transform_train(df_train)
    X_test, y_test, y_test_multi = preprocessor.transform(df_test)

    # Save preprocessor artifact
    os.makedirs("models", exist_ok=True)
    preprocessor.save("models/preprocessor.pkl")
    print(f"Saved preprocessor artifact to 'models/preprocessor.pkl'.")

    models = [
        LogisticRegressionIDS(random_state=42),
        RandomForestIDS(random_state=42),
        XGBoostIDS(random_state=42)
    ]

    results = {}
    for m in models:
        print(f"Training {m.name}...")
        m.train(X_train, y_train)
        metrics = m.evaluate(X_test, y_test)
        results[m.name] = metrics
        
        # Save model artifact
        save_name = m.name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        m.save(f"models/{save_name}.pkl")
        print(f"  -> Accuracy: {metrics['accuracy']} | Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1: {metrics['f1_score']} | FPR: {metrics['false_positive_rate']}")

    summary = {
        "is_demo_data": is_demo,
        "status_message": status_msg,
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "features_count": X_train.shape[1],
        "models": results
    }

    os.makedirs("results", exist_ok=True)
    with open("results/baselines_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved baseline evaluation summary to 'results/baselines_summary.json'.")
    return summary


if __name__ == "__main__":
    run_all_baselines()
