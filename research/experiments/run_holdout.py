import argparse
import json
from research.experiments.holdout_experiment import run_holdout_experiment


def main():
    parser = argparse.ArgumentParser(description="Run Held-Out / Unseen Attack IDS Experiment")
    parser.add_argument("--attack", type=str, default="Web Attack", help="Attack class name to completely hold out from training")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "logistic_regression", "xgboost"], help="Baseline model to use")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing CICIDS2017 raw CSV files")

    args = parser.parse_args()

    print(f"============================================================")
    print(f"RUNNING HELD-OUT ATTACK EXPERIMENT")
    print(f"Held-Out Class: {args.attack}")
    print(f"Model Type:     {args.model}")
    print(f"============================================================")

    res = run_holdout_experiment(
        attack_name=args.attack,
        data_dir=args.data_dir,
        model_type=args.model,
        save_results=True
    )

    print("\n--- EXPERIMENT RESULTS ---")
    print(f"Status:                      {res['status_note']}")
    print(f"Train Samples (Non-Held-Out): {res['train_samples_count']}")
    print(f"Held-Out Train Samples:      {res['held_out_train_samples_count']} (CRITICAL VERIFICATION PASS)")
    print(f"Held-Out Test Samples:       {res['held_out_test_samples_count']}")
    print(f"Overall Precision:           {res['overall_precision']}")
    print(f"Overall Recall:              {res['overall_recall']}")
    print(f"Overall F1-Score:            {res['overall_f1']}")
    print(f"False Positive Rate (FPR):   {res['false_positive_rate']}")
    print(f"Benign Traffic Recall:       {res['benign_recall']}")
    print(f"Known Attacks Recall:        {res['known_attack_recall']}")
    print(f"HELD-OUT ATTACK RECALL:      {res['held_out_attack_recall']} <--- (OOD Zero-Shot Detection Rate)")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()
