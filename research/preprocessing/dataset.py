import os
import glob
import pandas as pd
from typing import Tuple, List, Optional
from research.preprocessing.cleaner import clean_dataframe
from research.data.synthetic_generator import create_demo_data_if_missing


def load_cicids2017_csv_data(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, bool, str]:
    """
    Loads all CICIDS2017 CSV files in `data_dir` if present.
    If no CSV files are found, falls back to generating/loading DEMO DATA.

    Returns:
        (df_combined, is_demo_mode, status_message)
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if len(csv_files) > 0:
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(f, encoding='latin1', low_memory=False)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        message = f"Loaded {len(csv_files)} real CICIDS2017 CSV file(s) with total {len(combined_df)} records."
        return combined_df, False, message
    else:
        # Fallback to demo mode
        demo_csv_path, _ = create_demo_data_if_missing()
        df = pd.read_csv(demo_csv_path)
        message = f"DEMO DATA — NOT RESEARCH RESULTS. Real CICIDS2017 files not found in '{data_dir}'. Loaded demo dataset."
        return df, True, message
