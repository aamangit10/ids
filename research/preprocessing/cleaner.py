import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Explicit list of columns removed to prevent data leakage & trivial identification
LEAKAGE_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Source Port",
    "Destination IP",
    "Dst IP",
    "Timestamp",
    "Unnamed: 0",
    "Flow Bytes/s",    # Infinite or trivially calculated string values in raw CICIDS2017
    "Flow Packets/s"
]

DEFAULT_TARGET_COL = "Label"
BENIGN_LABEL = "BENIGN"


def normalize_column_name(col: str) -> str:
    """Clean column headers by stripping whitespace and replacing special chars."""
    col = col.strip()
    col = re.sub(r'[^\w\s\/-]', '', col)
    return col


def clean_dataframe(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL) -> pd.DataFrame:
    """
    Clean raw CICIDS2017 DataFrame:
    1. Clean column headers and normalize whitespace.
    2. Normalize target column string values (strip spaces).
    3. Replace infinity strings and numerical inf values with NaN.
    4. Remove rows with missing feature/label values.
    5. Deduplicate rows.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    # Find normalized target column
    matched_target = None
    for col in df.columns:
        if col.lower() == target_col.lower():
            matched_target = col
            break

    if matched_target:
        if matched_target != target_col:
            df.rename(columns={matched_target: target_col}, inplace=True)
        df[target_col] = df[target_col].astype(str).str.strip()

    # Process all non-target columns
    for col in df.columns:
        if col == target_col:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # Column is natively numeric (int/float)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # String / object / non-numeric column
            s_str = df[col].astype(str).str.strip()
            s_cleaned = s_str.replace(
                ['Infinity', 'inf', 'INF', '-Infinity', '-inf', '-INF', 'nan', 'NaN', 'None', 'null', ''],
                np.nan
            )
            converted = pd.to_numeric(s_cleaned, errors='coerce')
            valid_ratio = converted.notna().sum() / float(max(1, len(df)))

            if valid_ratio > 0.3:
                df[col] = converted
            else:
                df[col] = s_cleaned

    # Replace infinite values in all numeric columns with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop rows where target_col is missing or empty
    if target_col in df.columns:
        df = df[df[target_col].notna() & (df[target_col] != "")]

    # Drop invalid/NaN rows across numeric features
    if len(numeric_cols) > 0:
        df = df.dropna(subset=numeric_cols)

    # Remove duplicate rows if any
    df.drop_duplicates(inplace=True)

    return df


def strip_leakage_features(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL) -> Tuple[pd.DataFrame, List[str]]:
    """
    Excludes features that introduce data leakage (IP addresses, Timestamps, Flow IDs, Ports).
    Returns (cleaned_df, removed_columns_list).
    """
    removed = []
    cols_to_drop = []

    for col in df.columns:
        if col == target_col:
            continue
        col_clean = normalize_column_name(col).lower()
        for leak in LEAKAGE_COLUMNS:
            if col_clean == leak.lower() or leak.lower() in col_clean:
                cols_to_drop.append(col)
                removed.append(f"{col} (Reason: Data leakage / Socket metadata column)")
                break

    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    return df_clean, removed


class TabularPreprocessor:
    """
    Handles fitting scalers strictly on training data, encoding labels,
    and transforming test/inference data reproducibly.
    """

    def __init__(self, target_col: str = DEFAULT_TARGET_COL, benign_label: str = BENIGN_LABEL):
        self.target_col = target_col
        self.benign_label = benign_label
        self.scaler = StandardScaler()
        self.multiclass_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.removed_features: List[str] = []
        self.is_fitted: bool = False

    def fit_transform_train(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Processes training data:
        - Cleans & strips leakage columns.
        - Extracts binary target (0=BENIGN, 1=ATTACK).
        - Extracts multiclass label strings and encodes integer IDs.
        - Fits StandardScaler strictly on training feature matrix X.
        - Returns (X_scaled, y_binary, y_multiclass, processed_df).
        """
        df_clean = clean_dataframe(df, target_col=self.target_col)
        df_clean, self.removed_features = strip_leakage_features(df_clean, target_col=self.target_col)

        if df_clean.empty:
            raise ValueError("Processed dataframe is empty after cleaning and leakage stripping.")

        # Extract target
        y_str = df_clean[self.target_col].astype(str).str.strip().values

        # Binary target: 0 for BENIGN, 1 for any attack
        y_binary = np.where(y_str == self.benign_label, 0, 1).astype(np.int64)

        # Multiclass target encoding
        y_multiclass = self.multiclass_encoder.fit_transform(y_str)

        # Extract feature matrix X
        X_df = df_clean.drop(columns=[self.target_col])
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        self.feature_names = list(X_df.columns)
        X_scaled = self.scaler.fit_transform(X_df.values)
        self.is_fitted = True

        return X_scaled, y_binary, y_multiclass, df_clean

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Transforms test or inference data using existing fitted scaler & columns.
        Does NOT fit scalers to avoid data leakage.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transform().")

        df_clean = clean_dataframe(df, target_col=self.target_col)

        y_binary = None
        y_multiclass = None

        if self.target_col in df_clean.columns:
            y_str = df_clean[self.target_col].astype(str).str.strip().values
            y_binary = np.where(y_str == self.benign_label, 0, 1).astype(np.int64)
            # Safely encode known multiclass labels, defaulting unknown labels to BENIGN for target indexing
            known_classes = set(self.multiclass_encoder.classes_)
            y_str_safe = [label if label in known_classes else self.benign_label for label in y_str]
            y_multiclass = self.multiclass_encoder.transform(y_str_safe)

        # Align feature matrix X with trained feature_names
        X_df = pd.DataFrame(index=df_clean.index)
        for col in self.feature_names:
            if col in df_clean.columns:
                X_df[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
            else:
                X_df[col] = 0.0

        X_scaled = self.scaler.transform(X_df.values)
        return X_scaled, y_binary, y_multiclass

    def save(self, filepath: str) -> None:
        """Serialize preprocessor state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'TabularPreprocessor':
        """Load serialized preprocessor state from disk."""
        return joblib.load(filepath)
