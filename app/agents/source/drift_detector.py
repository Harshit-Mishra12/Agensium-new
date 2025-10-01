import pandas as pd
import numpy as np
import sqlite3
import json
import os
from scipy.stats import ks_2samp, chi2_contingency


class DriftDetector:
    @staticmethod
    def _detect_drift_between_dfs(baseline_df: pd.DataFrame, current_df: pd.DataFrame):
        """
        Detect drift between two DataFrames and return in unified 'columns' format.
        """
        columns = {}
        baseline_cols = set(baseline_df.columns)
        current_cols = set(current_df.columns)

        all_cols = baseline_cols | current_cols

        for col in all_cols:
            drift_info = {
                "drift_score": None,
                "p_value": None,
                "direction": None,
                # "schema_change": None,
                # "earliest_baseline": None,
                # "earliest_current": None,
                # "latest_baseline": None,
                # "latest_current": None,
                # "new_categories": []
            }

            # --- Schema drift ---
            if col not in baseline_cols:
                drift_info["schema_change"] = f"new column detected: {col}"
            elif col not in current_cols:
                drift_info["schema_change"] = f"column missing: {col}"
            elif col in baseline_cols & current_cols:
                if baseline_df[col].dtype != current_df[col].dtype:
                    drift_info["schema_change"] = f"type change: {baseline_df[col].dtype} → {current_df[col].dtype}"

            # --- Skip empty columns ---
            if col not in baseline_df.columns or col not in current_df.columns:
                columns[col] = drift_info
                continue

            base_col = baseline_df[col].dropna()
            curr_col = current_df[col].dropna()
            if base_col.empty or curr_col.empty:
                columns[col] = drift_info
                continue

            # --- Numeric drift ---
            if pd.api.types.is_numeric_dtype(base_col):
                try:
                    ks_stat, p_value = ks_2samp(base_col, curr_col)
                    direction = "increase" if curr_col.mean() > base_col.mean() else "decrease"
                    drift_info.update({
                        "drift_score": round(float(ks_stat), 2),
                        "p_value": round(float(p_value), 2),
                        "direction": f"{direction} in mean {col}"
                    })
                except Exception:
                    pass

            # --- Categorical drift ---
            elif pd.api.types.is_object_dtype(base_col):
                try:
                    base_counts = base_col.value_counts()
                    curr_counts = curr_col.value_counts()
                    all_categories = list(set(base_counts.index) | set(curr_counts.index))
                    base_freq = [base_counts.get(cat, 0) for cat in all_categories]
                    curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]

                    chi2, p_value, _, _ = chi2_contingency([base_freq, curr_freq])
                    new_cats = list(set(curr_col.unique()) - set(base_col.unique()))
                    direction = "new categories appeared" if new_cats else "distribution changed"
                    drift_info.update({
                        "drift_score": round(float(chi2), 2),
                        "p_value": round(float(p_value), 2),
                        "direction": direction,
                        # "new_categories": new_cats
                    })
                except Exception:
                    pass

            # --- Datetime drift ---
            elif pd.api.types.is_datetime64_any_dtype(base_col) or "date" in col.lower():
                try:
                    base_dates = pd.to_datetime(base_col, errors="coerce").dropna()
                    curr_dates = pd.to_datetime(curr_col, errors="coerce").dropna()
                    if not base_dates.empty and not curr_dates.empty:
                        drift_info.update({
                            "earliest_baseline": str(base_dates.min().date()),
                            "earliest_current": str(curr_dates.min().date()),
                            "latest_baseline": str(base_dates.max().date()),
                            "latest_current": str(curr_dates.max().date())
                        })
                except Exception:
                    pass

            columns[col] = drift_info

        return {"columns": columns}

    # --- Loaders ---
    @staticmethod
    def _load_sql_to_dfs(sql_path: str):
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_text = f.read()
        conn = sqlite3.connect(":memory:")
        try:
            cursor = conn.cursor()
            cursor.executescript(sql_text)
            tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            table_dfs = {}
            for table in tables_df["name"].tolist():
                table_dfs[table] = pd.read_sql(f"SELECT * FROM {table};", conn)
            return table_dfs
        finally:
            conn.close()

    @staticmethod
    def _load_json_to_df(json_path: str) -> pd.DataFrame:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure for drift detection")

    # --- Main Drift Detector ---
    @staticmethod
    def detect_drift(baseline_file: str, current_file: str, baseline_name: str = None, current_name: str = None):
        baseline_label = os.path.splitext(os.path.basename(baseline_name or baseline_file))[0]
        current_label = os.path.splitext(os.path.basename(current_name or current_file))[0]
        top_key = f"{baseline_label}_{current_label}"

        # --- CSV ---
        if baseline_file.endswith(".csv") and current_file.endswith(".csv"):
            baseline_df = pd.read_csv(baseline_file)
            current_df = pd.read_csv(current_file)
            return {top_key: {"main": DriftDetector._detect_drift_between_dfs(baseline_df, current_df)}}

        # --- SQL ---
        elif baseline_file.endswith(".sql") and current_file.endswith(".sql"):
            baseline_tables = DriftDetector._load_sql_to_dfs(baseline_file)
            current_tables = DriftDetector._load_sql_to_dfs(current_file)
            report = {}
            for table in set(baseline_tables.keys()) & set(current_tables.keys()):
                report[table] = DriftDetector._detect_drift_between_dfs(
                    baseline_tables[table], current_tables[table]
                )
            return {top_key: report}

        # --- JSON ---
        elif baseline_file.endswith(".json") and current_file.endswith(".json"):
            baseline_df = DriftDetector._load_json_to_df(baseline_file)
            current_df = DriftDetector._load_json_to_df(current_file)
            return {top_key: {"main": DriftDetector._detect_drift_between_dfs(baseline_df, current_df)}}

        # --- Excel ---
        elif baseline_file.endswith((".xls", ".xlsx")) and current_file.endswith((".xls", ".xlsx")):
            baseline_xls = pd.ExcelFile(baseline_file)
            current_xls = pd.ExcelFile(current_file)
            report = {}
            for sheet in set(baseline_xls.sheet_names) & set(current_xls.sheet_names):
                baseline_df = pd.read_excel(baseline_file, sheet_name=sheet)
                current_df = pd.read_excel(current_file, sheet_name=sheet)
                report[sheet] = DriftDetector._detect_drift_between_dfs(baseline_df, current_df)
            return {top_key: report}

        else:
            raise ValueError("Both files must be of the same type (.csv, .sql, .json, .xls, .xlsx)")
