"""
Script: spark/validate_features.py
Phase 3: Feature Engineering Validation Report

This script reads data/processed/jobs_features.parquet and validates:
1. Schema integrity & feature column additions (job_text, skill_count, description_length, experience_level, remote_flag, location_normalized)
2. Null counts across feature columns
3. Summary statistics for engineered numeric features
4. Sample feature record inspection
"""

import os
import sys
import time
from typing import Dict, Any, List

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import DataFrame, functions as F
from src.spark_session import get_spark_session


def analyze_nulls(df: DataFrame) -> List[Dict[str, Any]]:
    """
    Computes null counts per column in PySpark DataFrame.
    """
    total_rows = df.count()
    if total_rows == 0:
        return []

    agg_exprs = []
    for col_name, data_type in df.dtypes:
        is_string = data_type.startswith("string")
        is_float_or_double = data_type in ["double", "float"]

        cond = F.col(f"`{col_name}`").isNull()
        if is_string:
            cond = cond | (F.trim(F.col(f"`{col_name}`")) == "")
        elif is_float_or_double:
            cond = cond | F.isnan(F.col(f"`{col_name}`"))

        agg_exprs.append(F.count(F.when(cond, 1)).alias(col_name))

    null_row = df.select(agg_exprs).collect()[0]

    null_summary = []
    for col_name in df.columns:
        cnt = null_row[col_name]
        pct = (cnt / total_rows) * 100
        null_summary.append({
            "column": col_name,
            "null_count": cnt,
            "null_pct": round(pct, 2)
        })

    return null_summary


def validate_features():
    """
    Executes feature validation suite.
    """
    print("=" * 80)
    print(" PHASE 3: FEATURE ENGINEERING VALIDATION REPORT ")
    print("=" * 80)
    
    start_time = time.time()
    spark = get_spark_session(app_name="Phase3_Validation")
    
    features_path = "data/processed/jobs_features.parquet"
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature dataset not found at '{features_path}'. Please run spark/feature_engineering.py first.")
        
    print(f"\n[1] Reading feature Parquet dataset from '{features_path}'...")
    df_features = spark.read.parquet(features_path)
    
    row_count = df_features.count()
    col_count = len(df_features.columns)
    
    print("\n" + "-" * 80)
    print(f" Dataset Path        : {features_path}")
    print(f" Total Row Count     : {row_count:,}")
    print(f" Total Column Count  : {col_count}")
    print("-" * 80)
    
    print("\n[2] Engineered Feature Schema:")
    df_features.printSchema()
    
    print("\n[3] Feature Null / Missing Value Audit:")
    null_info = analyze_nulls(df_features)
    print("-" * 80)
    print(f" {'COLUMN NAME':<25} | {'NULL COUNT':<15} | {'NULL %':<10}")
    print("-" * 80)
    for item in null_info:
        print(f" {item['column']:<25} | {item['null_count']:<15,} | {item['null_pct']:<10.2f}%")
    print("-" * 80)
    
    print("\n[4] Summary Statistics for Engineered Features:")
    df_features.select(
        "skill_count", "description_length", "experience_level", "remote_flag"
    ).describe().show()
    
    print("\n[5] Sample Record Feature Preview:")
    sample = df_features.limit(2).collect()
    for idx, row in enumerate(sample, 1):
        print(f"\n--- Feature Sample #{idx} ---")
        row_dict = row.asDict()
        for key in [
            "job_id", "job_title", "company", "skill_count", "description_length",
            "experience_level", "remote_flag", "location_normalized"
        ]:
            if key in row_dict:
                print(f"  {key:<22}: {row_dict[key]}")
        print(f"  {'job_text (preview)':<22}: {str(row_dict.get('job_text', ''))[:120]}...")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" FEATURE VALIDATION COMPLETED IN {elapsed:.2f} SECONDS ")
    print("=" * 80)
    
    spark.stop()


if __name__ == "__main__":
    validate_features()
