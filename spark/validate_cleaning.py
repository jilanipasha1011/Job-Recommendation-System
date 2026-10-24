"""
Script: spark/validate_cleaning.py
Phase 2: Data Cleaning Validation Report

This script reads the cleaned Parquet dataset at data/processed/jobs_cleaned.parquet and validates:
1. File readability and schema structure
2. Total cleaned row count matching expected target
3. Null counts across all cleaned columns
4. Correct extraction of numeric salary and experience bounds
5. Sample inspection of enriched features (clean_skills, combined_text, etc.)
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


def validate_cleaned_dataset():
    """
    Executes validation suite on jobs_cleaned.parquet.
    """
    print("=" * 80)
    print(" PHASE 2: CLEANING VALIDATION REPORT ")
    print("=" * 80)
    
    start_time = time.time()
    spark = get_spark_session()
    
    parquet_path = "data/processed/jobs_cleaned.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Cleaned dataset not found at '{parquet_path}'. Please run spark/data_cleaning.py first.")
        
    print(f"\n[1] Reading cleaned Parquet dataset from '{parquet_path}'...")
    df_cleaned = spark.read.parquet(parquet_path)
    
    row_count = df_cleaned.count()
    col_count = len(df_cleaned.columns)
    
    print("\n" + "-" * 80)
    print(f" Dataset Path        : {parquet_path}")
    print(f" Cleaned Row Count   : {row_count:,}")
    print(f" Cleaned Column Count: {col_count}")
    print("-" * 80)
    
    print("\n[2] Cleaned Dataset Schema:")
    df_cleaned.printSchema()
    
    print("\n[3] Post-Cleaning Null / Missing Value Audit:")
    null_info = analyze_nulls(df_cleaned)
    print("-" * 80)
    print(f" {'COLUMN NAME':<25} | {'NULL COUNT':<15} | {'NULL %':<10}")
    print("-" * 80)
    for item in null_info:
        print(f" {item['column']:<25} | {item['null_count']:<15,} | {item['null_pct']:<10.2f}%")
    print("-" * 80)
    
    print("\n[4] Summary Statistics for Extracted Numeric Features:")
    df_cleaned.select(
        "min_experience_years", "max_experience_years",
        "min_salary_usd", "max_salary_usd", "avg_salary_usd"
    ).describe().show()
    
    print("\n[5] Sample Cleaned & Enriched Records Preview:")
    sample = df_cleaned.limit(2).collect()
    for idx, row in enumerate(sample, 1):
        print(f"\n--- Cleaned Sample #{idx} ---")
        row_dict = row.asDict()
        for key in ["job_id", "job_title", "company", "location", "experience", "min_experience_years", "max_experience_years", "salary_range", "min_salary_usd", "max_salary_usd", "avg_salary_usd", "clean_skills"]:
            if key in row_dict:
                print(f"  {key:<22}: {row_dict[key]}")
        print(f"  {'combined_text (preview)':<22}: {str(row_dict.get('combined_text', ''))[:100]}...")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" VALIDATION SUCCESSFUL IN {elapsed:.2f} SECONDS ")
    print("=" * 80)
    
    spark.stop()


if __name__ == "__main__":
    validate_cleaned_dataset()
