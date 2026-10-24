"""
Script: spark/inspect_dataset.py
Phase 1: PySpark Data Ingestion & Dataset Inspection

This script performs comprehensive, non-destructive inspection of the raw job dataset located in `data/raw/`.
It uses PySpark as the primary data processing engine to analyze schema, missing values, duplicates, summary statistics,
and conceptual column mappings.
"""

import os
import sys
import time
from typing import Dict, Any, List

# Ensure parent directory is in sys.path to allow importing from src if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


def get_spark_session(app_name: str = "Phase1_DatasetInspection") -> SparkSession:
    """
    Initializes and returns a PySpark SparkSession configured for local execution with 4GB RAM.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def detect_file_format(file_path: str) -> str:
    """
    Detects data file format based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return "CSV"
    elif ext == ".json":
        return "JSON"
    elif ext in [".parquet", ".pq"]:
        return "Parquet"
    else:
        return f"Unknown ({ext})"


def load_raw_dataset(spark: SparkSession, raw_dir: str = "data/raw") -> tuple[DataFrame, str, str]:
    """
    Locates and reads the raw dataset file from data/raw/ into a PySpark DataFrame.
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory '{raw_dir}' does not exist.")
    
    files = [f for f in os.listdir(raw_dir) if not f.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No dataset files found inside '{raw_dir}'")
    
    # Pick the dataset file (job_descriptions.csv or primary file)
    target_file = files[0]
    file_path = os.path.join(raw_dir, target_file)
    fmt = detect_file_format(file_path)
    
    print(f"Located dataset file: {file_path}")
    print(f"Detected file format: {fmt}")
    
    if fmt == "CSV":
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("quote", '"')
            .option("escape", '"')
            .option("multiLine", "true")
            .csv(file_path)
        )
    elif fmt == "JSON":
        df = spark.read.option("multiLine", "true").json(file_path)
    elif fmt == "Parquet":
        df = spark.read.parquet(file_path)
    else:
        # Default to CSV reading
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
        
    return df, target_file, fmt


def compute_null_counts(df: DataFrame) -> List[Dict[str, Any]]:
    """
    Computes null/missing counts per column using PySpark aggregations.
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


def compute_duplicate_count(df: DataFrame, primary_key: str = "Job Id") -> int:
    """
    Computes total duplicate row count in PySpark DataFrame.
    """
    total_count = df.count()
    if primary_key in df.columns:
        distinct_count = df.select(primary_key).distinct().count()
    else:
        distinct_count = df.dropDuplicates().count()
        
    return total_count - distinct_count


def map_conceptual_columns(columns: List[str]) -> Dict[str, str]:
    """
    Maps dataset columns to standard conceptual entities required for recommendation pipelines.
    """
    col_lower_map = {col.lower(): col for col in columns}
    
    mapping = {
        "title": col_lower_map.get("job title") or col_lower_map.get("role") or col_lower_map.get("title") or "NOT FOUND",
        "company": col_lower_map.get("company") or col_lower_map.get("organization") or "NOT FOUND",
        "location": col_lower_map.get("location") or col_lower_map.get("country") or "NOT FOUND",
        "skills": col_lower_map.get("skills") or col_lower_map.get("key skills") or "NOT FOUND",
        "experience": col_lower_map.get("experience") or col_lower_map.get("experience level") or "NOT FOUND",
        "salary": col_lower_map.get("salary range") or col_lower_map.get("salary") or "NOT FOUND",
        "description": col_lower_map.get("job description") or col_lower_map.get("responsibilities") or "NOT FOUND",
        "job URL": col_lower_map.get("job url") or col_lower_map.get("url") or col_lower_map.get("link") or "NOT FOUND (Job Portal column exists)"
    }
    return mapping


def run_inspection():
    """
    Main dataset inspection workflow.
    """
    print("=" * 80)
    print(" PHASE 1: DATASET INSPECTION (PYSPARK) ")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. Start SparkSession
    print("\n[1] Starting PySpark SparkSession...")
    spark = get_spark_session()
    print(f"-> Active SparkSession version: {spark.version}")
    
    # 2. Locate and load raw dataset
    print("\n[2] Reading raw dataset from data/raw/...")
    df, filename, file_fmt = load_raw_dataset(spark, "data/raw")
    
    # 3. Structural Properties
    total_rows = df.count()
    total_cols = len(df.columns)
    actual_columns = df.columns
    
    print("\n" + "-" * 80)
    print(f" Dataset Filename  : {filename}")
    print(f" Detected Format   : {file_fmt}")
    print(f" Total Row Count   : {total_rows:,}")
    print(f" Total Column Count: {total_cols}")
    print("-" * 80)
    
    # 4. Display Schema
    print("\n[3] PySpark Dataset Schema:")
    df.printSchema()
    
    # 5. Display Null Counts
    print("\n[4] Null / Missing Value Audit:")
    null_info = compute_null_counts(df)
    print("-" * 80)
    print(f" {'COLUMN NAME':<25} | {'NULL COUNT':<15} | {'NULL %':<10}")
    print("-" * 80)
    for item in null_info:
        print(f" {item['column']:<25} | {item['null_count']:<15,} | {item['null_pct']:<10.2f}%")
    print("-" * 80)
    
    # 6. Display Duplicate Count
    print("\n[5] Duplicate Row Check:")
    dup_count = compute_duplicate_count(df, primary_key="Job Id")
    print(f" Total Duplicate Records (by Job Id): {dup_count:,}")
    
    # 7. Conceptual Column Mapping
    print("\n[6] Conceptual Column Mappings:")
    concept_map = map_conceptual_columns(actual_columns)
    for concept, actual in concept_map.items():
        print(f"  * {concept:<15} -> {actual}")
        
    # 8. Basic Numeric / Categorical Summary Statistics
    print("\n[7] Basic Summary Statistics (Sample Columns):")
    df.select("Experience", "Qualifications", "Salary Range", "Work Type", "Company Size").describe().show()
    
    # 9. Sample Records Preview
    print("\n[8] Top 2 Sample Records:")
    sample_records = df.limit(2).collect()
    for idx, row in enumerate(sample_records, 1):
        print(f"\n--- Record #{idx} ---")
        for col in actual_columns[:10]:  # Show first 10 fields per sample record
            print(f"  {col:<20}: {row[col]}")
            
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" INSPECTION COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS ")
    print("=" * 80)
    
    spark.stop()


if __name__ == "__main__":
    run_inspection()
