"""
Main Script: main_phase1.py
Phase 1: PySpark Data Ingestion and Dataset Inspection

This script executes Phase 1 of the Job Recommendation System project:
1. Initializes a PySpark SparkSession.
2. Loads the existing job_descriptions.csv dataset using PySpark.
3. Automatically detects dataset columns, data types, and total row count.
4. Performs a missing/null value audit on every column using PySpark functions.
5. Displays a preview of sample job records.
"""

import sys
import time
from src.spark_session import get_spark_session
from src.data_ingestion import load_job_data, get_dataset_summary, analyze_missing_values


def run_phase1(dataset_path: str = "job_descriptions.csv"):
    """
    Executes Phase 1 Data Ingestion and Inspection.
    """
    print("=" * 70)
    print(" PHASE 1: PYSPARK DATA INGESTION & DATASET INSPECTION ")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Initialize PySpark SparkSession
    print("\n[Step 1] Initializing PySpark SparkSession...")
    spark = get_spark_session(app_name="Phase1_DataIngestion")
    print(f"-> SparkSession active. PySpark version: {spark.version}")
    
    # Step 2: Load Job Dataset into PySpark DataFrame
    print(f"\n[Step 2] Reading dataset '{dataset_path}' into PySpark DataFrame...")
    df = load_job_data(spark, dataset_path)
    
    # Step 3: Extract Summary & Schema Metadata
    print("\n[Step 3] Inspecting Schema and Column Structure...")
    summary = get_dataset_summary(df)
    
    print("-" * 70)
    print(f" Total Record Count : {summary['total_rows']:,}")
    print(f" Total Column Count : {summary['total_columns']}")
    print("-" * 70)
    print(" Detected Columns and Data Types:")
    for idx, (col_name, dtype) in enumerate(summary['schema'], 1):
        print(f"  {idx:2d}. {col_name:<25} | Type: {dtype}")
    print("-" * 70)
    
    # Step 4: Perform Null / Missing Value Audit
    print("\n[Step 4] Running Missing Value Analysis using PySpark SQL functions...")
    missing_list = analyze_missing_values(df)
    
    print("-" * 70)
    print(f" {'COLUMN NAME':<25} | {'MISSING COUNT':<15} | {'MISSING %':<10}")
    print("-" * 70)
    for item in missing_list:
        print(f" {item['column']:<25} | {item['null_count']:<15,} | {item['missing_pct']:<10.2f}%")
    print("-" * 70)
    
    # Step 5: Data Sample Preview
    print("\n[Step 5] Displaying Top 3 Job Records Preview:")
    print("-" * 70)
    sample_records = df.limit(3).collect()
    for i, row in enumerate(sample_records, 1):
        print(f"\n--- Job Sample #{i} ---")
        row_dict = row.asDict()
        for col in ["Job Id", "Job Title", "Role", "Company", "Experience", "Qualifications", "Salary Range", "location", "skills"]:
            if col in row_dict:
                val = str(row_dict[col])
                if len(val) > 80:
                    val = val[:77] + "..."
                print(f"  {col:<18}: {val}")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f" PHASE 1 COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS ")
    print("=" * 70)
    
    spark.stop()


if __name__ == "__main__":
    run_phase1()
