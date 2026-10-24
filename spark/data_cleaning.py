"""
Script: spark/data_cleaning.py
Phase 2: PySpark Data Cleaning & Preprocessing

This script builds a scalable PySpark data cleaning pipeline for the 1.61M job dataset:
1. Ingests raw data from data/raw/job_descriptions.csv
2. Removes exact duplicate records and deduplicates by Job Id
3. Imputes missing values safely without losing rows
4. Normalizes column names into snake_case
5. Trims and normalizes text fields (titles, skills, location, description)
6. Extracts numeric structured fields for experience (min/max years) and salary (min/max/avg USD)
7. Constructs combined text field for NLP/TF-IDF processing
8. Saves the processed dataset as data/processed/jobs_cleaned.parquet
"""

import os
import sys
import time
from typing import Dict, Any, List

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import DataFrame, functions as F
from src.spark_session import get_spark_session


def clean_job_dataset(df_raw: DataFrame) -> tuple[DataFrame, Dict[str, Any]]:
    """
    Applies pure PySpark transformations to clean, standardize, and enrich the job dataset.
    
    Returns:
        tuple: (df_cleaned, report_metrics)
    """
    initial_count = df_raw.count()
    print(f"Initial raw record count: {initial_count:,}")

    # 1. Deduplication by Job Id and full row
    df_dedup = df_raw.dropDuplicates(["Job Id"]).dropDuplicates()
    dedup_count = df_dedup.count()
    removed_duplicates = initial_count - dedup_count
    print(f"Duplicates removed: {removed_duplicates:,} | Remaining records: {dedup_count:,}")

    # 2. Rename columns to standardized snake_case
    column_mapping = {
        "Job Id": "job_id",
        "Experience": "experience",
        "Qualifications": "qualifications",
        "Salary Range": "salary_range",
        "location": "location",
        "Country": "country",
        "latitude": "latitude",
        "longitude": "longitude",
        "Work Type": "work_type",
        "Company Size": "company_size",
        "Job Posting Date": "job_posting_date",
        "Preference": "preference",
        "Contact Person": "contact_person",
        "Contact": "contact",
        "Job Title": "job_title",
        "Role": "role",
        "Job Portal": "job_portal",
        "Job Description": "job_description",
        "Benefits": "benefits",
        "skills": "skills",
        "Responsibilities": "responsibilities",
        "Company": "company",
        "Company Profile": "company_profile"
    }

    df_renamed = df_dedup
    for old_col, new_col in column_mapping.items():
        if old_col in df_renamed.columns:
            df_renamed = df_renamed.withColumnRenamed(old_col, new_col)

    # 3. Handle missing values intelligently
    # Impute missing company_profile with "Not Provided"
    df_imputed = df_renamed.withColumn(
        "company_profile",
        F.coalesce(F.col("company_profile"), F.lit("Not Provided"))
    )

    # 4. Text cleaning and trimming across string columns
    string_cols = [c for c, d in df_imputed.dtypes if d == "string"]
    
    df_trimmed = df_imputed
    for col_name in string_cols:
        # Trim leading/trailing whitespace and normalize internal spaces
        df_trimmed = df_trimmed.withColumn(
            col_name,
            F.regexp_replace(F.trim(F.col(col_name)), r"\s+", " ")
        )

    # 5. Extract structured numeric Experience fields (min_experience_years, max_experience_years)
    # Format sample: "5 to 15 Years"
    df_exp = df_trimmed.withColumn(
        "min_experience_years",
        F.regexp_extract(F.col("experience"), r"(\d+)\s*to\s*(\d+)", 1).cast("integer")
    ).withColumn(
        "max_experience_years",
        F.regexp_extract(F.col("experience"), r"(\d+)\s*to\s*(\d+)", 2).cast("integer")
    )

    # 6. Extract structured numeric Salary fields (min_salary_usd, max_salary_usd, avg_salary_usd)
    # Format sample: "$59K-$99K" -> 59000 to 99000
    df_salary = df_exp.withColumn(
        "min_salary_usd",
        (F.regexp_extract(F.col("salary_range"), r"\$(\d+)K-\$(\d+)K", 1).cast("integer") * 1000)
    ).withColumn(
        "max_salary_usd",
        (F.regexp_extract(F.col("salary_range"), r"\$(\d+)K-\$(\d+)K", 2).cast("integer") * 1000)
    ).withColumn(
        "avg_salary_usd",
        (F.col("min_salary_usd") + F.col("max_salary_usd")) / 2.0
    )

    # 7. Create normalized lowercased NLP feature columns
    df_nlp = df_salary.withColumn(
        "clean_job_title", F.lower(F.col("job_title"))
    ).withColumn(
        "clean_role", F.lower(F.col("role"))
    ).withColumn(
        "clean_skills", F.lower(F.col("skills"))
    ).withColumn(
        "clean_location", F.lower(F.col("location"))
    ).withColumn(
        "clean_country", F.lower(F.col("country"))
    ).withColumn(
        "clean_description", F.lower(F.col("job_description"))
    )

    # 8. Create combined_text column for downstream TF-IDF / Similarity indexing
    df_cleaned = df_nlp.withColumn(
        "combined_text",
        F.concat_ws(
            " ",
            F.col("clean_job_title"),
            F.col("clean_role"),
            F.col("clean_skills"),
            F.col("responsibilities"),
            F.col("clean_description")
        )
    )

    final_count = df_cleaned.count()
    print(f"Final cleaned record count: {final_count:,}")

    metrics = {
        "initial_count": initial_count,
        "cleaned_count": final_count,
        "removed_duplicates": removed_duplicates,
        "added_columns": [
            "min_experience_years", "max_experience_years",
            "min_salary_usd", "max_salary_usd", "avg_salary_usd",
            "clean_job_title", "clean_role", "clean_skills",
            "clean_location", "clean_country", "clean_description", "combined_text"
        ]
    }

    return df_cleaned, metrics


def main():
    print("=" * 80)
    print(" PHASE 2: PYSPARK DATA CLEANING & PREPROCESSING ")
    print("=" * 80)
    
    start_time = time.time()
    spark = get_spark_session()
    
    raw_path = "data/raw/job_descriptions.csv"
    output_dir = "data/processed"
    output_path = os.path.join(output_dir, "jobs_cleaned.parquet")
    
    print(f"\n[Step 1] Loading raw dataset from '{raw_path}'...")
    df_raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", "true")
        .csv(raw_path)
    )
    
    print("\n[Step 2] Executing PySpark Cleaning Pipeline...")
    df_cleaned, metrics = clean_job_dataset(df_raw)
    
    print(f"\n[Step 3] Saving cleaned dataset to '{output_path}' in Parquet format...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as Parquet overwrite mode
    df_cleaned.write.mode("overwrite").parquet(output_path)
    print("-> Cleaned Parquet dataset written successfully!")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" DATA CLEANING COMPLETED IN {elapsed:.2f} SECONDS ")
    print("=" * 80)
    
    spark.stop()


if __name__ == "__main__":
    main()
