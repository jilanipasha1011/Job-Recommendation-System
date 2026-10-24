"""
Script: spark/feature_engineering.py
Phase 3: PySpark Feature Engineering

This script builds recommendation features from data/processed/jobs_cleaned.parquet:
1. Constructs the primary text feature 'job_text' combining job title, role, skills, experience, location, and description.
2. Computes structured numerical and categorical features using PySpark SQL functions:
   - skill_count: Number of extracted skills
   - description_length: Total character length of job description
   - experience_level: Categorical tier (Entry-Level, Mid-Level, Senior)
   - remote_flag: Binary flag (1 if remote work indicated, 0 otherwise)
   - location_normalized: Standardized lowercased location text
3. Saves the feature-enriched dataset as data/processed/jobs_features.parquet
"""

import os
import sys
import time
from typing import Dict, Any

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import DataFrame, functions as F
from src.spark_session import get_spark_session


def build_recommendation_features(df_cleaned: DataFrame) -> DataFrame:
    """
    Applies pure PySpark transformations to engineer rich text and structured feature columns.
    """
    # 1. Create primary combined textual feature: job_text
    df_text = df_cleaned.withColumn(
        "job_text",
        F.concat_ws(
            " ",
            F.col("clean_job_title"),
            F.col("clean_role"),
            F.col("clean_skills"),
            F.col("experience"),
            F.col("clean_location"),
            F.col("clean_country"),
            F.col("clean_description")
        )
    )

    # 2. Extract structured feature: skill_count
    # Split skills by comma or whitespace and calculate array size
    df_skills = df_text.withColumn(
        "skill_count",
        F.size(F.split(F.trim(F.col("skills")), r"[,;]+"))
    )

    # 3. Extract structured feature: description_length
    df_desc_len = df_skills.withColumn(
        "description_length",
        F.length(F.col("job_description"))
    )

    # 4. Extract structured feature: experience_level
    # Based on min_experience_years: 0-2 -> Entry-Level, 3-5 -> Mid-Level, >5 -> Senior
    df_exp_level = df_desc_len.withColumn(
        "experience_level",
        F.when(F.col("min_experience_years") <= 2, F.lit("Entry-Level"))
         .when((F.col("min_experience_years") >= 3) & (F.col("min_experience_years") <= 5), F.lit("Mid-Level"))
         .otherwise(F.lit("Senior"))
    )

    # 5. Extract structured feature: remote_flag
    # Detects keywords 'remote', 'telecommute', 'work from home' in location, work_type, or description
    df_remote = df_exp_level.withColumn(
        "remote_flag",
        F.when(
            F.lower(F.col("location")).contains("remote") |
            F.lower(F.col("work_type")).contains("remote") |
            F.lower(F.col("job_description")).contains("work from home") |
            F.lower(F.col("job_description")).contains("remote"),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    # 6. Extract structured feature: location_normalized
    df_features = df_remote.withColumn(
        "location_normalized",
        F.concat_ws(", ", F.col("clean_location"), F.col("clean_country"))
    )

    return df_features


def main():
    print("=" * 80)
    print(" PHASE 3: PYSPARK FEATURE ENGINEERING ")
    print("=" * 80)
    
    start_time = time.time()
    spark = get_spark_session(app_name="Phase3_FeatureEngineering")
    
    input_path = "data/processed/jobs_cleaned.parquet"
    output_dir = "data/processed"
    output_path = os.path.join(output_dir, "jobs_features.parquet")
    
    print(f"\n[Step 1] Loading cleaned dataset from '{input_path}'...")
    df_cleaned = spark.read.parquet(input_path)
    print(f"-> Loaded {df_cleaned.count():,} cleaned job records.")
    
    print("\n[Step 2] Engineering textual and structured recommendation features...")
    df_features = build_recommendation_features(df_cleaned)
    
    print(f"\n[Step 3] Writing feature dataset to '{output_path}' in Parquet format...")
    df_features.write.mode("overwrite").parquet(output_path)
    print("-> Feature dataset written successfully!")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" FEATURE ENGINEERING COMPLETED IN {elapsed:.2f} SECONDS ")
    print("=" * 80)
    
    spark.stop()


if __name__ == "__main__":
    main()
