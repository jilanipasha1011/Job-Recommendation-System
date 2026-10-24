"""
Module: spark/live_job_pipeline.py
Phase 11: PySpark Live Data Processing & Feature Engineering Pipeline

Converts normalized JSON job objects retrieved from JobNews210 API into a PySpark DataFrame,
applies distributed PySpark text normalization & feature extraction, and formats job records
for multi-factor recommendation ranking.

CRITICAL REQUIREMENT:
All processing after API retrieval is executed strictly in PySpark (no Pandas fallback).
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, DoubleType

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spark_session import get_spark_session


def process_live_jobs_with_pyspark(
    spark: SparkSession,
    normalized_jobs: List[Dict[str, Any]]
) -> DataFrame:
    """
    Transforms normalized live JobNews210 JSON records into a fully feature-engineered PySpark DataFrame.

    Parameters:
        spark (SparkSession): Active PySpark session
        normalized_jobs (List[Dict[str, Any]]): List of canonical job dictionaries

    Returns:
        DataFrame: PySpark DataFrame equipped with filtered_words array and numeric features
    """
    # Define explicit PySpark schema for incoming normalized JSON dictionaries
    input_schema = StructType([
        StructField("job_id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("company", StringType(), True),
        StructField("location", StringType(), True),
        StructField("skills", StringType(), True),
        StructField("experience", StringType(), True),
        StructField("salary", StringType(), True),
        StructField("description", StringType(), True),
        StructField("job_url", StringType(), True),
        StructField("posted_date", StringType(), True),
        StructField("work_mode", StringType(), True)
    ])

    if not normalized_jobs:
        return spark.createDataFrame([], input_schema)

    # Sanitize dictionary values to ensure string types match schema
    clean_records = []
    for item in normalized_jobs:
        rec = {
            "job_id": str(item.get("job_id", "")),
            "title": str(item.get("title", "")),
            "company": str(item.get("company", "")),
            "location": str(item.get("location", "")),
            "skills": str(item.get("skills", "")),
            "experience": str(item.get("experience", "")),
            "salary": str(item.get("salary", "")),
            "description": str(item.get("description", "")),
            "job_url": str(item.get("job_url", "")) if item.get("job_url") else None,
            "posted_date": str(item.get("posted_date", "")),
            "work_mode": str(item.get("work_mode", ""))
        }
        clean_records.append(rec)

    # Write clean records to temporary JSON file to avoid Python 3.14 RDD pickling recursion
    import json
    import tempfile

    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "temp_live"))
    os.makedirs(temp_dir, exist_ok=True)
    temp_json_path = os.path.join(temp_dir, "temp_live_jobs.json")

    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_records, f)

    df_raw = spark.read.schema(input_schema).option("multiline", "true").json(temp_json_path)

    # Standardize column naming and mapping
    df_mapped = (
        df_raw
        .withColumn("role", F.col("title"))
        .withColumnRenamed("title", "job_title")
        .withColumnRenamed("salary", "salary_range")
        .withColumnRenamed("work_mode", "work_type")
        .withColumn("country", F.lit("Not Specified"))
        .withColumn("job_portal", F.lit("JobNews210"))
    )

    # 1. Structured Experience Years Extraction using PySpark Regex
    df_exp = (
        df_mapped
        .withColumn(
            "min_experience_years",
            F.when(
                F.lower(F.col("experience")).contains("fresher") | F.lower(F.col("experience")).contains("entry"),
                F.lit(0)
            ).otherwise(
                F.coalesce(
                    F.regexp_extract(F.col("experience"), r"(\d+)\s*to\s*(\d+)", 1).cast("integer"),
                    F.regexp_extract(F.col("experience"), r"(\d+)", 1).cast("integer"),
                    F.lit(0)
                )
            )
        )
        .withColumn(
            "max_experience_years",
            F.when(
                F.lower(F.col("experience")).contains("fresher") | F.lower(F.col("experience")).contains("entry"),
                F.lit(3)
            ).otherwise(
                F.coalesce(
                    F.regexp_extract(F.col("experience"), r"(\d+)\s*to\s*(\d+)", 2).cast("integer"),
                    F.col("min_experience_years") + F.lit(5)
                )
            )
        )
        .withColumn(
            "remote_flag",
            F.when(
                F.lower(F.col("location")).contains("remote") | F.lower(F.col("work_type")).contains("remote"),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
    )

    # 2. Text Combination & Tokenization for PySpark Multi-Factor Recommendation
    # Combine job_title, role, skills, location, description for PySpark array Tokenizer
    df_text = df_exp.withColumn(
        "job_text",
        F.lower(
            F.concat_ws(
                " ",
                F.coalesce(F.col("job_title"), F.lit("")),
                F.coalesce(F.col("role"), F.lit("")),
                F.coalesce(F.col("skills"), F.lit("")),
                F.coalesce(F.col("location"), F.lit("")),
                F.coalesce(F.col("description"), F.lit(""))
            )
        )
    )

    # PySpark Split into word tokens array (lowercase words of length >= 2)
    df_processed = df_text.withColumn(
        "filtered_words",
        F.split(F.regexp_replace(F.col("job_text"), r"[^\w\s+#]", " "), r"\s+")
    )

    return df_processed
