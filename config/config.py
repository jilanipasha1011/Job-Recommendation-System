"""
Module: config/config.py
Configuration file for Job Recommendation System.

Centralized weights and hyperparameters for multi-factor recommendation ranking.
"""

import os

# Base Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Dataset File Paths
RAW_DATASET_PATH = os.path.join(DATA_RAW_DIR, "job_descriptions.csv")
CLEANED_DATASET_PATH = os.path.join(DATA_PROCESSED_DIR, "jobs_cleaned.parquet")
FEATURES_DATASET_PATH = os.path.join(DATA_PROCESSED_DIR, "jobs_features.parquet")
VECTORS_DATASET_PATH = os.path.join(DATA_PROCESSED_DIR, "jobs_tfidf_vectors.parquet")
TFIDF_MODEL_DIR = os.path.join(MODELS_DIR, "tfidf")

# Spark Memory Configuration
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "4g"

# Multi-Factor Ranking Weights (Sum must equal 1.0)
WEIGHT_SKILL_MATCH = 0.40        # 40% Weight for skill alignment
WEIGHT_ROLE_MATCH = 0.25         # 25% Weight for job title/role match
WEIGHT_EXPERIENCE_MATCH = 0.15   # 15% Weight for experience level compatibility
WEIGHT_LOCATION_MATCH = 0.10     # 10% Weight for location/remote match
WEIGHT_TEXT_SIMILARITY = 0.10    # 10% Weight for overall TF-IDF text similarity

# Validate weights sum to 1.0
TOTAL_WEIGHT = (
    WEIGHT_SKILL_MATCH +
    WEIGHT_ROLE_MATCH +
    WEIGHT_EXPERIENCE_MATCH +
    WEIGHT_LOCATION_MATCH +
    WEIGHT_TEXT_SIMILARITY
)
assert abs(TOTAL_WEIGHT - 1.0) < 1e-5, f"Weights must sum to 1.0, got {TOTAL_WEIGHT}"
