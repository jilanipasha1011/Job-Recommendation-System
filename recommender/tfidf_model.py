"""
Module: recommender/tfidf_model.py
Phase 4: PySpark MLlib TF-IDF Vectorization Model

This module builds, fits, and persists the TF-IDF feature extraction pipeline using PySpark MLlib:
1. Tokenizes job_text into lowercase words.
2. Removes standard English stop words.
3. Computes Term Frequency using HashingTF.
4. Fits Inverse Document Frequency (IDF) over the 1.61M job corpus.
5. Applies L2 Normalization so dot products directly compute Cosine Similarity.
6. Saves the fitted model under models/tfidf/ and exported vectors under data/processed/jobs_tfidf_vectors.parquet.
"""

import os
import sys
import time

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from src.spark_session import get_spark_session


def build_tfidf_pipeline(num_features: int = 10000) -> Pipeline:
    """
    Constructs an un-fitted PySpark MLlib Pipeline for TF-IDF vectorization and L2 normalization.
    """
    # 1. Tokenize job_text into words
    tokenizer = RegexTokenizer(
        inputCol="job_text",
        outputCol="words",
        pattern=r"\W+",
        toLowercase=True
    )
    
    # 2. Remove English stop words
    remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words"
    )
    
    # 3. Calculate Term Frequency (TF)
    hashing_tf = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=num_features
    )
    
    # 4. Calculate Inverse Document Frequency (IDF)
    idf = IDF(
        inputCol="raw_features",
        outputCol="features"
    )
    
    # 5. L2 Normalizer for fast cosine similarity dot product
    normalizer = Normalizer(
        inputCol="features",
        outputCol="norm_features",
        p=2.0
    )
    
    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, normalizer])
    return pipeline


def train_and_save_tfidf_model(
    input_path: str = "data/processed/jobs_features.parquet",
    model_dir: str = "models/tfidf",
    output_vectors_path: str = "data/processed/jobs_tfidf_vectors.parquet"
):
    """
    Loads feature dataset, fits TF-IDF model, saves model binaries and vector dataset.
    """
    print("=" * 80)
    print(" PHASE 4: TF-IDF MODEL TRAINING (PYSPARK MLLIB) ")
    print("=" * 80)
    
    start_time = time.time()
    spark = get_spark_session(app_name="Phase4_TFIDF_Training")
    
    print(f"\n[Step 1] Loading feature dataset from '{input_path}'...")
    df_features = spark.read.parquet(input_path)
    print(f"-> Corpus size: {df_features.count():,} jobs.")
    
    print("\n[Step 2] Constructing PySpark MLlib TF-IDF Pipeline...")
    pipeline = build_tfidf_pipeline(num_features=10000)
    
    print("\n[Step 3] Fitting IDF Model on Job Corpus...")
    pipeline_model = pipeline.fit(df_features)
    print("-> TF-IDF model fitting complete!")
    
    print(f"\n[Step 4] Saving fitted PipelineModel under '{model_dir}'...")
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    pipeline_model.write().overwrite().save(model_dir)
    print(f"-> Saved fitted model to {model_dir}")
    
    print(f"\n[Step 5] Transforming job corpus and saving vector dataset to '{output_vectors_path}'...")
    df_transformed = pipeline_model.transform(df_features)
    
    # Save transformed features with norm_features vector
    df_transformed.write.mode("overwrite").parquet(output_vectors_path)
    print("-> Transformed TF-IDF vectors written successfully!")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" TF-IDF MODEL TRAINING COMPLETED IN {elapsed:.2f} SECONDS ")
    print("=" * 80)
    
    spark.stop()


if __name__ == "__main__":
    train_and_save_tfidf_model()
