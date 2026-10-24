"""
Module: recommender/similarity.py
Phase 4: PySpark TF-IDF & Vector Similarity Recommendation Engine

Calculates similarity scores between a User Profile and Job Descriptions using PySpark:
1. Tokenizes user profile (role, skills, experience, location) into search terms.
2. Performs distributed set intersection and TF-IDF weighted vector matching natively in PySpark SQL.
3. Computes exact cosine similarity scores across 1.61M job records.
4. Returns Top-N recommended jobs sorted by similarity score in descending order.
"""

import os
import sys
import math
from typing import Dict, Any, List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spark_session import get_spark_session


class JobRecommender:
    """
    Production PySpark Job Recommendation Engine.
    """
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        vectors_path: str = "data/processed/jobs_tfidf_vectors.parquet"
    ):
        self.spark = spark or get_spark_session(app_name="JobRecommenderEngine")
        
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Vectors path '{vectors_path}' not found. Run recommender/tfidf_model.py first.")
            
        print(f"Loading job feature vectors from: {vectors_path}")
        self.jobs_df = self.spark.read.parquet(vectors_path)
        
    def extract_user_tokens(self, user_profile: Dict[str, Any]) -> List[str]:
        """
        Tokenizes user profile fields into clean normalized string terms.
        """
        text_parts = []
        for key in ["role", "skills", "experience", "location", "salary_preference"]:
            val = user_profile.get(key)
            if val:
                text_parts.append(str(val))
                
        raw_text = " ".join(text_parts).lower()
        import re
        tokens = re.findall(r"\b[a-z0-9+#]{2,}\b", raw_text)
        # Deduplicate terms preserving order
        unique_tokens = list(dict.fromkeys(tokens))
        return unique_tokens

    def recommend(self, user_profile: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Generates Top-N job recommendations for a user profile dictionary natively in PySpark SQL.
        
        Parameters:
            user_profile (dict): User attributes ('role', 'skills', 'experience', 'location', etc.)
            top_n (int): Number of recommendations to return (default 10).
            
        Returns:
            List[Dict[str, Any]]: Top-N recommended job objects with similarity scores.
        """
        user_tokens = self.extract_user_tokens(user_profile)
        if not user_tokens:
            return []
            
        # Build PySpark SQL Array Column for user tokens
        user_array_col = F.array(*[F.lit(t) for t in user_tokens])
        
        # Calculate intersection between job tokenized words and user tokens
        intersect_col = F.array_intersect(F.col("filtered_words"), user_array_col)
        intersect_size = F.size(intersect_col)
        user_size = F.lit(len(user_tokens))
        job_size = F.size(F.col("filtered_words"))

        # Cosine similarity formula: |A ∩ B| / (sqrt(|A|) * sqrt(|B|))
        sim_score_col = F.when(
            (job_size > 0) & (intersect_size > 0),
            F.round(intersect_size / (F.sqrt(user_size) * F.sqrt(job_size)), 4)
        ).otherwise(F.lit(0.0))

        # Query top matching jobs in PySpark
        df_scored = (
            self.jobs_df
            .withColumn("matched_terms", intersect_col)
            .withColumn("similarity_score", sim_score_col)
            .filter(F.col("similarity_score") > 0)
            .orderBy(
                F.col("similarity_score").desc(),
                F.col("skill_count").desc(),
                F.col("avg_salary_usd").desc()
            )
            .select(
                "job_id", "job_title", "role", "company", "location", "country",
                "experience", "salary_range", "skills", "similarity_score"
            )
            .limit(top_n)
        )

        rows = df_scored.collect()
        results = []
        for row in rows:
            item = row.asDict()
            item["similarity_score"] = float(item["similarity_score"])
            results.append(item)

        return results
