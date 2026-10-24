"""
Module: recommender/ranking.py
Phase 5: Multi-Factor Job Recommendation & Relevance Ranking

Calculates transparent, multi-signal matching scores for job recommendation:
1. Skill Match (40%): Overlap between user skills and job skills.
2. Role Match (25%): Alignment between requested role and job title/role.
3. Experience Match (15%): Compatibility between candidate experience level/years and job requirements.
4. Location Match (10%): Match between candidate preferred location/country and job location (with remote support).
5. Text Similarity (10%): Overall TF-IDF / Token textual similarity.

Final Score represents relevance according to the configured weighted scoring system (not a probability).
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    WEIGHT_SKILL_MATCH,
    WEIGHT_ROLE_MATCH,
    WEIGHT_EXPERIENCE_MATCH,
    WEIGHT_LOCATION_MATCH,
    WEIGHT_TEXT_SIMILARITY,
    VECTORS_DATASET_PATH
)
from src.spark_session import get_spark_session


def parse_experience_to_years(exp_str: str) -> Optional[int]:
    """
    Parses user experience text into representative integer years.
    Returns None if missing or unparseable.
    """
    if not exp_str:
        return None
    s = str(exp_str).strip().lower()
    if not s:
        return None
    if "fresher" in s or "entry" in s or "intern" in s:
        return 0
    
    match = re.search(r"(\d+)", s)
    if match:
        return int(match.group(1))
    return None


def calculate_single_job_score(user_profile: Dict[str, Any], job_record: Dict[str, Any]) -> Dict[str, float]:
    """
    Pure Python single record scoring utility (useful for unit testing and instant evaluation).
    """
    # 1. Extract User Inputs safely
    user_skills_raw = str(user_profile.get("skills", "")).lower()
    user_skills = set(s.strip() for s in re.split(r"[,;]+", user_skills_raw) if s.strip())
    
    user_role = str(user_profile.get("role", "")).strip().lower()
    user_role_tokens = set(re.findall(r"\b[a-z0-9+#]{2,}\b", user_role))
    user_exp_years = parse_experience_to_years(user_profile.get("experience", ""))
    user_location = str(user_profile.get("location", "")).strip().lower()

    # 2. Extract Job Fields safely
    job_skills_str = str(job_record.get("skills", "")).lower()
    job_skills_set = set(s.strip() for s in re.split(r"[,;]+", job_skills_str) if s.strip())
    job_title = str(job_record.get("job_title", "")).lower()
    job_role = str(job_record.get("role", "")).lower()
    job_location = str(job_record.get("location", "")).lower()
    job_country = str(job_record.get("country", "")).lower()
    min_exp = job_record.get("min_experience_years")
    max_exp = job_record.get("max_experience_years")

    # --- Signal 1: Skill Match ---
    if user_skills and job_skills_set:
        matched_skills = user_skills.intersection(job_skills_set)
        skill_match = round(len(matched_skills) / len(user_skills), 4)
    else:
        skill_match = 0.0

    # --- Signal 2: Role Match ---
    if user_role and (job_title or job_role):
        if user_role in job_title or user_role in job_role or job_title in user_role:
            role_match = 1.0
        elif user_role_tokens:
            job_title_tokens = set(re.findall(r"\b[a-z0-9+#]{2,}\b", job_title + " " + job_role))
            matched_role_tokens = user_role_tokens.intersection(job_title_tokens)
            role_match = round(len(matched_role_tokens) / len(user_role_tokens), 4)
        else:
            role_match = 0.0
    else:
        role_match = 0.0

    # --- Signal 3: Experience Match ---
    if user_exp_years is not None and min_exp is not None and max_exp is not None:
        if min_exp <= user_exp_years <= max_exp:
            experience_match = 1.0
        elif user_exp_years < min_exp:
            experience_match = max(0.0, round(1.0 - 0.25 * (min_exp - user_exp_years), 4))
        else:
            experience_match = max(0.2, round(1.0 - 0.15 * (user_exp_years - max_exp), 4))
    else:
        experience_match = 0.0

    # --- Signal 4: Location Match ---
    if user_location:
        if user_location in job_location or user_location in job_country or job_location in user_location:
            location_match = 1.0
        elif "remote" in user_location or "remote" in job_location or job_record.get("remote_flag") == 1:
            location_match = 0.8
        else:
            location_match = 0.0
    else:
        location_match = 0.0

    # --- Signal 5: Text Similarity ---
    text_similarity = float(job_record.get("text_similarity", 0.0))

    # --- Composite Final Score ---
    final_score = round(
        WEIGHT_SKILL_MATCH * skill_match +
        WEIGHT_ROLE_MATCH * role_match +
        WEIGHT_EXPERIENCE_MATCH * experience_match +
        WEIGHT_LOCATION_MATCH * location_match +
        WEIGHT_TEXT_SIMILARITY * text_similarity,
        4
    )

    return {
        "skill_match": skill_match,
        "role_match": role_match,
        "experience_match": experience_match,
        "location_match": location_match,
        "text_similarity": text_similarity,
        "final_score": final_score
    }


class MultiFactorRecommender:
    """
    Production PySpark Multi-Factor Job Recommendation & Ranking Engine.
    """
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        vectors_path: str = VECTORS_DATASET_PATH
    ):
        self.spark = spark or get_spark_session(app_name="MultiFactorRecommenderEngine")
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Vectors path '{vectors_path}' not found.")
            
        self.jobs_df = self.spark.read.parquet(vectors_path)

    def recommend(
        self,
        user_profile: Dict[str, Any],
        top_n: int = 10,
        jobs_df: Optional[DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates multi-signal scores natively in PySpark SQL and returns Top-N recommendations.
        Supports custom live PySpark DataFrames (e.g. from JobNews210 API pipeline).
        """
        target_df = jobs_df if jobs_df is not None else self.jobs_df

        # Ensure optional columns exist in target_df before select
        available_cols = set(target_df.columns)
        if "job_url" not in available_cols:
            target_df = target_df.withColumn("job_url", F.lit(None).cast("string"))
        if "posted_date" not in available_cols:
            target_df = target_df.withColumn("posted_date", F.lit("Recently Posted").cast("string"))
        if "salary_range" not in available_cols:
            target_df = target_df.withColumn("salary_range", F.lit("Not Specified").cast("string"))
        if "work_type" not in available_cols:
            target_df = target_df.withColumn("work_type", F.lit("Full-Time").cast("string"))
        if "job_portal" not in available_cols:
            target_df = target_df.withColumn("job_portal", F.lit("Job Board").cast("string"))
        # Parse user skills
        user_skills_raw = str(user_profile.get("skills", "")).strip()
        user_skill_tokens = list(set(re.findall(r"\b[a-z0-9+#]{2,}\b", user_skills_raw.lower())))
        
        # Parse user role
        user_role_raw = str(user_profile.get("role", "")).strip()
        user_role_tokens = list(set(re.findall(r"\b[a-z0-9+#]{2,}\b", user_role_raw.lower())))
        
        # Parse user location
        user_loc_raw = str(user_profile.get("location", "")).strip().lower()
        
        # Parse user experience
        user_exp_years = parse_experience_to_years(user_profile.get("experience", ""))

        # 1. PySpark SQL Skill Match Column
        if user_skill_tokens:
            user_skill_arr = F.array(*[F.lit(s) for s in user_skill_tokens])
            skill_intersect = F.array_intersect(F.col("filtered_words"), user_skill_arr)
            skill_match_col = F.round(F.size(skill_intersect) / F.lit(len(user_skill_tokens)), 4)
        else:
            skill_match_col = F.lit(0.0)

        # 2. PySpark SQL Role Match Column
        if user_role_tokens:
            user_role_arr = F.array(*[F.lit(r) for r in user_role_tokens])
            role_intersect = F.array_intersect(F.col("filtered_words"), user_role_arr)
            role_match_col = F.when(
                F.lower(F.col("job_title")).contains(user_role_raw.lower()) |
                F.lower(F.col("role")).contains(user_role_raw.lower()),
                F.lit(1.0)
            ).otherwise(
                F.round(F.size(role_intersect) / F.lit(len(user_role_tokens)), 4)
            )
        else:
            role_match_col = F.lit(0.0)

        # 3. PySpark SQL Experience Match Column
        if user_exp_years is not None:
            exp_match_col = F.when(
                (F.col("min_experience_years") <= user_exp_years) & (user_exp_years <= F.col("max_experience_years")),
                F.lit(1.0)
            ).when(
                user_exp_years < F.col("min_experience_years"),
                F.greatest(F.lit(0.0), F.round(F.lit(1.0) - (F.lit(0.25) * (F.col("min_experience_years") - F.lit(user_exp_years))), 4))
            ).otherwise(
                F.greatest(F.lit(0.2), F.round(F.lit(1.0) - (F.lit(0.15) * (F.lit(user_exp_years) - F.col("max_experience_years"))), 4))
            )
        else:
            exp_match_col = F.lit(0.0)

        # 4. PySpark SQL Location Match Column
        if user_loc_raw:
            loc_match_col = F.when(
                F.lower(F.col("location")).contains(user_loc_raw) |
                F.lower(F.col("country")).contains(user_loc_raw),
                F.lit(1.0)
            ).when(
                F.lit("remote").contains(user_loc_raw) |
                F.lower(F.col("location")).contains("remote") |
                (F.col("remote_flag") == 1),
                F.lit(0.8)
            ).otherwise(F.lit(0.0))
        else:
            loc_match_col = F.lit(0.0)

        # 5. PySpark SQL Text Similarity Column
        all_user_terms = list(set(user_skill_tokens + user_role_tokens + ([user_loc_raw] if user_loc_raw else [])))
        if all_user_terms:
            user_all_arr = F.array(*[F.lit(t) for t in all_user_terms])
            all_intersect = F.array_intersect(F.col("filtered_words"), user_all_arr)
            text_sim_col = F.when(
                (F.size(F.col("filtered_words")) > 0) & (F.size(all_intersect) > 0),
                F.round(F.size(all_intersect) / (F.sqrt(F.lit(len(all_user_terms))) * F.sqrt(F.size(F.col("filtered_words")))), 4)
            ).otherwise(F.lit(0.0))
        else:
            text_sim_col = F.lit(0.0)

        # Composite final_score formula using configurable weights
        final_score_col = F.round(
            (F.lit(WEIGHT_SKILL_MATCH) * skill_match_col) +
            (F.lit(WEIGHT_ROLE_MATCH) * role_match_col) +
            (F.lit(WEIGHT_EXPERIENCE_MATCH) * exp_match_col) +
            (F.lit(WEIGHT_LOCATION_MATCH) * loc_match_col) +
            (F.lit(WEIGHT_TEXT_SIMILARITY) * text_sim_col),
            4
        )

        df_ranked = (
            target_df
            .withColumn("skill_match", skill_match_col)
            .withColumn("role_match", role_match_col)
            .withColumn("experience_match", exp_match_col)
            .withColumn("location_match", loc_match_col)
            .withColumn("text_similarity", text_sim_col)
            .withColumn("final_score", final_score_col)
            .filter(F.col("final_score") > 0)
            .orderBy(F.col("final_score").desc(), F.col("skill_match").desc())
            .select(
                "job_id",
                F.col("job_title").alias("title"),
                "company",
                "location",
                "skills",
                "experience",
                "salary_range",
                "work_type",
                "job_portal",
                "job_url",
                "posted_date",
                "text_similarity",
                "skill_match",
                "role_match",
                "experience_match",
                "location_match",
                "final_score"
            )
            .limit(top_n)
        )

        rows = df_ranked.collect()
        results = []
        for r in rows:
            item = r.asDict()
            for k in ["text_similarity", "skill_match", "role_match", "experience_match", "location_match", "final_score"]:
                item[k] = float(item[k])
            results.append(item)

        return results
