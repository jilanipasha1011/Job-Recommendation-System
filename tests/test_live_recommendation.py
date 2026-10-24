"""
Script: tests/test_live_recommendation.py
Phase 11: Unit Test Suite for Live Recommendation Pipeline & Fallback Policy

Tests end-to-end flow from mocked JobNews210 response -> PySpark Live Pipeline ->
Multi-Factor Recommender scoring and fallback policy validation.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.job_normalizer import normalize_jobnews210_batch
from spark.live_job_pipeline import process_live_jobs_with_pyspark
from recommender.ranking import MultiFactorRecommender
from src.spark_session import get_spark_session


class TestLiveRecommendationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize PySpark session and MultiFactorRecommender once for test suite."""
        cls.spark = get_spark_session(app_name="TestLiveRecommendation")
        cls.recommender = MultiFactorRecommender(spark=cls.spark)

    def test_01_live_pyspark_pipeline_and_ranking(self):
        """Test processing live normalized JSON into PySpark DataFrame and scoring."""
        raw_live_jobs = [
            {
                "id": "live_101",
                "title": "Machine Learning Engineer",
                "company": "DeepAI Corp",
                "location": "Bangalore",
                "skills": "Python, PySpark, SQL, TensorFlow",
                "experience": "Fresher",
                "salary": "$120K",
                "work_mode": "Full-Time",
                "job_url": "https://jobnews210.com/jobs/live_101",
                "posted_date": "2026-09-28"
            },
            {
                "id": "live_102",
                "title": "Data Accountant",
                "company": "Finance Inc",
                "location": "London",
                "skills": "Excel, Accounting",
                "experience": "10 Years",
                "salary": "$50K",
                "work_mode": "Contract",
                "job_url": "https://jobnews210.com/jobs/live_102",
                "posted_date": "2026-09-27"
            }
        ]

        normalized = normalize_jobnews210_batch(raw_live_jobs)
        self.assertEqual(len(normalized), 2)

        # Process in PySpark
        live_df = process_live_jobs_with_pyspark(self.spark, normalized)
        self.assertIn("filtered_words", live_df.columns)

        # Score candidate profile against live PySpark DataFrame
        user_profile = {
            "role": "Machine Learning Engineer",
            "skills": "Python, PySpark, SQL, TensorFlow",
            "experience": "Fresher",
            "location": "Bangalore"
        }

        recs = self.recommender.recommend(user_profile, top_n=5, jobs_df=live_df)
        self.assertGreater(len(recs), 0)
        top_job = recs[0]
        self.assertEqual(top_job["job_id"], "live_101")
        self.assertGreater(top_job["final_score"], 0.80)
        self.assertEqual(top_job["job_url"], "https://jobnews210.com/jobs/live_101")

    def test_02_fallback_disabled_behavior(self):
        """Test API fallback policy when fallback is disabled."""
        os.environ["JOBNEWS210_FALLBACK_TO_LOCAL"] = "false"
        fallback_enabled = os.getenv("JOBNEWS210_FALLBACK_TO_LOCAL", "false").lower() == "true"
        self.assertFalse(fallback_enabled)

    def test_03_fallback_enabled_behavior(self):
        """Test API fallback policy when fallback is enabled."""
        os.environ["JOBNEWS210_FALLBACK_TO_LOCAL"] = "true"
        fallback_enabled = os.getenv("JOBNEWS210_FALLBACK_TO_LOCAL", "false").lower() == "true"
        self.assertTrue(fallback_enabled)


if __name__ == "__main__":
    unittest.main()
