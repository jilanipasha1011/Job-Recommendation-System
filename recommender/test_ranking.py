"""
Script: recommender/test_ranking.py
Phase 5: Multi-Factor Recommendation Unit Tests

Validates exact and edge-case behavior of signal calculations:
1. Exact skill match
2. Partial skill match
3. Different role
4. Matching location
5. Different experience level/years
6. Missing / empty user profile fields
7. End-to-end PySpark MultiFactorRecommender execution
"""

import os
import sys
import unittest
import time

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recommender.ranking import calculate_single_job_score, MultiFactorRecommender


class TestMultiFactorRanking(unittest.TestCase):

    def setUp(self):
        self.sample_job = {
            "job_id": 101,
            "job_title": "Machine Learning Engineer",
            "role": "ML Specialist",
            "company": "TechCorp",
            "location": "Bangalore",
            "country": "India",
            "skills": "Python, PySpark, SQL, TensorFlow, Scikit-Learn",
            "min_experience_years": 0,
            "max_experience_years": 3,
            "remote_flag": 0,
            "text_similarity": 0.80
        }

    def test_exact_skill_match(self):
        """Test exact skill alignment."""
        user_profile = {
            "role": "Machine Learning Engineer",
            "skills": "Python, PySpark, SQL, TensorFlow, Scikit-Learn",
            "experience": "Fresher",
            "location": "Bangalore"
        }
        scores = calculate_single_job_score(user_profile, self.sample_job)
        self.assertEqual(scores["skill_match"], 1.0)
        self.assertEqual(scores["role_match"], 1.0)
        self.assertEqual(scores["location_match"], 1.0)
        self.assertEqual(scores["experience_match"], 1.0)
        self.assertGreater(scores["final_score"], 0.90)

    def test_partial_skill_match(self):
        """Test partial skill overlap."""
        user_profile = {
            "role": "Machine Learning Engineer",
            "skills": "Python, PySpark, Java, HTML",
            "experience": "Fresher",
            "location": "Bangalore"
        }
        scores = calculate_single_job_score(user_profile, self.sample_job)
        # 2 out of 4 skills matched (python, pyspark match) -> 0.50
        self.assertEqual(scores["skill_match"], 0.50)

    def test_different_role(self):
        """Test different role scenario."""
        user_profile = {
            "role": "Accountant",
            "skills": "Python, PySpark",
            "experience": "Fresher",
            "location": "Bangalore"
        }
        scores = calculate_single_job_score(user_profile, self.sample_job)
        self.assertEqual(scores["role_match"], 0.0)

    def test_matching_location(self):
        """Test location matching."""
        user_profile = {
            "role": "Machine Learning Engineer",
            "skills": "Python",
            "experience": "Fresher",
            "location": "Bangalore"
        }
        scores = calculate_single_job_score(user_profile, self.sample_job)
        self.assertEqual(scores["location_match"], 1.0)

    def test_different_experience(self):
        """Test candidate with experience higher or lower than required."""
        user_profile = {
            "role": "Machine Learning Engineer",
            "skills": "Python",
            "experience": "10 Years",  # Job requires 0 to 3
            "location": "Bangalore"
        }
        scores = calculate_single_job_score(user_profile, self.sample_job)
        # Over-qualified -> penalized score < 1.0
        self.assertLess(scores["experience_match"], 1.0)

    def test_missing_fields(self):
        """Test user profile with missing optional fields."""
        user_profile = {
            "skills": "Python, PySpark"
        }
        scores = calculate_single_job_score(user_profile, self.sample_job)
        self.assertEqual(scores["role_match"], 0.0)
        self.assertEqual(scores["location_match"], 0.0)
        self.assertEqual(scores["experience_match"], 0.0)
        self.assertGreater(scores["skill_match"], 0.0)
        self.assertGreater(scores["final_score"], 0.0)


def run_e2e_pyspark_test():
    print("=" * 85)
    print(" END-TO-END PYSPARK MULTI-FACTOR RECOMMENDATION TEST ")
    print("=" * 85)
    
    start_time = time.time()
    recommender = MultiFactorRecommender()
    
    user_profile = {
        "role": "Machine Learning Engineer",
        "skills": "Python, PySpark, SQL, TensorFlow",
        "experience": "Fresher",
        "location": "Bangalore"
    }
    
    print("\n--- Input Candidate Profile ---")
    for k, v in user_profile.items():
        print(f"  * {k:<18}: {v}")
    print("-" * 85)
    
    print("\nGenerating Top 5 Multi-Factor Job Recommendations...")
    recommendations = recommender.recommend(user_profile, top_n=5)
    
    print("\n" + "=" * 115)
    print(
        f" {'RANK':<5} | {'JOB ID':<16} | {'JOB TITLE':<22} | {'COMPANY':<18} | "
        f"{'SKILL':<6} | {'ROLE':<6} | {'EXP':<6} | {'LOC':<6} | {'TEXT':<6} | {'FINAL SCORE':<11}"
    )
    print("=" * 115)
    
    for idx, job in enumerate(recommendations, 1):
        print(
            f" #{idx:<4} | "
            f"{str(job['job_id']):<16} | "
            f"{job['title'][:21]:<22} | "
            f"{job['company'][:17]:<18} | "
            f"{job['skill_match']:<6.2f} | "
            f"{job['role_match']:<6.2f} | "
            f"{job['experience_match']:<6.2f} | "
            f"{job['location_match']:<6.2f} | "
            f"{job['text_similarity']:<6.2f} | "
            f"{job['final_score']:<11.4f}"
        )
    print("=" * 115)
    
    elapsed = time.time() - start_time
    print(f"\nE2E PYSPARK TEST COMPLETED IN {elapsed:.2f} SECONDS")
    print("=" * 115)
    
    recommender.spark.stop()


if __name__ == "__main__":
    print("Running Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiFactorRanking)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\nAll Unit Tests Passed!")
        run_e2e_pyspark_test()
    else:
        sys.exit(1)
