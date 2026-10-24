"""
Script: recommender/test_recommendation.py
Phase 4: Recommendation Engine Verification Test

Tests the PySpark TF-IDF similarity recommendation pipeline using a sample user profile.
"""

import os
import sys
import time

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recommender.similarity import JobRecommender


def run_test():
    print("=" * 80)
    print(" PHASE 4: RECOMMENDATION ENGINE VERIFICATION TEST ")
    print("=" * 80)
    
    start_time = time.time()
    
    # Initialize Recommender Engine
    recommender = JobRecommender()
    
    # Define Sample User Profile
    user_profile = {
        "role": "Machine Learning Engineer",
        "skills": "Python, PySpark, SQL, TensorFlow",
        "experience": "Fresher",
        "location": "Bangalore"
    }
    
    print("\n--- Input User Profile ---")
    for k, v in user_profile.items():
        print(f"  * {k:<18}: {v}")
    print("-" * 80)
    
    print("\nGenerating Top 5 Job Recommendations...")
    recommendations = recommender.recommend(user_profile, top_n=5)
    
    print("\n" + "=" * 85)
    print(f" {'RANK':<5} | {'JOB ID':<18} | {'JOB TITLE':<25} | {'COMPANY':<20} | {'LOCATION':<12} | {'SCORE':<7}")
    print("=" * 85)
    
    for idx, job in enumerate(recommendations, 1):
        print(
            f" #{idx:<4} | "
            f"{str(job['job_id']):<18} | "
            f"{job['job_title'][:24]:<25} | "
            f"{job['company'][:19]:<20} | "
            f"{job['location'][:11]:<12} | "
            f"{job['similarity_score']:<7.4f}"
        )
    print("=" * 85)
    
    elapsed = time.time() - start_time
    print(f"\nRECOMMENDATION TEST COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 85)
    
    recommender.spark.stop()


if __name__ == "__main__":
    run_test()
