"""
Script: api/test_api.py
Phase 6: Comprehensive FastAPI Endpoint & Recommendation Engine Test Suite
"""

import os
import sys
import unittest

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app


class TestFastAPIBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize FastAPI TestClient with Lifespan startup event."""
        cls.client = TestClient(app)
        # Trigger lifespan startup
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        """Trigger lifespan shutdown event."""
        cls.client.__exit__(None, None, None)

    def test_01_root_endpoint(self):
        """Test GET / root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("message", json_data)
        self.assertEqual(json_data["status"], "online")

    def test_02_health_endpoint(self):
        """Test GET /health health status endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "healthy")
        self.assertTrue(json_data["model_loaded"])
        self.assertTrue(json_data["spark_session_active"])
        self.assertGreater(json_data["total_jobs_indexed"], 0)

    def test_03_list_jobs_endpoint(self):
        """Test GET /jobs paginated jobs listing."""
        response = self.client.get("/jobs?page=1&size=5")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("total_jobs", json_data)
        self.assertEqual(json_data["page"], 1)
        self.assertEqual(len(json_data["jobs"]), 5)

    def test_04_get_job_by_id_success(self):
        """Test GET /jobs/{job_id} for a valid job ID."""
        # Get a valid job ID from list endpoint
        list_res = self.client.get("/jobs?page=1&size=1")
        sample_job = list_res.json()["jobs"][0]
        job_id = sample_job["job_id"]

        response = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(response.status_code, 200)
        job_data = response.json()
        self.assertEqual(str(job_data["job_id"]), str(job_id))
        self.assertIn("job_title", job_data)
        self.assertIn("company", job_data)

    def test_05_get_job_by_id_not_found(self):
        """Test GET /jobs/{job_id} with invalid job ID returns 404."""
        response = self.client.get("/jobs/non_existent_job_99999999999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("detail", response.json())

    def test_06_recommend_endpoint_success(self):
        """Test POST /recommend with complete user candidate profile."""
        payload = {
            "role": "Machine Learning Engineer",
            "skills": ["Python", "PySpark", "SQL", "TensorFlow"],
            "experience": "Fresher",
            "location": "Bangalore",
            "top_n": 5
        }
        response = self.client.post("/recommend", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("recommendations", json_data)
        recs = json_data["recommendations"]
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 5)

        # Inspect fields of first recommendation
        first_rec = recs[0]
        self.assertIn("job_id", first_rec)
        self.assertIn("job_title", first_rec)
        self.assertIn("company", first_rec)
        self.assertIn("location", first_rec)
        self.assertIn("matching_signals", first_rec)
        self.assertIn("final_score", first_rec)

        signals = first_rec["matching_signals"]
        self.assertIn("skill_match", signals)
        self.assertIn("role_match", signals)
        self.assertIn("experience_match", signals)
        self.assertIn("location_match", signals)
        self.assertIn("text_similarity", signals)

    def test_07_recommend_missing_optional_fields(self):
        """Test POST /recommend handling missing optional fields (experience & location)."""
        payload = {
            "role": "Data Analyst",
            "skills": ["Python", "SQL"],
            "top_n": 3
        }
        response = self.client.post("/recommend", json=payload)
        self.assertEqual(response.status_code, 200)
        recs = response.json()["recommendations"]
        self.assertGreater(len(recs), 0)

    def test_08_recommend_validation_empty_role(self):
        """Test POST /recommend validation fails for empty role."""
        payload = {
            "role": "   ",
            "skills": ["Python"],
            "top_n": 5
        }
        response = self.client.post("/recommend", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_09_recommend_validation_invalid_top_n(self):
        """Test POST /recommend validation fails for top_n > 100."""
        payload = {
            "role": "Data Scientist",
            "skills": ["Python"],
            "top_n": 200
        }
        response = self.client.post("/recommend", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
