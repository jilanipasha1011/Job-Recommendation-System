"""
Script: tests/test_job_normalizer.py
Phase 11: Unit Test Suite for JobNews210 Data Normalizer
"""

import os
import sys
import unittest

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.job_normalizer import normalize_jobnews210_job, normalize_jobnews210_batch


class TestJobNormalizer(unittest.TestCase):

    def test_01_normalize_single_job_complete(self):
        """Test normalization of complete raw JobNews210 dictionary."""
        raw_job = {
            "id": "jn_999",
            "title": "Senior Data Scientist",
            "company_name": "Tech Corp",
            "location": "Bangalore",
            "required_skills": ["Python", "PySpark", "SQL"],
            "experience_level": "3-5 Years",
            "salary_range": "$120K-$150K",
            "description": "Building large-scale AI models.",
            "apply_url": "https://jobnews210.com/jobs/999",
            "created_at": "2026-09-28",
            "work_mode": "Hybrid"
        }

        norm = normalize_jobnews210_job(raw_job)
        self.assertEqual(norm["job_id"], "jn_999")
        self.assertEqual(norm["title"], "Senior Data Scientist")
        self.assertEqual(norm["company"], "Tech Corp")
        self.assertEqual(norm["location"], "Bangalore")
        self.assertIn("Python", norm["skills"])
        self.assertEqual(norm["experience"], "3-5 Years")
        self.assertEqual(norm["salary"], "$120K-$150K")
        self.assertEqual(norm["job_url"], "https://jobnews210.com/jobs/999")
        self.assertEqual(norm["work_mode"], "Hybrid")

    def test_02_normalize_missing_fields(self):
        """Test normalization handles missing optional fields safely without error."""
        raw_job = {
            "job_id": 505,
            "job_title": "Backend Engineer"
        }

        norm = normalize_jobnews210_job(raw_job)
        self.assertEqual(norm["job_id"], 505)
        self.assertEqual(norm["title"], "Backend Engineer")
        self.assertEqual(norm["company"], "Unknown Company")
        self.assertEqual(norm["location"], "Not Specified")
        self.assertEqual(norm["skills"], "")
        self.assertIsNone(norm["job_url"])

    def test_03_normalize_batch(self):
        """Test batch normalization of list of job objects."""
        raw_list = [
            {"id": "1", "title": "Job One"},
            {"id": "2", "title": "Job Two"}
        ]
        norm_list = normalize_jobnews210_batch(raw_list)
        self.assertEqual(len(norm_list), 2)
        self.assertEqual(norm_list[0]["job_id"], "1")
        self.assertEqual(norm_list[1]["job_id"], "2")

    def test_04_invalid_input_handling(self):
        """Test invalid non-dict or empty inputs return empty structures safely."""
        self.assertEqual(normalize_jobnews210_job(None), {})
        self.assertEqual(normalize_jobnews210_job("invalid_string"), {})
        self.assertEqual(normalize_jobnews210_batch(None), [])


if __name__ == "__main__":
    unittest.main()
