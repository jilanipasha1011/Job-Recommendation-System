"""
Script: tests/test_resume_processing.py
Phase 8: Unit Test Suite for PDF Resume Text Extraction & Profile Building

Tests resume parsing, text cleaning, skill extraction taxonomy, role inference,
and candidate profile construction without storing resume files on disk.
"""

import os
import sys
import unittest

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resume.parser import clean_resume_text
from resume.skill_extractor import extract_skills_from_text
from resume.profile_builder import (
    infer_role_from_text,
    infer_experience_from_text,
    build_profile_from_resume_text
)


class TestResumeProcessing(unittest.TestCase):

    def setUp(self):
        self.sample_resume_text = """
        John Doe
        Email: john.doe@example.com | Phone: +1234567890
        
        OBJECTIVE:
        Experienced Machine Learning Engineer seeking senior role.
        
        SKILLS:
        Programming: Python, SQL, Java, C++
        Machine Learning: TensorFlow, PySpark, Scikit-Learn, Deep Learning, NLP
        Data & Cloud: AWS, Docker, PostgreSQL, Pandas, Git
        
        EXPERIENCE:
        ML Engineer at Tech Corp (3 years of experience)
        - Developed PySpark data pipelines and TensorFlow models.
        - Deployed microservices using Docker and FastAPI on AWS.
        """

    def test_01_clean_resume_text(self):
        """Test normalization of dirty text strings."""
        raw_text = "John   Doe \r\n\t  Machine   Learning \n\n Engineer  "
        cleaned = clean_resume_text(raw_text)
        self.assertEqual(cleaned, "John Doe Machine Learning Engineer")

    def test_02_extract_skills(self):
        """Test skill extraction taxonomy on resume text."""
        skills = extract_skills_from_text(self.sample_resume_text)
        self.assertIn("Python", skills)
        self.assertIn("PySpark", skills)
        self.assertIn("SQL", skills)
        self.assertIn("TensorFlow", skills)
        self.assertIn("Scikit-Learn", skills)
        self.assertIn("AWS", skills)
        self.assertIn("Docker", skills)

    def test_03_infer_role(self):
        """Test job role inference from text."""
        skills = ["Python", "PySpark", "TensorFlow"]
        role = infer_role_from_text(self.sample_resume_text, skills)
        self.assertEqual(role, "Machine Learning Engineer")

    def test_04_infer_experience(self):
        """Test experience level inference from text."""
        exp = infer_experience_from_text(self.sample_resume_text)
        self.assertEqual(exp, "1-3 Years")

    def test_05_build_profile(self):
        """Test complete candidate profile dictionary construction."""
        profile = build_profile_from_resume_text(self.sample_resume_text, default_location="Bangalore")
        self.assertEqual(profile["role"], "Machine Learning Engineer")
        self.assertIn("Python", profile["skills"])
        self.assertEqual(profile["location"], "Bangalore")
        self.assertGreater(profile["raw_text_length"], 0)


if __name__ == "__main__":
    unittest.main()
