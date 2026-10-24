"""
Module: resume/profile_builder.py
Phase 8: Automatic Candidate User Profile Construction from Resume Text

Infers candidate target role, experience level, and combines extracted skills
to form a canonical candidate profile dictionary for the recommendation engine.
"""

import re
from typing import Dict, Any, List
from resume.skill_extractor import extract_skills_from_text


# Predefined Job Role Keywords for Resume Classification
ROLE_KEYWORDS = [
    ("Machine Learning Engineer", ["machine learning engineer", "ml engineer", "ai engineer", "deep learning engineer"]),
    ("Data Scientist", ["data scientist", "data science", "nlp engineer", "ai researcher"]),
    ("Data Analyst", ["data analyst", "business analyst", "bi analyst", "analytics engineer"]),
    ("Data Engineer", ["data engineer", "big data engineer", "pyspark engineer", "etl developer"]),
    ("Software Engineer", ["software engineer", "software developer", "backend engineer", "full stack engineer"]),
    ("DevOps Engineer", ["devops engineer", "site reliability engineer", "sre", "cloud engineer"]),
    ("Solution Architect", ["solution architect", "cloud architect", "software architect"])
]


def infer_role_from_text(text: str, skills: List[str]) -> str:
    """
    Infers candidate job role from resume text and extracted skills.

    Parameters:
        text (str): Resume text
        skills (List[str]): List of extracted skills

    Returns:
        str: Inferred job role title
    """
    if not text:
        return "Machine Learning Engineer"

    text_lower = text.lower()

    # 1. Direct role keyword match
    for role_title, keywords in ROLE_KEYWORDS:
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return role_title

    # 2. Skill-based role inference fallback
    skills_set = set(s.lower() for s in skills)
    if {"pyspark", "tensorflow", "pytorch", "machine learning"}.intersection(skills_set):
        return "Machine Learning Engineer"
    elif {"tableau", "power bi", "excel", "data analysis"}.intersection(skills_set):
        return "Data Analyst"
    elif {"hadoop", "airflow", "kafka", "spark"}.intersection(skills_set):
        return "Data Engineer"
    elif {"react", "node.js", "django", "fastapi"}.intersection(skills_set):
        return "Software Engineer"
    elif {"aws", "docker", "kubernetes", "terraform"}.intersection(skills_set):
        return "DevOps Engineer"

    return "Machine Learning Engineer"


def infer_experience_from_text(text: str) -> str:
    """
    Infers experience level/years from resume text.

    Parameters:
        text (str): Resume text

    Returns:
        str: Experience category ('Fresher', '1-3 Years', '3-5 Years', '5-10 Years', '10+ Years')
    """
    if not text:
        return "Fresher"

    text_lower = text.lower()

    # Check for Fresher / Entry Level / Student terms
    if any(k in text_lower for k in ["fresher", "intern", "graduate", "student", "entry level"]):
        return "Fresher"

    # Search for numeric years of experience patterns
    match = re.search(r"(\d+)\+?\s*(?:years|yrs)\s*(?:of)?\s*(?:experience|exp)?", text_lower)
    if match:
        years = int(match.group(1))
        if years == 0:
            return "Fresher"
        elif 1 <= years <= 3:
            return "1-3 Years"
        elif 3 < years <= 5:
            return "3-5 Years"
        elif 5 < years <= 10:
            return "5-10 Years"
        else:
            return "10+ Years"

    # Check for Senior / Lead indicators
    if "senior" in text_lower or "lead" in text_lower or "principal" in text_lower:
        return "5-10 Years"

    return "Fresher"


def build_profile_from_resume_text(
    text: str,
    default_location: str = "Bangalore"
) -> Dict[str, Any]:
    """
    Constructs a complete candidate profile object from extracted resume text.

    Parameters:
        text (str): Cleaned resume text string
        default_location (str): Default target location

    Returns:
        Dict[str, Any]: Profile dictionary containing role, skills, experience, location
    """
    extracted_skills = extract_skills_from_text(text)
    inferred_role = infer_role_from_text(text, extracted_skills)
    inferred_experience = infer_experience_from_text(text)

    return {
        "role": inferred_role,
        "skills": extracted_skills,
        "skills_str": ", ".join(extracted_skills),
        "experience": inferred_experience,
        "location": default_location,
        "raw_text_length": len(text)
    }
