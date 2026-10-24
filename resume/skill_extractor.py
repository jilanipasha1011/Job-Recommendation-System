"""
Module: resume/skill_extractor.py
Phase 8: Automated Skill Extraction from Resume Text

Scans clean resume text against a comprehensive taxonomy of technical, domain,
and data engineering skills to extract an accurate list of candidate skills.
"""

import re
from typing import List


# Master Taxonomy of Technical & Domain Skills
SKILL_TAXONOMY = [
    # Programming Languages
    "Python", "PySpark", "SQL", "Java", "C++", "C#", "JavaScript", "TypeScript",
    "Go", "Scala", "R", "PHP", "HTML", "CSS", "Bash", "Shell",

    # Machine Learning & AI
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-Learn",
    "Keras", "OpenCV", "NLP", "Natural Language Processing", "Computer Vision",
    "Neural Networks", "MLOps", "Generative AI", "LLM", "Transformers",

    # Data Engineering & Big Data
    "Spark", "Hadoop", "Hive", "Kafka", "Airflow", "ETL", "Data Engineering",
    "Data Warehousing", "Snowflake", "Databricks", "BigQuery", "Redshift",

    # Cloud & Infrastructure / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Terraform",
    "Linux", "Git", "Cloud Computing",

    # Web & Backend Frameworks
    "FastAPI", "Flask", "Django", "React", "Node.js", "Express", "REST API",
    "Microservices", "GraphQL",

    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Oracle", "Cassandra",

    # Analytics & Data Science
    "Data Analysis", "Pandas", "NumPy", "Statistics", "Tableau", "Power BI",
    "Excel", "Data Visualization", "Matplotlib", "Seaborn"
]


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extracts technical skills from clean resume text using word-boundary regex matching.

    Parameters:
        text (str): Cleaned resume text string

    Returns:
        List[str]: Deduplicated list of detected skill names
    """
    if not text:
        return []

    detected_skills = []
    text_lower = text.lower()

    for skill in SKILL_TAXONOMY:
        skill_clean = skill.strip()
        # Escape special regex characters like ++ or # or .
        pattern = r"\b" + re.escape(skill_clean.lower()) + r"\b"
        
        # Handle special cases like C++ or C# or .NET
        if skill_clean.lower() in ["c++", "c#", ".net", "node.js", "ci/cd"]:
            pattern = re.escape(skill_clean.lower())

        if re.search(pattern, text_lower):
            detected_skills.append(skill_clean)

    # Deduplicate while preserving order
    unique_skills = list(dict.fromkeys(detected_skills))
    return unique_skills
