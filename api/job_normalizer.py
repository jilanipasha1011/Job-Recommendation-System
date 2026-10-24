"""
Module: api/job_normalizer.py
Phase 11: Data Normalization for JobNews210 Platform Responses

Maps raw JSON job items returned by JobNews210 API into the recommendation engine's
canonical job dictionary schema.
"""

from typing import Dict, Any, List, Optional


def normalize_jobnews210_job(raw_job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a single JobNews210 raw job object into the system's canonical dictionary structure.

    Canonical Schema:
    - job_id (str or int)
    - title (str)
    - company (str)
    - location (str)
    - skills (str)
    - experience (str)
    - salary (str)
    - description (str)
    - job_url (str)
    - posted_date (str)
    - work_mode (str)
    """
    if not isinstance(raw_job, dict):
        return {}

    # Extract Job ID
    job_id = (
        raw_job.get("job_id") or
        raw_job.get("id") or
        raw_job.get("jobId") or
        raw_job.get("_id") or
        "N/A"
    )

    # Extract Job Title
    title = (
        raw_job.get("title") or
        raw_job.get("job_title") or
        raw_job.get("jobTitle") or
        raw_job.get("role") or
        "Untitled Job"
    )

    # Extract Company
    company = (
        raw_job.get("company") or
        raw_job.get("company_name") or
        raw_job.get("companyName") or
        raw_job.get("employer") or
        "Unknown Company"
    )

    # Extract Location
    location = (
        raw_job.get("location") or
        raw_job.get("city") or
        raw_job.get("country") or
        raw_job.get("job_location") or
        "Not Specified"
    )

    # Extract Skills
    raw_skills = raw_job.get("skills") or raw_job.get("required_skills") or raw_job.get("tags") or ""
    if isinstance(raw_skills, list):
        skills = ", ".join([str(s).strip() for s in raw_skills if s])
    else:
        skills = str(raw_skills).strip()

    # Extract Experience
    experience = (
        raw_job.get("experience") or
        raw_job.get("experience_level") or
        raw_job.get("experience_years") or
        raw_job.get("experienceRequired") or
        "Not Specified"
    )

    # Extract Salary
    salary = (
        raw_job.get("salary") or
        raw_job.get("salary_range") or
        raw_job.get("salaryRange") or
        raw_job.get("compensation") or
        "Not Specified"
    )

    # Extract Description
    description = (
        raw_job.get("description") or
        raw_job.get("job_description") or
        raw_job.get("jobDescription") or
        raw_job.get("summary") or
        ""
    )

    # Extract Job URL / Application URL
    job_url = (
        raw_job.get("job_url") or
        raw_job.get("url") or
        raw_job.get("apply_url") or
        raw_job.get("applyUrl") or
        raw_job.get("link") or
        None
    )

    # Extract Posted Date
    posted_date = (
        raw_job.get("posted_date") or
        raw_job.get("created_at") or
        raw_job.get("date_posted") or
        raw_job.get("job_posting_date") or
        "Recently Posted"
    )

    # Extract Work Mode / Remote Flag
    work_mode = (
        raw_job.get("work_mode") or
        raw_job.get("work_type") or
        raw_job.get("workType") or
        ("Remote" if raw_job.get("remote_flag") == 1 or "remote" in str(location).lower() else "Full-Time")
    )

    return {
        "job_id": job_id,
        "title": title,
        "company": company,
        "location": location,
        "skills": skills,
        "experience": experience,
        "salary": salary,
        "description": description,
        "job_url": job_url,
        "posted_date": posted_date,
        "work_mode": work_mode
    }


def normalize_jobnews210_batch(raw_jobs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes a list of raw JobNews210 job objects.
    """
    if not raw_jobs_list or not isinstance(raw_jobs_list, list):
        return []

    normalized_jobs = []
    for item in raw_jobs_list:
        norm = normalize_jobnews210_job(item)
        if norm and norm.get("job_id"):
            normalized_jobs.append(norm)

    return normalized_jobs
