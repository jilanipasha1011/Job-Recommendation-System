"""
Module: api/main.py
Phase 11: FastAPI Backend Service for Multi-Factor Job Recommendation Engine
Integrates Live JobNews210 API Data Pipeline & PySpark Feature Engineering

Exposes recommendation engine, health status, and job listings via RESTful endpoints:
- GET / : Root API welcome & documentation link
- GET /health : Health status of API, JobNews210 integration, and PySpark session
- GET /jobs : Paginated list of jobs from preprocessed dataset
- GET /jobs/{job_id} : Detailed job information by ID
- POST /recommend : Multi-signal live job recommendation generation
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, status
from pydantic import BaseModel, Field, field_validator

# Add parent directory to sys.path for importing recommender modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recommender.ranking import MultiFactorRecommender
from api.jobnews210_client import JobNews210Client
from api.job_normalizer import normalize_jobnews210_batch
from spark.live_job_pipeline import process_live_jobs_with_pyspark
from resume.parser import extract_text_from_pdf_bytes
from resume.profile_builder import build_profile_from_resume_text
from pyspark.sql import functions as F


# Global Recommender & API Client Instances (Loaded once during application startup)
recommender_engine: Optional[MultiFactorRecommender] = None
jobnews_client: Optional[JobNews210Client] = None
total_jobs_count_cached: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager:
    Loads PySpark models, datasets & API clients ONCE during application startup.
    Prevents reloading models/data on every API request.
    """
    global recommender_engine, jobnews_client, total_jobs_count_cached
    print("\n" + "=" * 80)
    print(" STARTING FASTAPI BACKEND & INITIALIZING RECOMMENDATION ENGINE ")
    print("=" * 80)
    try:
        # Initialize MultiFactorRecommender (loads SparkSession and dataset once)
        recommender_engine = MultiFactorRecommender()
        # Initialize JobNews210 API Client
        jobnews_client = JobNews210Client()
        print("-> Recommendation Engine & JobNews210 Client initialized successfully!")
        print(f"-> JobNews210 API Configured: {jobnews_client.is_configured()}")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize recommendation engine on startup: {e}")
        recommender_engine = None

    yield

    # Application Shutdown Cleanup
    print("\n" + "=" * 80)
    print(" SHUTTING DOWN FASTAPI BACKEND & STOPPING SPARK SESSION ")
    print("=" * 80)
    if recommender_engine and recommender_engine.spark:
        try:
            recommender_engine.spark.stop()
            print("-> PySpark session stopped cleanly.")
        except Exception as e:
            print(f"-> Error stopping PySpark session: {e}")


# Initialize FastAPI App
app = FastAPI(
    title="Job Recommendation Engine API",
    description=(
        "Production Multi-Factor Job Recommendation & Relevance Ranking API "
        "built with FastAPI, PySpark, TF-IDF similarity vectors, and JobNews210 Live API Integration."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# =============================================================================
# Pydantic Schemas & Request/Response Models
# =============================================================================

class RecommendRequest(BaseModel):
    """
    Request payload for multi-factor job recommendation endpoint.
    """
    role: str = Field(
        ...,
        description="Target candidate job role or title (cannot be empty)",
        json_schema_extra={"example": "Machine Learning Engineer"}
    )
    skills: List[str] = Field(
        ...,
        description="List of candidate technical and domain skills",
        json_schema_extra={"example": ["Python", "PySpark", "SQL", "TensorFlow"]}
    )
    experience: Optional[str] = Field(
        default=None,
        description="Experience level or years (e.g., 'Fresher', 'Entry Level', '3 Years')",
        json_schema_extra={"example": "Fresher"}
    )
    location: Optional[str] = Field(
        default=None,
        description="Preferred job location or country (e.g., 'Bangalore', 'Remote', 'India')",
        json_schema_extra={"example": "Bangalore"}
    )
    salary_preference: Optional[str] = Field(
        default=None,
        description="Optional salary preference or range (e.g., '$80k', '$100K-$150K')",
        json_schema_extra={"example": "$80K"}
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of recommendations to return (1 to 100)",
        json_schema_extra={"example": 10}
    )

    @field_validator("role")
    @classmethod
    def validate_role_not_empty(cls, v: str) -> str:
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("role cannot be empty or whitespace only")
        return v.strip()

    @field_validator("skills")
    @classmethod
    def validate_skills_list(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("skills must be a list of strings")
        cleaned = [str(s).strip() for s in v if s and str(s).strip()]
        return cleaned


class MatchingSignals(BaseModel):
    """
    Breakdown of individual match signal scores for transparent recommendation scoring.
    """
    skill_match: float = Field(..., description="Skill alignment score (0.0 to 1.0)")
    role_match: float = Field(..., description="Role alignment score (0.0 to 1.0)")
    experience_match: float = Field(..., description="Experience level compatibility score (0.0 to 1.0)")
    location_match: float = Field(..., description="Location match score (0.0 to 1.0)")
    text_similarity: float = Field(..., description="TF-IDF overall text similarity score (0.0 to 1.0)")


class RecommendationItem(BaseModel):
    """
    Individual job recommendation item containing job information, matching signals, and final score.
    """
    job_id: Union[int, str] = Field(..., description="Unique job identifier")
    job_title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    skills: str = Field(..., description="Required skills string")
    experience: str = Field(..., description="Experience requirement string")
    salary_range: Optional[str] = Field(default="Not Specified", description="Salary range")
    work_type: Optional[str] = Field(default="Full-Time", description="Work arrangement type")
    job_portal: Optional[str] = Field(default="JobNews210", description="Job posting portal")
    job_url: Optional[str] = Field(default=None, description="External job application link")
    posted_date: Optional[str] = Field(default="Recently Posted", description="Date job was posted")
    matching_signals: MatchingSignals = Field(..., description="Breakdown of matching signals")
    final_score: float = Field(..., description="Composite multi-factor recommendation score")


class RecommendResponse(BaseModel):
    """
    Response schema for /recommend endpoint.
    """
    source: str = Field(..., description="Data source origin ('jobnews210' or 'local_dataset')")
    recommendations: List[RecommendationItem] = Field(
        ...,
        description="List of top recommended jobs ordered by final recommendation score"
    )


class JobSummary(BaseModel):
    """
    Summary schema for job listing endpoints.
    """
    job_id: Union[int, str]
    job_title: str
    role: Optional[str] = None
    company: str
    location: str
    country: Optional[str] = None
    experience: Optional[str] = None
    salary_range: Optional[str] = None
    skills: Optional[str] = None
    work_type: Optional[str] = None


class JobListResponse(BaseModel):
    total_jobs: int
    page: int
    size: int
    jobs: List[JobSummary]


class HealthResponse(BaseModel):
    status: str
    jobnews210_configured: bool
    model_loaded: bool
    spark_session_active: bool
    total_jobs_indexed: Optional[int] = None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["General"])
def read_root():
    """
    Root Endpoint.
    Returns basic API info, version, and Swagger UI documentation link.
    """
    return {
        "message": "Welcome to the Job Recommendation System API",
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def check_health():
    """
    Health Check Endpoint (Step 11.11).
    Returns operational status of the API service, JobNews210 configuration, and PySpark model.
    NEVER returns secrets or API keys.
    """
    global total_jobs_count_cached
    is_loaded = recommender_engine is not None and recommender_engine.jobs_df is not None
    spark_active = False

    if recommender_engine and recommender_engine.spark:
        try:
            spark_active = not recommender_engine.spark._jsc.sc().isStopped()
        except Exception:
            spark_active = False

    if is_loaded and total_jobs_count_cached is None:
        try:
            total_jobs_count_cached = recommender_engine.jobs_df.count()
        except Exception:
            total_jobs_count_cached = None

    is_jobnews_configured = jobnews_client.is_configured() if jobnews_client else False

    return {
        "status": "healthy" if (is_loaded and spark_active) else "degraded",
        "jobnews210_configured": is_jobnews_configured,
        "model_loaded": is_loaded,
        "spark_session_active": spark_active,
        "total_jobs_indexed": total_jobs_count_cached
    }


@app.post("/parse-resume", tags=["Resume Processing"])
async def parse_resume(file: UploadFile = File(...)):
    """
    POST /parse-resume (Phase 8): Accepts PDF resume upload in-memory,
    extracts skills, infers job role & experience, and builds a candidate user profile.
    Does NOT permanently store the uploaded PDF file on disk.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF resume files (.pdf) are supported"
        )
    try:
        content = await file.read()
        resume_text = extract_text_from_pdf_bytes(content)
        if not resume_text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text from the uploaded PDF resume."
            )
        profile = build_profile_from_resume_text(resume_text)
        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing resume PDF: {str(e)}"
        )


@app.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
def list_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size limit")
):
    """
    GET /jobs: Returns a paginated list of jobs from the preprocessed dataset.
    """
    global total_jobs_count_cached
    if not recommender_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine is not initialized"
        )

    try:
        skip = (page - 1) * size

        if total_jobs_count_cached is None:
            total_jobs_count_cached = recommender_engine.jobs_df.count()

        rows = (
            recommender_engine.jobs_df
            .select(
                "job_id",
                F.col("job_title").alias("job_title"),
                "role",
                "company",
                "location",
                "country",
                "experience",
                "salary_range",
                "skills",
                "work_type"
            )
            .limit(skip + size)
            .collect()
        )[skip:skip + size]

        jobs_list = [r.asDict() for r in rows]

        return {
            "total_jobs": total_jobs_count_cached or len(jobs_list),
            "page": page,
            "size": len(jobs_list),
            "jobs": jobs_list
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving job listings: {str(e)}"
        )


@app.get("/jobs/{job_id}", tags=["Jobs"])
def get_job_by_id(job_id: str):
    """
    GET /jobs/{job_id}: Returns full details for a specific job by Job ID.
    Raises 404 NOT FOUND if the job ID does not exist.
    """
    if not recommender_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine is not initialized"
        )

    try:
        df = recommender_engine.jobs_df

        # Filter by long integer or string job_id
        if job_id.isdigit():
            matched = df.filter((F.col("job_id") == int(job_id)) | (F.col("job_id") == job_id)).limit(1).collect()
        else:
            matched = df.filter(F.col("job_id") == job_id).limit(1).collect()

        if not matched:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job with ID '{job_id}' not found"
            )

        job_dict = matched[0].asDict()

        # Omit internal binary/vector columns for clean JSON output
        exclude_cols = ["words", "filtered_words", "raw_features", "features", "norm_features"]
        for key in exclude_cols:
            job_dict.pop(key, None)

        return job_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving job details for '{job_id}': {str(e)}"
        )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation Engine"])
def recommend_jobs(request: RecommendRequest):
    """
    POST /recommend (Step 11.6): Generates multi-factor job recommendations using Live JobNews210 API.

    Flow:
    User Request -> FastAPI -> JobNews210 API -> Live Jobs -> PySpark Processing
    -> Feature Engineering -> TF-IDF / Similarity -> Multi-Factor Ranking -> Top-N Recommendations.
    """
    if not recommender_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine is not initialized"
        )

    user_profile = {
        "role": request.role,
        "skills": ", ".join(request.skills) if isinstance(request.skills, list) else str(request.skills),
        "experience": request.experience or "",
        "location": request.location or "",
        "salary_preference": request.salary_preference or ""
    }

    source_used = "jobnews210"
    raw_results = []
    live_fetch_success = False

    # 1. Attempt Live JobNews210 API Retrieval if configured
    if jobnews_client and jobnews_client.is_configured():
        try:
            raw_jobs = jobnews_client.search_jobs(
                role=request.role,
                location=request.location or "",
                skills=request.skills,
                experience=request.experience or "",
                limit=max(request.top_n * 3, 30)
            )

            if raw_jobs:
                normalized_jobs = normalize_jobnews210_batch(raw_jobs)
                if normalized_jobs:
                    # Execute PySpark Live Job Processing & Feature Extraction
                    live_spark_df = process_live_jobs_with_pyspark(
                        spark=recommender_engine.spark,
                        normalized_jobs=normalized_jobs
                    )
                    # Run Multi-Factor Ranking Engine over Live PySpark DataFrame
                    raw_results = recommender_engine.recommend(
                        user_profile=user_profile,
                        top_n=request.top_n,
                        jobs_df=live_spark_df
                    )
                    live_fetch_success = True
                    source_used = "jobnews210"
        except Exception as e:
            print(f"[WARNING] Live JobNews210 API retrieval failed: {e}")
            live_fetch_success = False

    # 2. Check Fallback Policy (Step 11.9)
    fallback_enabled = os.getenv("JOBNEWS210_FALLBACK_TO_LOCAL", "false").lower() == "true"

    if not live_fetch_success:
        if fallback_enabled:
            print("[INFO] Falling back to preprocessed local dataset as fallback is enabled...")
            raw_results = recommender_engine.recommend(
                user_profile=user_profile,
                top_n=request.top_n
            )
            source_used = "local_dataset"
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "JobNews210 API live job retrieval failed or API is unconfigured, "
                    "and local dataset fallback is disabled (JOBNEWS210_FALLBACK_TO_LOCAL=false)."
                )
            )

    # 3. Format Response Items
    formatted_items = []
    for item in raw_results:
        portal = item.get("job_portal") or "JobNews210"
        job_title = item.get("title") or item.get("job_title", "N/A")
        company = item.get("company", "N/A")
        job_url = item.get("job_url") or f"https://www.google.com/search?q={job_title.replace(' ', '+')}+{company.replace(' ', '+')}+jobs"

        formatted_item = {
            "job_id": item.get("job_id"),
            "job_title": job_title,
            "company": company,
            "location": item.get("location", "N/A"),
            "skills": item.get("skills", "N/A"),
            "experience": item.get("experience", "N/A"),
            "salary_range": item.get("salary_range") or item.get("salary") or "Not Specified",
            "work_type": item.get("work_type") or item.get("work_mode") or "Full-Time",
            "job_portal": portal,
            "job_url": job_url,
            "posted_date": item.get("posted_date") or "Recently Posted",
            "matching_signals": {
                "skill_match": float(item.get("skill_match", 0.0)),
                "role_match": float(item.get("role_match", 0.0)),
                "experience_match": float(item.get("experience_match", 0.0)),
                "location_match": float(item.get("location_match", 0.0)),
                "text_similarity": float(item.get("text_similarity", 0.0))
            },
            "final_score": float(item.get("final_score", 0.0))
        }
        formatted_items.append(formatted_item)

    return {
        "source": source_used,
        "recommendations": formatted_items
    }
