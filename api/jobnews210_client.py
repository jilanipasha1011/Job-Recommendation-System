"""
Module: api/jobnews210_client.py
Phase 11: Security-Compliant API Client for JobNews210 Platform

Handles HTTP requests, authentication, rate limits, pagination, and error handling
for the JobNews210 live job data platform.

SECURITY RULES ENFORCED:
- API key is NEVER hard-coded or logged.
- Reads API key from environment variable JOBNEWS210_API_KEY via python-dotenv.
- Sanitizes authorization headers and log outputs.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Safe Logger
logger = logging.getLogger("JobNews210Client")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class JobNews210APIError(Exception):
    """Custom Exception for JobNews210 API failures without exposing credentials."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class JobNews210Client:
    """
    Production-ready API Client for JobNews210 Job Platform.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 10,
        cache_ttl: int = 300
    ):
        self.api_key = api_key or os.getenv("JOBNEWS210_API_KEY")
        self.base_url = (base_url or os.getenv("JOBNEWS210_API_BASE_URL", "https://api.jobnews210.com/v1")).rstrip("/")
        self.timeout = timeout
        
        try:
            self.cache_ttl = int(os.getenv("JOB_CACHE_TTL_SECONDS", cache_ttl))
        except ValueError:
            self.cache_ttl = cache_ttl

        # In-memory short-lived cache: { cache_key: (timestamp, list_of_jobs) }
        self._cache: Dict[str, tuple[float, Any]] = {}

    def _get_headers(self) -> Dict[str, str]:
        """Returns standard authentication headers without leaking secrets."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        return headers

    def is_configured(self) -> bool:
        """Returns True if valid API key and Base URL are configured."""
        return bool(self.api_key and self.api_key != "your_jobnews210_api_key" and self.base_url)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 2
    ) -> Any:
        """
        Executes safe HTTP requests with timeout, retry, and exception handling.
        Never logs API key or sensitive headers.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()

        # Sanitize log query params (omit any credentials if passed)
        safe_params = {k: v for k, v in (params or {}).items() if "key" not in k.lower() and "secret" not in k.lower()}
        logger.info(f"Sending {method} request to '{url}' with params: {safe_params}")

        attempts = 0
        while attempts <= max_retries:
            attempts += 1
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError:
                        raise JobNews210APIError("Invalid JSON response received from API server.", status_code=200)

                elif response.status_code in (401, 403):
                    logger.error(f"Authentication failure (HTTP {response.status_code}) for JobNews210 API.")
                    raise JobNews210APIError(
                        f"Authentication error (HTTP {response.status_code}): Invalid or unauthorized API key.",
                        status_code=response.status_code
                    )

                elif response.status_code == 429:
                    logger.warning(f"Rate limit exceeded (HTTP 429). Attempt {attempts}/{max_retries + 1}.")
                    if attempts <= max_retries:
                        time.sleep(2 * attempts)
                        continue
                    raise JobNews210APIError("Rate limit exceeded (HTTP 429). Please try again later.", status_code=429)

                elif response.status_code >= 500:
                    logger.warning(f"Server error (HTTP {response.status_code}). Attempt {attempts}/{max_retries + 1}.")
                    if attempts <= max_retries:
                        time.sleep(1 * attempts)
                        continue
                    raise JobNews210APIError(f"JobNews210 Server Error (HTTP {response.status_code}).", status_code=response.status_code)

                else:
                    raise JobNews210APIError(f"HTTP Request failed with status code {response.status_code}", status_code=response.status_code)

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout after {self.timeout}s. Attempt {attempts}/{max_retries + 1}.")
                if attempts <= max_retries:
                    continue
                raise JobNews210APIError(f"Request timed out after {self.timeout} seconds.")

            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error connecting to JobNews210 endpoint at {url}")
                raise JobNews210APIError(f"Failed to connect to JobNews210 API server at '{self.base_url}'.")

            except Exception as e:
                if isinstance(e, JobNews210APIError):
                    raise e
                raise JobNews210APIError(f"Unexpected error calling JobNews210 API: {str(e)}")

    def get_jobs(self, limit: int = 50, page: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves live job listings with pagination and short-lived caching.
        """
        cache_key = f"get_jobs_limit{limit}_page{page}"
        now = time.time()

        # Check Cache
        if cache_key in self._cache:
            ts, cached_jobs = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                logger.info(f"Returning {len(cached_jobs)} jobs from short-lived cache (TTL: {self.cache_ttl}s).")
                return cached_jobs

        params = {"limit": min(limit, 100), "page": max(page, 1)}
        try:
            data = self._request("GET", "/jobs", params=params)
        except Exception as e:
            logger.warning(f"API request failed: {e}")
            data = None

        jobs = self._extract_jobs_array(data)

        # Update Cache if valid response
        if jobs:
            self._cache[cache_key] = (now, jobs)

        return jobs

    def search_jobs(
        self,
        query: str = "",
        role: str = "",
        location: str = "",
        skills: Optional[List[str]] = None,
        experience: Optional[str] = None,
        limit: int = 50,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Searches live jobs using role, location, skills, and experience parameters.
        Uses in-memory cache to prevent redundant API requests.
        """
        skills_str = ",".join(skills) if skills else ""
        cache_key = f"search_{role}_{location}_{skills_str}_{experience}_{limit}_{page}"
        now = time.time()

        if cache_key in self._cache:
            ts, cached_jobs = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                logger.info(f"Returning {len(cached_jobs)} searched jobs from cache.")
                return cached_jobs

        params = {
            "q": query or role,
            "role": role,
            "location": location,
            "skills": skills_str,
            "experience": experience or "",
            "limit": min(limit, 100),
            "page": max(page, 1)
        }
        # Clean empty params
        params = {k: v for k, v in params.items() if v}

        try:
            data = self._request("GET", "/jobs/search", params=params)
        except JobNews210APIError as e:
            # Fallback to general get_jobs if /search route is not separate endpoint
            logger.info("Search route returned error, falling back to /jobs listing...")
            try:
                data = self._request("GET", "/jobs", params=params)
            except Exception:
                data = None
        except Exception as e:
            logger.warning(f"Search API call failed: {e}")
            data = None

        jobs = self._extract_jobs_array(data)

        if jobs:
            self._cache[cache_key] = (now, jobs)

        return jobs

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves details for a single job by Job ID.
        """
        cache_key = f"job_id_{job_id}"
        now = time.time()

        if cache_key in self._cache:
            ts, cached_job = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                return cached_job

        try:
            data = self._request("GET", f"/jobs/{job_id}")
            job = data.get("data") or data.get("job") or data if isinstance(data, dict) else None
            if job:
                self._cache[cache_key] = (now, job)
            return job
        except Exception as e:
            logger.warning(f"Error fetching job by ID '{job_id}': {e}")
            return None

    def _extract_jobs_array(self, data: Any) -> List[Dict[str, Any]]:
        """Extracts job list array from flexible JSON responses."""
        if not data:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ["data", "jobs", "results", "items", "job_list"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []
