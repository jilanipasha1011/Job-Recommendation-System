"""
Script: tests/test_jobnews210_client.py
Phase 11: Unit Test Suite for JobNews210 API Client

Mocks HTTP requests to validate authentication, timeout, rate limits, status codes,
and error handling without calling live endpoints or exposing secrets.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.jobnews210_client import JobNews210Client, JobNews210APIError


class TestJobNews210Client(unittest.TestCase):

    def setUp(self):
        self.mock_api_key = "test_dummy_key_12345"
        self.mock_base_url = "https://mock.jobnews210.com/v1"
        self.client = JobNews210Client(
            api_key=self.mock_api_key,
            base_url=self.mock_base_url,
            timeout=2,
            cache_ttl=60
        )

    def test_01_client_auth_headers(self):
        """Verify API key is included in headers without exposing credentials in logs."""
        headers = self.client._get_headers()
        self.assertEqual(headers["Authorization"], f"Bearer {self.mock_api_key}")
        self.assertEqual(headers["X-API-Key"], self.mock_api_key)
        self.assertTrue(self.client.is_configured())

    @patch("requests.request")
    def test_02_successful_api_response(self, mock_request):
        """Test successful GET /jobs response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "data": [
                {
                    "id": "jn_101",
                    "title": "Machine Learning Engineer",
                    "company": "AI Tech",
                    "location": "Bangalore"
                }
            ]
        }
        mock_request.return_value = mock_resp

        jobs = self.client.get_jobs(limit=10, page=1)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["id"], "jn_101")
        self.assertEqual(jobs[0]["title"], "Machine Learning Engineer")

    @patch("requests.request")
    def test_03_empty_job_response(self, mock_request):
        """Test API returning empty job list."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "success", "data": []}
        mock_request.return_value = mock_resp

        jobs = self.client.get_jobs()
        self.assertEqual(jobs, [])

    @patch("requests.request")
    def test_04_invalid_json_response(self, mock_request):
        """Test handling invalid non-JSON HTTP response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        mock_request.return_value = mock_resp

        with self.assertRaises(JobNews210APIError) as ctx:
            self.client._request("GET", "/jobs")
        self.assertIn("Invalid JSON", str(ctx.exception))

    @patch("requests.request")
    def test_05_request_timeout(self, mock_request):
        """Test handling request timeout."""
        import requests
        mock_request.side_effect = requests.exceptions.Timeout("Request timed out")

        with self.assertRaises(JobNews210APIError) as ctx:
            self.client._request("GET", "/jobs", max_retries=0)
        self.assertIn("timed out", str(ctx.exception).lower())

    @patch("requests.request")
    def test_06_http_401_unauthorized(self, mock_request):
        """Test handling HTTP 401 Unauthorized status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_request.return_value = mock_resp

        with self.assertRaises(JobNews210APIError) as ctx:
            self.client._request("GET", "/jobs")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Authentication error", str(ctx.exception))

    @patch("requests.request")
    def test_07_http_403_forbidden(self, mock_request):
        """Test handling HTTP 403 Forbidden status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_request.return_value = mock_resp

        with self.assertRaises(JobNews210APIError) as ctx:
            self.client._request("GET", "/jobs")
        self.assertEqual(ctx.exception.status_code, 403)

    @patch("requests.request")
    def test_08_http_429_rate_limit(self, mock_request):
        """Test handling HTTP 429 Rate Limit status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_request.return_value = mock_resp

        with self.assertRaises(JobNews210APIError) as ctx:
            self.client._request("GET", "/jobs", max_retries=0)
        self.assertEqual(ctx.exception.status_code, 429)

    @patch("requests.request")
    def test_09_http_500_server_error(self, mock_request):
        """Test handling HTTP 500 Server Error status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_request.return_value = mock_resp

        with self.assertRaises(JobNews210APIError) as ctx:
            self.client._request("GET", "/jobs", max_retries=0)
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("requests.request")
    def test_10_pagination_parameters(self, mock_request):
        """Test page and limit pagination query parameters passed to API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_request.return_value = mock_resp

        self.client.search_jobs(role="ML Engineer", limit=25, page=2)
        args, kwargs = mock_request.call_args
        params = kwargs.get("params", {})
        self.assertEqual(params.get("limit"), 25)
        self.assertEqual(params.get("page"), 2)


if __name__ == "__main__":
    unittest.main()
