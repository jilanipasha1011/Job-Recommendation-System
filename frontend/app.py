"""
Module: frontend/app.py
Phase 8: Streamlit User Interface for Resume-Based & Manual Job Recommendation System

Provides an interactive web dashboard for candidates to enter their job preferences manually
OR upload a PDF resume for automatic skill extraction and multi-factor job recommendation.
"""

import os
import sys
import json
import requests
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="Job Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
RECOMMEND_ENDPOINT = f"{API_BASE_URL}/recommend"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
PARSE_RESUME_ENDPOINT = f"{API_BASE_URL}/parse-resume"


# Custom CSS Styling for Premium Aesthetics (No icons)
st.markdown("""
<style>
    /* Global Container Padding & Colors */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Section Styling */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    .header-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0;
    }
    
    .source-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }
    .source-live { background-color: #0284c7; color: #ffffff; border: 1px solid #38bdf8; }
    .source-local { background-color: #475569; color: #f1f5f9; border: 1px solid #94a3b8; }

    /* Job Card Styling */
    .job-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .job-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.15);
        transform: translateY(-2px);
    }
    .job-card-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .job-card-subtitle {
        color: #38bdf8;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    
    /* Signal Badges & Pills */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-location { background-color: #0369a1; color: #e0f2fe; }
    .badge-experience { background-color: #334155; color: #f1f5f9; }
    .badge-worktype { background-color: #047857; color: #d1fae5; }
    .badge-salary { background-color: #b45309; color: #fef3c7; }
    .badge-date { background-color: #475569; color: #e2e8f0; }
    
    .skill-pill {
        display: inline-block;
        background-color: #1e293b;
        color: #cbd5e1;
        border: 1px solid #475569;
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        font-size: 0.78rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    
    .detected-skill-pill {
        display: inline-block;
        background-color: #0369a1;
        color: #f0f9ff;
        border: 1px solid #38bdf8;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Recommendation Score Container */
    .score-container {
        background-color: #0f172a;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        border: 1px solid #334155;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    .score-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .score-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .score-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> dict:
    """Checks whether FastAPI backend is responsive."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=3)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "degraded", "jobnews210_configured": False}
    except Exception:
        return {"status": "offline", "jobnews210_configured": False}


def parse_resume_via_api(pdf_bytes: bytes, filename: str) -> tuple[dict, str]:
    """
    Sends PDF bytes to FastAPI /parse-resume endpoint for skill & role extraction.
    Does NOT write files to disk.
    """
    try:
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        resp = requests.post(PARSE_RESUME_ENDPOINT, files=files, timeout=20)
        if resp.status_code == 200:
            return resp.json(), ""
        else:
            err_msg = resp.json().get("detail", "Error parsing PDF resume.")
            return {}, err_msg
    except Exception as e:
        return {}, f"Failed to connect to resume parser API: {str(e)}"


def fetch_recommendations(payload: dict) -> tuple[list, str, str]:
    """
    Sends request to FastAPI /recommend endpoint and returns (recommendations_list, data_source, error_message).
    """
    try:
        response = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=45)
        if response.status_code == 200:
            data = response.json()
            return data.get("recommendations", []), data.get("source", "jobnews210"), ""
        elif response.status_code == 422:
            return [], "", "Validation Error: Please enter a valid job role and skills."
        else:
            err_detail = response.json().get("detail", "Server error occurred")
            return [], "", f"API Error ({response.status_code}): {err_detail}"
    except requests.exceptions.ConnectionError:
        return [], "", f"Connection Failed: Could not connect to API server at {API_BASE_URL}. Ensure uvicorn backend is running."
    except requests.exceptions.Timeout:
        return [], "", "Request Timeout: Recommendation calculation took too long. Please try again."
    except Exception as e:
        return [], "", f"Unexpected Error: {str(e)}"


# =============================================================================
# Main Application Layout
# =============================================================================

def main():
    # Header Banner
    st.markdown("""
        <div class="header-box">
            <div class="header-title">Job Recommendation System</div>
            <div class="header-subtitle">
                Multi-Factor Relevance Ranking Powered by PySpark, FastAPI & Resume PDF Parser
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Check API Health Status
    health_info = check_api_health()
    if health_info.get("status") == "offline":
        st.warning(
            f"FastAPI Backend Offline: Cannot connect to backend server at `{API_BASE_URL}`. "
            "Please ensure `uvicorn api.main:app --reload` is active."
        )

    # Sidebar: Input Mode Selection (Manual Entry vs PDF Resume Upload)
    st.sidebar.header("Candidate Profile Mode")
    input_mode = st.sidebar.radio(
        "Choose Input Method:",
        options=["Upload Resume (PDF)", "Manual Entry"],
        index=0
    )
    st.sidebar.markdown("---")

    # Initialize Session State Variables
    if "parsed_profile" not in st.session_state:
        st.session_state["parsed_profile"] = None
    if "recommendations" not in st.session_state:
        st.session_state["recommendations"] = None
    if "data_source" not in st.session_state:
        st.session_state["data_source"] = ""
    if "search_error" not in st.session_state:
        st.session_state["search_error"] = ""

    # Mode 1: PDF Resume Upload Mode (Phase 8)
    if input_mode == "Upload Resume (PDF)":
        st.sidebar.subheader("Upload Resume PDF")
        uploaded_pdf = st.sidebar.file_uploader(
            "Upload candidate PDF resume:",
            type=["pdf"],
            help="Resumes are processed in-memory for privacy and never permanently stored."
        )

        if uploaded_pdf is not None:
            if st.sidebar.button("EXTRACT RESUME PROFILE", use_container_width=True):
                with st.spinner("Extracting text, detecting technical skills, and building profile..."):
                    pdf_bytes = uploaded_pdf.read()
                    profile, err = parse_resume_via_api(pdf_bytes, uploaded_pdf.name)
                    if err:
                        st.sidebar.error(err)
                    else:
                        st.session_state["parsed_profile"] = profile
                        st.session_state["search_error"] = ""

        parsed = st.session_state.get("parsed_profile")

        if parsed:
            st.markdown("### Resume Analysis Results")
            st.markdown("Review detected skills and inferred role before running job recommendations:")
            
            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                st.info(f"**Inferred Target Role**: {parsed.get('role', 'N/A')}")
            with col_p2:
                st.info(f"**Inferred Experience Level**: {parsed.get('experience', 'N/A')}")

            detected_skills = parsed.get("skills", [])
            st.markdown("#### Detected Skills:")
            if detected_skills:
                pills_html = "".join([f'<span class="detected-skill-pill">{s}</span>' for s in detected_skills])
                st.markdown(f'<div style="margin-bottom: 1.5rem;">{pills_html}</div>', unsafe_allow_html=True)
            else:
                st.warning("No standard technical skills were automatically detected. You can enter them below.")

            st.markdown("---")

            # Form to confirm / customize and run RECOMMEND JOBS
            with st.form(key="resume_recommendation_form"):
                st.subheader("Confirm Preferences for Job Recommendation")
                
                f_role = st.text_input("Target Job Role", value=parsed.get("role", "Machine Learning Engineer"))
                f_skills_str = st.text_area("Skills (Comma Separated)", value=", ".join(detected_skills))
                
                c1, c2 = st.columns(2)
                with c1:
                    exp_opts = ["Fresher", "1-3 Years", "3-5 Years", "5-10 Years", "10+ Years"]
                    default_exp_idx = exp_opts.index(parsed.get("experience")) if parsed.get("experience") in exp_opts else 0
                    f_exp = st.selectbox("Experience Level", options=exp_opts, index=default_exp_idx)
                with c2:
                    f_loc = st.text_input("Preferred Location", value="Bangalore")

                f_salary = st.text_input("Salary Preference (Optional)", value="$100K - $150K")
                f_top_n = st.slider("Number of Recommendations", min_value=1, max_value=30, value=10)

                recommend_btn = st.form_submit_button("RECOMMEND JOBS", use_container_width=True)

            if recommend_btn:
                skills_list = [s.strip() for s in f_skills_str.split(",") if s.strip()]
                payload = {
                    "role": f_role.strip(),
                    "skills": skills_list,
                    "experience": f_exp,
                    "location": f_loc.strip() if f_loc else None,
                    "salary_preference": f_salary.strip() if f_salary else None,
                    "top_n": f_top_n
                }

                with st.spinner("Executing recommendation engine..."):
                    recs, source, err = fetch_recommendations(payload)
                    st.session_state["recommendations"] = recs
                    st.session_state["data_source"] = source
                    st.session_state["search_error"] = err

    # Mode 2: Manual Profile Input Mode (Phase 7)
    else:
        st.sidebar.subheader("Manual Candidate Profile")
        with st.sidebar.form(key="manual_recommendation_form"):
            role_input = st.text_input(
                "Desired Job Role *",
                value="Machine Learning Engineer",
                placeholder="e.g. Data Scientist, Software Engineer"
            )

            skills_raw = st.text_area(
                "Technical Skills *",
                value="Python, PySpark, SQL, TensorFlow, Scikit-Learn",
                placeholder="e.g. Python, SQL, Docker (comma-separated)"
            )

            col_exp, col_loc = st.columns(2)
            with col_exp:
                experience_input = st.selectbox(
                    "Experience Level",
                    options=["Fresher", "1-3 Years", "3-5 Years", "5-10 Years", "10+ Years"],
                    index=0
                )
            with col_loc:
                location_input = st.text_input("Preferred Location", value="Bangalore")

            salary_input = st.text_input("Salary Preference (Optional)", value="$100K - $150K")
            top_n_input = st.slider("Number of Recommendations", min_value=1, max_value=30, value=10)

            submit_button = st.form_submit_button(label="RECOMMEND JOBS", use_container_width=True)

        if submit_button:
            if not role_input or not role_input.strip():
                st.error("Please enter a Desired Job Role.")
                return

            skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()]
            if not skills_list:
                st.error("Please provide at least one skill.")
                return

            payload = {
                "role": role_input.strip(),
                "skills": skills_list,
                "experience": experience_input,
                "location": location_input.strip() if location_input else None,
                "salary_preference": salary_input.strip() if salary_input else None,
                "top_n": top_n_input
            }

            with st.spinner("Executing recommendation engine..."):
                recs, source, err = fetch_recommendations(payload)
                st.session_state["recommendations"] = recs
                st.session_state["data_source"] = source
                st.session_state["search_error"] = err

    # Display Errors if any
    if st.session_state["search_error"]:
        st.error(st.session_state["search_error"])

    # Display Recommendation Results
    recommendations = st.session_state.get("recommendations")
    data_source = st.session_state.get("data_source", "")

    if recommendations is not None:
        if not recommendations:
            st.info("No job recommendations found matching your criteria. Try expanding your skills or location preference.")
        else:
            # Data Source Badge Indicator
            if data_source == "jobnews210":
                st.markdown('<div class="source-badge source-live">Data Source: JobNews210 Live Jobs</div>', unsafe_allow_html=True)
                st.subheader(f"LIVE JOBS FROM JOBNEWS210 ({len(recommendations)} Matches Found)")
            else:
                st.markdown('<div class="source-badge source-local">Data Source: Preprocessed Job Dataset</div>', unsafe_allow_html=True)
                st.subheader(f"Recommended Jobs ({len(recommendations)} Matches Found)")

            # Results Filters Row
            st.markdown("---")
            st.markdown("### Filter & Search Results")
            col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 2])

            with col_f1:
                min_score_filter = st.slider(
                    "Min Recommendation Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )

            with col_f2:
                all_locations = sorted(list(set(r["location"] for r in recommendations if r.get("location"))))
                location_filter = st.selectbox(
                    "Filter by Location",
                    options=["All Locations"] + all_locations,
                    index=0
                )

            with col_f3:
                keyword_filter = st.text_input(
                    "Search within results",
                    value="",
                    placeholder="Search by company, title, or skill..."
                )

            # Apply Interactive Filters
            filtered_recs = []
            for r in recommendations:
                if r["final_score"] < min_score_filter:
                    continue
                if location_filter != "All Locations" and r["location"] != location_filter:
                    continue
                if keyword_filter.strip():
                    kw = keyword_filter.lower().strip()
                    searchable_text = f"{r['job_title']} {r['company']} {r['location']} {r['skills']}".lower()
                    if kw not in searchable_text:
                        continue
                filtered_recs.append(r)

            st.markdown(f"**Showing {len(filtered_recs)} of {len(recommendations)} recommendations**")
            st.markdown("---")

            if not filtered_recs:
                st.warning("No jobs match your selected filter criteria. Try adjusting the score threshold or location filter.")
            else:
                # Render Job Cards
                for idx, job in enumerate(filtered_recs, 1):
                    signals = job.get("matching_signals", {})
                    final_score = job.get("final_score", 0.0)
                    skills_str = job.get("skills", "")
                    skills_array = [s.strip() for s in skills_str.split(",") if s.strip()]

                    with st.container():
                        st.markdown(f"""
                            <div class="job-card">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div>
                                        <div class="job-card-title">#{idx}. {job['job_title']}</div>
                                        <div class="job-card-subtitle">{job['company']}</div>
                                    </div>
                                    <div style="text-align: right;">
                                        <span class="badge badge-location">{job['location']}</span>
                                        <span class="badge badge-experience">{job['experience']}</span>
                                        <span class="badge badge-worktype">{job.get('work_type', 'Full-Time')}</span>
                                        <span class="badge badge-salary">{job.get('salary_range', 'Not Specified')}</span>
                                        <span class="badge badge-date">{job.get('posted_date', 'Recently Posted')}</span>
                                    </div>
                                </div>
                        """, unsafe_allow_html=True)

                        # Recommendation Score Section (Explicitly labeled Recommendation Score)
                        st.markdown(f"""
                            <div class="score-container">
                                <div class="score-header">
                                    <span class="score-label">Recommendation Score</span>
                                    <span class="score-val">{final_score:.4f}</span>
                                </div>
                        """, unsafe_allow_html=True)
                        st.progress(min(max(final_score, 0.0), 1.0))
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Breakdown of Matching Signals Grid
                        st.markdown("**Matching Signals Breakdown:**")
                        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                        with m_col1:
                            st.metric("Skill Match", f"{signals.get('skill_match', 0.0)*100:.1f}%")
                        with m_col2:
                            st.metric("Role Match", f"{signals.get('role_match', 0.0)*100:.1f}%")
                        with m_col3:
                            st.metric("Experience Match", f"{signals.get('experience_match', 0.0)*100:.1f}%")
                        with m_col4:
                            st.metric("Location Match", f"{signals.get('location_match', 0.0)*100:.1f}%")
                        with m_col5:
                            st.metric("Text Similarity", f"{signals.get('text_similarity', 0.0)*100:.1f}%")

                        # Required Skills Tags
                        if skills_array:
                            st.markdown("**Key Required Skills:**")
                            pills_html = "".join([f'<span class="skill-pill">{s}</span>' for s in skills_array[:12]])
                            st.markdown(f'<div style="margin-bottom: 1rem;">{pills_html}</div>', unsafe_allow_html=True)

                        # View / Apply Link using actual job_url
                        job_url = job.get("job_url") or f"https://www.google.com/search?q={job['job_title'].replace(' ', '+')}+{job['company'].replace(' ', '+')}"
                        
                        st.markdown(f"[View / Apply]({job_url})")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.write("")


if __name__ == "__main__":
    main()
