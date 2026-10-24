# Multi-Factor Job Recommendation System

A scalable, distributed Job Recommendation Engine powered by **Apache PySpark**, **FastAPI**, **Streamlit**, and **JobNews210 live job pipeline integration**.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Streamlit Web Interface   |
                      |     (http://localhost:8501) |
                      +--------------+--------------+
                                     |  HTTP REST
                                     v
                      +-----------------------------+
                      |       FastAPI Backend       |
                      |   (http://127.0.0.1:8000)   |
                      +-------+--------------+------+
                              |              |
              +---------------+              +---------------+
              v                                              v
+-------------------------------+             +-------------------------------+
|  JobNews210 Live API Client   |             |   MultiFactorRecommender      |
|  (Live job fetching/search)   |             |   (PySpark Cosine/TF-IDF)     |
+---------------+---------------+             +---------------+---------------+
                |                                             |
                v                                             v
+-------------------------------+             +-------------------------------+
|  Live Job PySpark Pipeline    |             |  Preprocessed Parquet Corpus  |
|  (In-memory feature vector)   |             |  (1.61M jobs indexed offline) |
+-------------------------------+             +-------------------------------+
```

---

## Prerequisites

1. **Python 3.10+** (tested on Python 3.14)
2. **Java 8 or 11/17** installed and added to `JAVA_HOME` / `PATH` (required by Apache Spark)
3. Windows Hadoop binary (`winutils.exe`) is already bundled in the `hadoop/bin` folder.

---

## 1. Quick Start: Running the Application

The system uses a 2-tier architecture:
1. **Backend API** (FastAPI + PySpark engine)
2. **Frontend UI** (Streamlit dashboard)

### Step 1: Activate Virtual Environment
Open your terminal (PowerShell or Command Prompt) in the project root:

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1
```

*(Or Command Prompt: `.\venv\Scripts\activate.bat`)*

---

### Step 2: Start the FastAPI Backend
In your first terminal window:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

> **Note**: During startup, PySpark initializes and pre-loads the model vectors into memory (~15-20 seconds).

---

### Step 3: Start the Streamlit Frontend
Open a **second terminal window**, activate the virtual environment, and run:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

The Streamlit UI will open automatically in your browser at:
- **URL**: [http://localhost:8501](http://localhost:8501)

You can:
- **Upload a PDF Resume**: Extracts candidate skills, titles, and generates recommendations automatically.
- **Manual Search**: Filter by Target Role, Skills, Experience Tier, Minimum Salary, and Preferred Location.

---

## 2. Environment Configuration (`.env`)

The project includes pre-configured settings in `.env`:

```ini
JOBNEWS210_API_KEY=your_key_here
JOBNEWS210_API_BASE_URL=https://api.jobnews210.com/v1
JOB_CACHE_TTL_SECONDS=300
JOBNEWS210_FALLBACK_TO_LOCAL=true
```

- When `JOBNEWS210_FALLBACK_TO_LOCAL=true`, the engine seamlessly falls back to the 1.61M preprocessed local job records if live API queries fail or encounter network issues.

---

## 3. Running Test Suites

Run the unit tests and API integration tests:

```powershell
# Run core system unit tests
python -m unittest discover tests

# Run API endpoint integration tests
python -m unittest api/test_api.py
```

---

## 4. PySpark Pipeline Scripts (Optional / Offline Training)

The offline preprocessed Parquet datasets and models are already generated in `data/processed/` and `models/tfidf/`. If you wish to re-run the offline pipeline:

1. **Phase 1: Dataset Inspection**:
   ```powershell
   python main_phase1.py
   ```
2. **Phase 2: Data Cleaning & Deduplication**:
   ```powershell
   python spark/data_cleaning.py
   ```
3. **Phase 3: Feature Engineering**:
   ```powershell
   python spark/feature_engineering.py
   ```
4. **Phase 4: TF-IDF Model Fitting & Vectorization**:
   ```powershell
   python recommender/tfidf_model.py
   ```
