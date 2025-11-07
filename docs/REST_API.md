# REST API Documentation

## Overview

The REST API provides HTTP endpoints for duplicate detection, enabling programmatic access and integration with other systems.

## Quick Start

### Start Server

```bash
# Using Python
python -m duplicate_detector.api.rest.server

# Or using uvicorn directly
uvicorn duplicate_detector.api.rest.api:app --host 0.0.0.0 --port 8000
```

### Access API Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Endpoints

### POST /analyze

Upload PDF and start analysis.

**Request:**
- `file`: PDF file (multipart/form-data)
- `preset`: Configuration preset (fast/balanced/thorough)
- Optional parameters: `sim_threshold`, `phash_max_dist`, etc.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Analysis started...",
  "created_at": "2025-01-XX..."
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@paper.pdf" \
  -F "preset=balanced"
```

### GET /status/{job_id}

Get status of analysis job.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing|completed|failed",
  "progress": 0.75,
  "results": {...},
  "created_at": "...",
  "completed_at": "..."
}
```

### GET /results/{job_id}/download

Download results as TSV file.

### GET /results/{job_id}/json

Get results as JSON.

### DELETE /jobs/{job_id}

Delete job and associated files.

### GET /jobs

List all jobs.

### GET /health

Health check endpoint.

## Python Client Example

```python
import requests

# Start analysis
response = requests.post(
    "http://localhost:8000/analyze",
    files={"file": open("paper.pdf", "rb")},
    data={"preset": "balanced"}
)
job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/status/{job_id}").json()

# Get results when completed
if status["status"] == "completed":
    results = requests.get(f"http://localhost:8000/results/{job_id}/json").json()
    print(f"Found {results['total_pairs']} duplicates")
```

## Production Deployment

### With Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[dev]"
CMD ["uvicorn", "duplicate_detector.api.rest.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### With Gunicorn

```bash
gunicorn duplicate_detector.api.rest.api:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

## Authentication (Future)

For production, add authentication:

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/analyze")
async def analyze_pdf(
    token: str = Security(security),
    ...
):
    # Verify token
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    ...
```

## Rate Limiting (Future)

Add rate limiting with `slowapi`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_pdf(...):
    ...
```

