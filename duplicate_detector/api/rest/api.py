"""
REST API for Duplicate Detector using FastAPI.

Provides HTTP endpoints for duplicate detection, suitable for
SaaS deployment and programmatic access.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
import uuid
import shutil
import tempfile
from datetime import datetime

from duplicate_detector import DuplicateDetector
from duplicate_detector.models.config import DetectorConfig


app = FastAPI(
    title="Duplicate Detector API",
    description="REST API for scientific figure duplicate detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (use database in production)
jobs = {}


class AnalysisRequest(BaseModel):
    """Request model for duplicate detection."""
    preset: Optional[str] = Field("balanced", description="Configuration preset: fast, balanced, or thorough")
    sim_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="CLIP similarity threshold")
    phash_max_dist: Optional[int] = Field(None, ge=0, le=64, description="pHash max Hamming distance")
    ssim_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="SSIM threshold")
    use_phash_bundles: Optional[bool] = Field(None, description="Enable pHash bundles")
    use_orb_ransac: Optional[bool] = Field(None, description="Enable ORB-RANSAC")
    use_tier_gating: Optional[bool] = Field(None, description="Enable tier gating")


class AnalysisResponse(BaseModel):
    """Response model for analysis job."""
    job_id: str
    status: str
    message: str
    created_at: str


class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[float] = None
    results: Optional[dict] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Duplicate Detector API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to analyze"),
    preset: str = "balanced",
    sim_threshold: Optional[float] = None,
    phash_max_dist: Optional[int] = None,
    ssim_threshold: Optional[float] = None,
    use_phash_bundles: Optional[bool] = None,
    use_orb_ransac: Optional[bool] = None,
    use_tier_gating: Optional[bool] = None
):
    """
    Upload PDF and start duplicate detection analysis.
    
    Returns a job ID that can be used to check status and retrieve results.
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create temporary directory for this job
    job_dir = Path(tempfile.gettempdir()) / f"duplicate_detector_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    pdf_path = job_dir / file.filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create output directory
    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize job status
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "created_at": datetime.now().isoformat(),
        "results": None,
        "error": None
    }
    
    # Create configuration
    config = DetectorConfig.from_preset(preset)
    config.pdf_path = pdf_path
    config.output_dir = output_dir
    
    # Override with request parameters
    if sim_threshold is not None:
        config.duplicate_detection.sim_threshold = sim_threshold
    if phash_max_dist is not None:
        config.duplicate_detection.phash_max_dist = phash_max_dist
    if ssim_threshold is not None:
        config.duplicate_detection.ssim_threshold = ssim_threshold
    if use_phash_bundles is not None:
        config.feature_flags.use_phash_bundles = use_phash_bundles
    if use_orb_ransac is not None:
        config.feature_flags.use_orb_ransac = use_orb_ransac
    if use_tier_gating is not None:
        config.feature_flags.use_tier_gating = use_tier_gating
    
    # Start background task
    background_tasks.add_task(process_analysis, job_id, config)
    
    return AnalysisResponse(
        job_id=job_id,
        status="pending",
        message="Analysis started. Use /status/{job_id} to check progress.",
        created_at=jobs[job_id]["created_at"]
    )


async def process_analysis(job_id: str, config: DetectorConfig):
    """Background task to process analysis."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1
        
        # Run detection (config already has pdf_path and output_dir set)
        detector = DuplicateDetector(config=config)
        results = detector.analyze_pdf()
        
        jobs[job_id]["progress"] = 0.9
        
        # Store results
        jobs[job_id]["results"] = {
            "total_pairs": results.total_pairs,
            "tier_a_count": len(results.tier_a_pairs),
            "tier_b_count": len(results.tier_b_pairs),
            "tier_a_pairs": results.tier_a_pairs[:10] if results.tier_a_pairs else [],  # Limit to first 10
            "tier_b_pairs": results.tier_b_pairs[:10] if results.tier_b_pairs else [],
            "metadata": results.metadata
        }
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of an analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        results=job.get("results"),
        error=job.get("error"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at")
    )


@app.get("/results/{job_id}/download")
async def download_results(job_id: str):
    """Download results as TSV file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    results_file = Path(job["output_dir"]) / "final_merged_report.tsv"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        results_file,
        media_type="text/tab-separated-values",
        filename=f"duplicate_results_{job_id}.tsv"
    )


@app.get("/results/{job_id}/json")
async def get_results_json(job_id: str):
    """Get results as JSON."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if job["results"] is None:
        raise HTTPException(status_code=404, detail="Results not available")
    
    return JSONResponse(content=job["results"])


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    job_dir = Path(job["output_dir"]).parent
    
    # Clean up files
    if job_dir.exists():
        shutil.rmtree(job_dir)
    
    # Remove from jobs dict
    del jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"]
            }
            for job_id, job in jobs.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

