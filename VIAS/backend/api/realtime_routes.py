"""
Real-Time API Routes for VIAS
Provides endpoints for:
  - Real-time video processing with async job tracking
  - Job status monitoring and metrics
  - Server-Sent Events (SSE) for live streaming
  - Background task management
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
from datetime import datetime
import asyncio
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/realtime", tags=["realtime"])


# ============================================================================
# DATA MODELS
# ============================================================================

class ProcessingJob(BaseModel):
    """Represents a video processing job."""
    job_id: str
    status: str  # pending, processing, completed, failed
    file_name: str
    uploaded_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 100.0
    detections: int = 0
    tracks: int = 0
    activities: int = 0
    error: Optional[str] = None
    metrics: Dict = {}


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: str
    progress: float
    message: str


class ProgressUpdate(BaseModel):
    """Progress update event."""
    job_id: str
    progress: float
    timestamp: datetime
    message: str


# ============================================================================
# IN-MEMORY JOB STORE (Replace with database in production)
# ============================================================================

jobs_db: Dict[str, ProcessingJob] = {}


# ============================================================================
# REAL-TIME ENDPOINTS
# ============================================================================

@router.post("/process-video")
async def process_video_realtime(
    file_path: str,
    background_tasks: BackgroundTasks
) -> Dict:
    """
    Start async video processing job.
    
    Returns job ID immediately without blocking.
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Create job record
    job = ProcessingJob(
        job_id=job_id,
        status="pending",
        file_name=file_path,
        uploaded_at=datetime.now(),
        metrics={}
    )
    
    jobs_db[job_id] = job
    
    # Schedule background task
    background_tasks.add_task(
        _process_video_background,
        job_id=job_id,
        file_path=file_path
    )
    
    logger.info(f"Started processing job {job_id}")
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Video processing started. Job ID: {job_id}"
    }


@router.get("/job-status/{job_id}")
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status and progress of a processing job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_db[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        progress=job.progress,
        message=f"Job is {job.status}. Progress: {job.progress:.1f}%"
    )


@router.get("/job-metrics/{job_id}")
async def get_job_metrics(job_id: str) -> Dict:
    """Get detailed metrics for a completed job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_db[job_id]
    
    if job.status != "completed":
        return {
            "status": job.status,
            "progress": job.progress,
            "message": f"Job is still {job.status}"
        }
    
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": 100.0,
        "detections": job.detections,
        "tracks": job.tracks,
        "activities": job.activities,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "processing_time": (job.completed_at - job.started_at).total_seconds() if job.completed_at else None,
        "metrics": job.metrics
    }


@router.get("/jobs-list")
async def list_all_jobs() -> List[Dict]:
    """List all processing jobs with their status."""
    return [
        {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "file_name": job.file_name,
            "uploaded_at": job.uploaded_at
        }
        for job in jobs_db.values()
    ]


@router.post("/cancel-job/{job_id}")
async def cancel_job(job_id: str) -> Dict:
    """Cancel a processing job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_db[job_id]
    
    if job.status == "completed" or job.status == "failed":
        return {
            "success": False,
            "message": f"Cannot cancel job with status: {job.status}"
        }
    
    job.status = "cancelled"
    logger.info(f"Cancelled job {job_id}")
    
    return {
        "success": True,
        "message": f"Job {job_id} cancelled successfully"
    }


@router.get("/metrics-summary")
async def get_metrics_summary() -> Dict:
    """Get summary statistics of all jobs."""
    total_jobs = len(jobs_db)
    completed = sum(1 for j in jobs_db.values() if j.status == "completed")
    processing = sum(1 for j in jobs_db.values() if j.status == "processing")
    failed = sum(1 for j in jobs_db.values() if j.status == "failed")
    
    total_detections = sum(j.detections for j in jobs_db.values())
    total_tracks = sum(j.tracks for j in jobs_db.values())
    total_activities = sum(j.activities for j in jobs_db.values())
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed,
        "processing_jobs": processing,
        "failed_jobs": failed,
        "total_detections": total_detections,
        "total_tracks": total_tracks,
        "total_activities": total_activities,
        "timestamp": datetime.now()
    }


@router.get("/stream/{job_id}")
async def stream_job_updates(job_id: str):
    """
    Stream real-time updates for a job using Server-Sent Events (SSE).
    
    Usage in frontend:
    const eventSource = new EventSource(`/realtime/stream/${job_id}`);
    eventSource.onmessage = (event) => {
        const update = JSON.parse(event.data);
        console.log(update);
    };
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    async def event_generator():
        while True:
            job = jobs_db.get(job_id)
            if not job:
                break
            
            yield f"data: {{'job_id': '{job_id}', 'progress': {job.progress}, 'status': '{job.status}'}}\n\n"
            
            if job.status in ["completed", "failed", "cancelled"]:
                break
            
            await asyncio.sleep(1)  # Update every second
    
    return event_generator()


@router.post("/cleanup-old-jobs")
async def cleanup_old_jobs(days: int = 7) -> Dict:
    """Clean up old completed jobs (older than X days)."""
    from datetime import timedelta
    
    threshold = datetime.now() - timedelta(days=days)
    removed_count = 0
    
    for job_id in list(jobs_db.keys()):
        job = jobs_db[job_id]
        if job.status == "completed" and job.completed_at and job.completed_at < threshold:
            del jobs_db[job_id]
            removed_count += 1
    
    logger.info(f"Cleaned up {removed_count} old jobs")
    
    return {
        "removed_jobs": removed_count,
        "remaining_jobs": len(jobs_db)
    }


# ============================================================================
# BACKGROUND TASK WORKER
# ============================================================================

async def _process_video_background(job_id: str, file_path: str):
    """
    Background task that simulates video processing.
    In production, this would call the actual video processing pipeline.
    """
    if job_id not in jobs_db:
        return
    
    job = jobs_db[job_id]
    job.status = "processing"
    job.started_at = datetime.now()
    
    try:
        # Simulate processing stages
        processing_stages = [
            ("Loading video", 10),
            ("Detecting people", 30),
            ("Tracking motion", 50),
            ("Extracting poses", 70),
            ("Recognizing activities", 85),
            ("Storing results", 100),
        ]
        
        for stage_name, progress in processing_stages:
            job.progress = float(progress)
            job.metrics[stage_name] = f"✓ Completed"
            logger.info(f"Job {job_id}: {stage_name} ({progress}%)")
            await asyncio.sleep(2)  # Simulate work
        
        # Simulate results
        job.detections = 12
        job.tracks = 8
        job.activities = 24
        
        job.status = "completed"
        job.completed_at = datetime.now()
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now()
        logger.error(f"Job {job_id} failed: {str(e)}")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check() -> Dict:
    """Health check endpoint for real-time service."""
    return {
        "status": "healthy",
        "service": "realtime-api",
        "active_jobs": sum(1 for j in jobs_db.values() if j.status == "processing"),
        "timestamp": datetime.now()
    }
