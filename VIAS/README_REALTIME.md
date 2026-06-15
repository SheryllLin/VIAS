# VIAS Real-Time API Documentation

## Overview

VIAS (Visual Intelligence & Analysis System) now includes **real-time video processing** capabilities with:
- ✅ Asynchronous job tracking
- ✅ Live progress streaming
- ✅ Server-Sent Events (SSE) support
- ✅ Background task management

## Quick Start

### Start the Backend
```bash
cd d:\VIAS-main
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Start the Frontend
```bash
cd d:\VIAS-main
python VIAS/frontend/gradio_ui/app.py
```

### Access the System
- **Frontend UI**: http://127.0.0.1:7860
- **API Docs**: http://127.0.0.1:8000/docs
- **Backend**: http://127.0.0.1:8000

## Real-Time Endpoints

### 1. Start Video Processing
```http
POST /realtime/process-video
Content-Type: application/json

{
  "file_path": "/path/to/video.mp4"
}
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "queued",
  "message": "Video processing started. Job ID: a1b2c3d4"
}
```

### 2. Check Job Status
```http
GET /realtime/job-status/{job_id}
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "processing",
  "progress": 45.5,
  "message": "Job is processing. Progress: 45.5%"
}
```

### 3. Get Job Metrics
```http
GET /realtime/job-metrics/{job_id}
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "progress": 100.0,
  "detections": 12,
  "tracks": 8,
  "activities": 24,
  "processing_time": 45.32,
  "metrics": { "Loading video": "✓ Completed", ... }
}
```

### 4. List All Jobs
```http
GET /realtime/jobs-list
```

### 5. Cancel a Job
```http
POST /realtime/cancel-job/{job_id}
```

### 6. Get Metrics Summary
```http
GET /realtime/metrics-summary
```

### 7. Stream Live Updates (SSE)
```javascript
const eventSource = new EventSource('/realtime/stream/a1b2c3d4');
eventSource.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log(`Progress: ${update.progress}%`);
};
```

## Frontend Features

### Tab 1: 📹 Video Processing
- Upload video files
- Real-time progress tracking
- View detection results
- See processing metrics

### Tab 2: 👤 Person Analyzing
- Register reference images
- Search for person matches
- Multi-tier matching with confidence scores
- Uses face embeddings (ArcFace), body shape (OSNet), motion signatures (TMS)

### Tab 3: ❓ Query Engine
- Natural language queries
- SQL citations for results
- Behavior semantic search
- Confidence scoring

### Tab 4: 📊 Analytics & Status
- Real-time analytics dashboard
- Activity timeline
- System health monitoring
- Model and dataset status

## Architecture

```
Frontend (Gradio)
    ↓
Backend API (FastAPI)
    ├── /api/routes.py (11 original endpoints)
    ├── /api/realtime_routes.py (8 real-time endpoints)
    └── Services
        ├── Video Pipeline
        ├── Detector (YOLO)
        ├── Tracker
        ├── Activity Recognition
        ├── FAISS Store (embeddings)
        ├── Re-ID (body matching)
        ├── Face Recognition (ArcFace)
        └── TMS (motion signatures)
```

## Configuration

Set environment variables:
```powershell
$env:VIAS_API_BASE = "http://127.0.0.1:8000"
```

## Error Handling

All endpoints include comprehensive error handling:
- Connection errors
- Timeout handling
- Job not found (404)
- Backend failures (500)

## Performance Tips

1. **Video Processing**: 
   - Recommended max file size: 500MB
   - Timeout: 20 minutes
   
2. **Person Search**:
   - Register clear reference images
   - Top-K results: 10 (configurable)
   
3. **Queries**:
   - Timeout: 5 minutes
   - Use specific queries for better results

## Status Codes

- ✅ **completed**: Processing finished successfully
- 🔄 **processing**: Currently processing
- ⏳ **pending/queued**: Waiting to start
- ❌ **failed**: Processing failed with error
- 🛑 **cancelled**: Job was cancelled by user

## Troubleshooting

### Backend Connection Error
```
Error: Cannot reach backend at http://127.0.0.1:8000
Solution: Make sure backend is running with uvicorn
```

### Timeout on Video Upload
```
Solution: Check system resources and try a smaller video
```

### No Matches Found
```
Solution: Register reference images first before searching
```

## Next Steps

1. Read [START_HERE.md](START_HERE.md) for getting started
2. Check [IMPLEMENTATION.md](IMPLEMENTATION.md) for architecture details
3. See [REALTIME_ANALYSIS.md](REALTIME_ANALYSIS.md) for real-time features
