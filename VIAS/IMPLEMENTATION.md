# VIAS Implementation Details

## 📐 System Architecture

### Backend Stack
- **Framework**: FastAPI (async, high-performance)
- **Database**: SQLite (lightweight, portable)
- **Task Queue**: Background tasks (asyncio)
- **Vector Store**: FAISS (similarity search)
- **Models**:
  - Detection: YOLOv10
  - Face Recognition: ArcFace
  - Body Re-ID: OSNet
  - Pose: MediaPipe
  - Activities: Custom CNN

### Frontend Stack
- **Framework**: Gradio (web UI)
- **HTTP Client**: requests
- **Streaming**: Server-Sent Events (SSE)
- **Styling**: CSS custom themes

## 🔧 Component Details

### Backend API Routes

#### Original Routes (`backend/api/routes.py`)
```python
11 endpoints:
- POST /upload-video
- POST /upload-reference-image
- GET /search-person
- POST /query
- POST /behavior-search
- GET /activities
- GET /analytics
- GET /models/status
- GET /datasets
- GET /health
- GET /events
```

#### Real-Time Routes (`backend/api/realtime_routes.py`)
```python
8 new endpoints:
- POST /realtime/process-video
- GET /realtime/job-status/{job_id}
- GET /realtime/job-metrics/{job_id}
- GET /realtime/jobs-list
- POST /realtime/cancel-job/{job_id}
- GET /realtime/metrics-summary
- GET /realtime/stream/{job_id}
- POST /realtime/cleanup-old-jobs
- GET /realtime/health
```

### Frontend UI Components

#### Tab 1: Video Processing
```
VideoInput
  ├─ video_input: gr.Video()
  ├─ upload_button: gr.Button()
  └─ video_output: gr.Markdown()
```

#### Tab 2: Person Analyzing
```
PersonAnalysis
  ├─ Register Section
  │  ├─ ref_image: gr.Image()
  │  ├─ ref_person_id: gr.Textbox()
  │  └─ register_button: gr.Button()
  │
  └─ Search Section
     ├─ search_person_id: gr.Textbox()
     ├─ search_button: gr.Button()
     └─ search_output: gr.Markdown()
```

#### Tab 3: Query Engine
```
QueryEngine
  ├─ Query Section
  │  ├─ query_input: gr.Textbox()
  │  ├─ query_button: gr.Button()
  │  └─ query_output: gr.Markdown()
  │
  └─ Behavior Search
     ├─ behavior_input: gr.Textbox()
     ├─ behavior_button: gr.Button()
     └─ behavior_output: gr.Markdown()
```

#### Tab 4: Analytics & Status
```
Analytics
  ├─ analytics_button: gr.Button()
  ├─ analytics_output: gr.Markdown()
  │
  ├─ activities_button: gr.Button()
  ├─ activities_output: gr.Markdown()
  │
  ├─ status_button: gr.Button()
  └─ status_output: gr.Markdown()
```

### Data Models

#### ProcessingJob
```python
{
  "job_id": str,
  "status": str,
  "file_name": str,
  "uploaded_at": datetime,
  "started_at": datetime,
  "completed_at": datetime,
  "progress": float,
  "detections": int,
  "tracks": int,
  "activities": int,
  "error": Optional[str],
  "metrics": Dict
}
```

## 🔄 Processing Flow

### Video Upload Flow
```
1. User uploads video in Gradio
2. POST /realtime/process-video
3. Backend creates job record
4. Returns job_id immediately
5. Background task starts
6. Frontend polls /realtime/job-status/{job_id}
7. Progress updates in real-time
8. POST /realtime/stream/{job_id} for SSE
9. Frontend receives updates every 1 second
10. Job completion detected
11. GET /realtime/job-metrics/{job_id}
12. Display final results
```

### Person Search Flow
```
1. User registers reference image
2. POST /upload-reference-image
3. ArcFace generates face embedding
4. OSNet generates body embedding
5. Embeddings stored in FAISS
6. User enters person_id
7. POST /search-person
8. FAISS similarity search
9. Multi-tier matching (face + body + motion)
10. Results with confidence scores
11. Display matches with rankings
```

### Natural Language Query Flow
```
1. User types question in Gradio
2. POST /query {"query": "..."}
3. NLQ engine parses intent
4. Generates SQL query
5. Executes on SQLite database
6. Returns results with SQL citation
7. Format and display results
```

## 💾 Database Schema

### People Table
```sql
CREATE TABLE people (
  id INTEGER PRIMARY KEY,
  person_id TEXT UNIQUE,
  face_embedding BLOB,
  body_embedding BLOB,
  created_at TIMESTAMP
);
```

### Events Table
```sql
CREATE TABLE events (
  id INTEGER PRIMARY KEY,
  video_id TEXT,
  timestamp TIMESTAMP,
  event_type TEXT,
  person_id TEXT,
  confidence FLOAT,
  metadata JSON
);
```

### Activities Table
```sql
CREATE TABLE activities (
  id INTEGER PRIMARY KEY,
  track_id TEXT,
  activity_type TEXT,
  timestamp TIMESTAMP,
  confidence FLOAT,
  duration FLOAT
);
```

## 🔌 Error Handling Strategy

### Connection Errors
```python
if isinstance(error, requests.exceptions.ConnectionError):
    return "❌ Cannot reach backend at {API_BASE}"
```

### Timeout Errors
```python
if isinstance(error, requests.exceptions.Timeout):
    return "⏱️ Operation took too long. Try with smaller file."
```

### HTTP Errors
```python
if isinstance(error, requests.exceptions.HTTPError):
    return "❌ Backend Error: {error}"
```

## 🚀 Performance Optimizations

1. **Async Processing**: Background tasks don't block UI
2. **Batch Processing**: Process multiple frames in parallel
3. **GPU Acceleration**: CUDA for model inference
4. **Caching**: Store model predictions
5. **Lazy Loading**: Load models on-demand
6. **Streaming Results**: Don't wait for all results
7. **Connection Pooling**: Reuse HTTP connections

## 📊 Monitoring

### Health Checks
```
GET /realtime/health
{
  "status": "healthy",
  "service": "realtime-api",
  "active_jobs": 2,
  "timestamp": "2026-06-15T08:30:00"
}
```

### Metrics Summary
```
GET /realtime/metrics-summary
{
  "total_jobs": 45,
  "completed_jobs": 42,
  "processing_jobs": 2,
  "failed_jobs": 1,
  "total_detections": 1245,
  "timestamp": "2026-06-15T08:30:00"
}
```

## 🔐 Security Considerations

1. **CORS Enabled**: Allow frontend to communicate
2. **Input Validation**: Check all inputs
3. **Error Messages**: Don't leak sensitive info
4. **File Upload**: Validate file types
5. **Resource Limits**: Set timeouts for operations

## 📈 Scalability Path

### Current (Single Machine)
- In-memory job store
- SQLite database
- Single GPU

### Future (Distributed)
- Redis for job queue
- PostgreSQL for data
- Multiple GPU workers
- Load balancer
- Message broker (RabbitMQ)

## 🛠️ Troubleshooting Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| "No such module: gradio" | Gradio not installed | `pip install gradio` |
| "Connection Error" | Backend not running | Start backend: `uvicorn backend.main:app` |
| "Timeout Error" | Video too large | Use smaller video (< 100MB) |
| "Job not found" | Job ID incorrect | Check /realtime/jobs-list |
| "CUDA out of memory" | GPU memory full | Reduce batch size or use CPU |

## 📚 References

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Gradio Docs](https://gradio.app/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [YOLOv10 Docs](https://docs.ultralytics.com/)
