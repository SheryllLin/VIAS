# VIAS Real-Time Analysis System

## 🏗️ Overview

The VIAS Real-Time Analysis System provides live video processing capabilities with:
- **Non-blocking processing** - Returns immediately with job ID
- **Live progress tracking** - Monitor job status in real-time
- **Streaming updates** - Server-Sent Events for live metrics
- **Comprehensive metrics** - Detection counts, tracking accuracy, processing speed

## 🏗️ Real-Time Architecture

```
Frontend Request
    ↓
[FastAPI Endpoint]
    ↓
[Create Job Record] → Return Job ID immediately ✅
    ↓
[Background Task]
    ├── Stage 1: Loading video (10%)
    ├── Stage 2: Detecting people (30%)
    ├── Stage 3: Tracking motion (50%)
    ├── Stage 4: Extracting poses (70%)
    ├── Stage 5: Recognizing activities (85%)
    └── Stage 6: Storing results (100%)
```

## 🔄 Processing Pipeline

### Stage 1: Loading Video
- Read video file
- Extract metadata (FPS, resolution, duration)
- Initialize processing context

### Stage 2: Detecting People
- YOLOv10 person detection
- Bounding box extraction
- Confidence filtering

### Stage 3: Tracking Motion
- Multi-object tracking (MOT)
- Motion history
- Track ID assignment

### Stage 4: Extracting Poses
- Pose estimation
- Skeleton extraction
- Activity features

### Stage 5: Recognizing Activities
- Activity classification
- Behavior recognition
- Action detection

### Stage 6: Storing Results
- Save to database
- Index for searching
- Generate analytics

## 📊 Real-Time Metrics

### Per-Job Metrics
```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "progress": 100.0,
  "detections": 12,           // Total people detected
  "tracks": 8,                 // Unique individuals tracked
  "activities": 24,            // Total activity events
  "processing_time": 45.32,    // Seconds
  "frames_per_second": 28.5
}
```

### System-Wide Metrics
```json
{
  "total_jobs": 45,
  "completed_jobs": 42,
  "processing_jobs": 2,
  "failed_jobs": 1,
  "total_detections": 1245,
  "total_tracks": 342,
  "total_activities": 5678,
  "avg_processing_time": 42.5
}
```

## 🎬 Live Streaming with SSE

### JavaScript Frontend
```javascript
// Create event source
const eventSource = new EventSource('/realtime/stream/a1b2c3d4');

// Listen for updates
eventSource.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log(`Progress: ${update.progress}%`);
  
  // Update UI
  document.getElementById('progress').textContent = update.progress;
  document.getElementById('status').textContent = update.status;
  
  // Close when done
  if (update.status === 'completed') {
    eventSource.close();
  }
};

// Error handling
eventSource.onerror = () => {
  console.error('Connection lost');
  eventSource.close();
};
```

## 🔍 Multi-Tier Person Matching

### Tier 1: Face Recognition (ArcFace)
- Face embedding extraction
- Vector similarity (cosine)
- Confidence: 0.8-1.0
- Best for: Close-up faces

### Tier 2: Body Shape (OSNet)
- Body feature extraction
- Appearance matching
- Confidence: 0.6-0.9
- Best for: Full body visibility

### Tier 3: Motion Signatures (TMS)
- Walking pattern analysis
- Gait recognition
- Confidence: 0.5-0.8
- Best for: Motion patterns

### Combined Score
```
Final Confidence = (Face_Conf * 0.5) + (Body_Conf * 0.3) + (Motion_Conf * 0.2)
```

## 📈 Real-Time Analytics

### Activity Distribution
```
Standing:    45%  🟢
Walking:     30%  🟡
Waving:      15%  🔵
Running:     10%  🔴
```

## ⚠️ Error Handling

### Job Failure Scenarios
```
Status: "failed"
Error: "Video file corrupted"
Action: Retry or upload different file
```

## 🔒 Data Privacy

- No frames stored after processing
- Embeddings stored in FAISS (vector-only)
- Original images not persisted
- Metadata-only storage

## 📞 API Response Times

| Operation | Time | Notes |
|-----------|------|-------|
| Job creation | < 100ms | Immediate |
| Status check | < 50ms | Lightweight query |
| Metrics retrieval | < 200ms | Aggregation |
| Person search | 100-500ms | FAISS lookup |
| Query execution | 500-2000ms | Database query |
| SSE stream start | < 100ms | Event source |

## 🎓 Learning Path

1. **Basics**: Upload and process a video
2. **Intermediate**: Register and search for people
3. **Advanced**: Write natural language queries
4. **Expert**: Analyze real-time metrics and optimize
