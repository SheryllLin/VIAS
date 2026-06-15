# 🎬 START HERE - VIAS Getting Started

## Welcome to VIAS!

**Visual Intelligence & Analysis System** - A complete surveillance video analysis platform with real-time processing.

## ⚡ Quick Start (3 minutes)

### 1. Start Backend (Terminal 1)
```powershell
cd d:\VIAS-main
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
✅ Wait for: `Application startup complete`

### 2. Start Frontend (Terminal 2)
```powershell
cd d:\VIAS-main
python VIAS/frontend/gradio_ui/app.py
```
✅ Wait for: `Running on local URL:  http://127.0.0.1:7860`

### 3. Open Browser
Go to: **http://127.0.0.1:7860**

You should see:
- 🎥 VIAS banner
- 4 interactive tabs
- Ready to process!

## 🎯 What Can You Do?

### Tab 1: 📹 Video Processing
**Upload and analyze surveillance videos**
1. Click "Select Video File"
2. Choose a video (MP4, AVI, etc.)
3. Click "🚀 Process Video"
4. Watch real-time progress (0-100%)
5. See detected people, tracks, activities

**Result**: Video analysis with metrics

### Tab 2: 👤 Person Analyzing
**Find specific people across videos**

**Step 1 - Register a Person:**
1. Upload a reference photo of a person
2. Enter a Person ID (e.g., "john-doe")
3. Click "📝 Register Person"
4. System creates facial embedding

**Step 2 - Search:**
1. Enter the Person ID
2. Click "🔎 Search"
3. System finds all matches in video database
4. Shows confidence scores (🟢 high, 🟡 medium, 🔴 low)

**Result**: Matches with confidence scores

### Tab 3: ❓ Query Engine
**Ask natural language questions about videos**

**Natural Language Queries:**
- "How many people were detected?"
- "Show me all standing activities"
- "What activities are most common?"

**Semantic Behavior Search:**
- "Find people who waved"
- "Show running activities"
- "Find standing then walking sequences"

**Result**: Query results with SQL citations

### Tab 4: 📊 Analytics & Status
**Monitor system health and metrics**

**Available Reports:**
- 📈 Analytics Dashboard (total people, events, tracks)
- 📅 Activity Timeline (chronological events)
- 🔧 System Status (backend health, model status)

**Buttons:**
- "📈 Refresh Analytics" - Update dashboard
- "📅 Activity Timeline" - Show activity history
- "🔧 Check System Status" - Verify all systems

## 🛠️ System Requirements

### Hardware
- **CPU**: Intel Core i5 or better
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA with CUDA (optional but recommended)
- **Storage**: 10GB free space

### Software
- Python 3.9+
- FFmpeg (for video processing)
- CUDA 11+ (optional, for GPU acceleration)

### Installation
```powershell
# Install Python dependencies
pip install -r requirements.txt

# Install FFmpeg (Windows)
choco install ffmpeg
# Or download from https://ffmpeg.org/download.html
```

## 📋 Check Installation

Run health check:
```powershell
curl http://127.0.0.1:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "service": "realtime-api",
  "active_jobs": 0,
  "timestamp": "2026-06-15T..."
}
```

## 🔍 Example Workflows

### Workflow 1: Identify Missing Person
1. Go to **Tab 2: Person Analyzing**
2. Upload photo of missing person
3. Enter ID: "missing-person-001"
4. Click "Register Person"
5. System searches all videos
6. Shows all matches with timestamps

### Workflow 2: Activity Report
1. Go to **Tab 3: Query Engine**
2. Ask: "What happened at 2 PM?"
3. System shows all events from that time
4. See SQL query used in response

### Workflow 3: Suspicious Behavior
1. Go to **Tab 3: Query Engine**
2. Search: "Find unusual activities"
3. System highlights anomalies
4. Review with timestamps

## ⚙️ Configuration

### API Base URL
Default: `http://127.0.0.1:8000`

Change in frontend:
```python
API_BASE = os.getenv("VIAS_API_BASE", "http://127.0.0.1:8000")
```

### Video Processing Timeout
Default: 20 minutes (1200 seconds)

Edit in `app.py`:
```python
API_TIMEOUT_VIDEO = 1200  # seconds
```

## 🐛 Troubleshooting

### "Connection Error - Cannot reach backend"
```
❌ Solution: Backend not running
✅ Fix: Start backend in Terminal 1
```

### "Timeout - Operation took too long"
```
❌ Solution: Video file too large
✅ Fix: Use smaller video (< 500MB)
```

### "Job not found"
```
❌ Solution: Job ID invalid or expired
✅ Fix: Check recent jobs in jobs-list
```

### "No matches found"
```
❌ Solution: No reference images registered
✅ Fix: Register person images first in Tab 2
```

### Port Already in Use
```
# Backend (8000)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Frontend (7860)
netstat -ano | findstr :7860
taskkill /PID <PID> /F
```

## 📚 Next Steps

1. **Quick Start**: Read [QUICKSTART_REALTIME.md](QUICKSTART_REALTIME.md)
2. **API Reference**: Check [README_REALTIME.md](README_REALTIME.md)
3. **Real-Time Features**: Learn [REALTIME_ANALYSIS.md](REALTIME_ANALYSIS.md)
4. **Architecture**: Explore [IMPLEMENTATION.md](IMPLEMENTATION.md)
5. **File Organization**: See [FILE_STRUCTURE.md](FILE_STRUCTURE.md)

## 📞 API Testing

### Using curl
```bash
# Check health
curl http://127.0.0.1:8000/health

# List all jobs
curl http://127.0.0.1:8000/realtime/jobs-list

# Get metrics
curl http://127.0.0.1:8000/realtime/metrics-summary
```

### Using Python
```python
import requests

# Check status
response = requests.get('http://127.0.0.1:8000/realtime/metrics-summary')
print(response.json())
```

## 🎓 Learning Resources

- **Gradio**: https://gradio.app/
- **FastAPI**: https://fastapi.tiangolo.com/
- **YOLO**: https://docs.ultralytics.com/
- **FAISS**: https://github.com/facebookresearch/faiss

## 🚀 Optimization Tips

1. **Faster Processing**: Use GPU (CUDA)
2. **Smaller Videos**: Resample to 30 FPS
3. **Batch Processing**: Process multiple videos
4. **Caching**: Enable model caching
5. **Cleanup**: Run `/realtime/cleanup-old-jobs` weekly

## ✅ Verification Checklist

- ✅ Backend running (port 8000)
- ✅ Frontend running (port 7860)
- ✅ All 4 tabs visible
- ✅ Can upload video
- ✅ Can register person
- ✅ Can query database
- ✅ Can view analytics
- ✅ System health: OK

## 🎉 You're Ready!

Everything is set up and running. Start exploring VIAS:

1. Upload a test video
2. Register a reference person
3. Try a natural language query
4. Check system analytics

**Happy analyzing!** 🎥🔍

---

**Questions?** Check the [README.md](README.md) for more info.

**Need help?** See [IMPLEMENTATION.md](IMPLEMENTATION.md) for architecture details.
