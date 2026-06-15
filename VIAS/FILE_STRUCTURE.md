# VIAS Project File Structure

## 📁 Directory Tree

```
d:\VIAS-main\
├── README.md                          # Main project documentation
├── START_HERE.md                      # ← Start here first!
├── QUICKSTART_REALTIME.md             # Quick start guide
├── README_REALTIME.md                 # Real-time API reference
├── REALTIME_ANALYSIS.md               # Real-time features explained
├── IMPLEMENTATION.md                  # Architecture & implementation
├── FILE_STRUCTURE.md                  # This file
├── SYSTEM_COMPLETE.txt                # System completion status
├── COMPLETION_SUMMARY.txt             # What was implemented
├── START_REALTIME.ps1                 # PowerShell startup script
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Docker configuration
├── configs/
│   └── settings.yaml                  # System configuration
│
├── backend/
│   ├── __init__.py
│   ├── main.py                        # ⭐ FastAPI app entry point
│   ├── config.py                      # Configuration settings
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                  # Original 11 API endpoints
│   │   └── realtime_routes.py         # ⭐ NEW: 8 real-time endpoints
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db.py                      # Database connection
│   │   └── repository.py              # Data access layer
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                 # Pydantic schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detector.py                # YOLO detection
│   │   ├── tracker.py                 # Multi-object tracking
│   │   ├── activity.py                # Activity recognition
│   │   ├── behavior.py                # Behavior analysis
│   │   ├── reid.py                    # Re-identification
│   │   ├── pose.py                    # Pose estimation
│   │   ├── embedding_utils.py         # Embedding generation
│   │   ├── faiss_store.py             # Vector store
│   │   ├── model_registry.py          # Model management
│   │   ├── video_pipeline.py          # Video processing
│   │   ├── analytics.py               # Analytics computation
│   │   ├── query_engine.py            # NLQ engine
│   │   ├── dataset.py                 # Dataset management
│   │   └── tms.py                     # Motion signature
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py                 # Logging setup
│   │   └── device.py                  # GPU/CPU detection
│   │
│   └── uploads/                       # Uploaded files storage
│
├── frontend/
│   └── gradio_ui/
│       └── app.py                     # ⭐ UPDATED: Gradio UI (4 tabs)
│
├── models/
│   ├── yolov10/
│   │   └── yolov10n.pt               # YOLOv10 weights
│   └── arcface/
│       └── test_arcface.py           # Face recognition test
│
├── scripts/
│   ├── train_detection.py             # Train detection model
│   ├── train_reid.py                  # Train re-id model
│   ├── train_activity.py              # Train activity model
│   ├── train_tms.py                   # Train TMS model
│   ├── evaluation.py                  # Model evaluation
│   ├── benchmark.py                   # Performance benchmarking
│   └── sample_data_pipeline.py        # Sample data generation
│
├── datasets/
│   ├── README.md                      # Dataset documentation
│   ├── detection.yaml                 # Detection dataset config
│   └── [data files...]
│
├── faiss_store/
│   ├── face_embeddings.jsonl         # Face embeddings index
│   ├── body_embeddings.jsonl         # Body embeddings index
│   └── tms_embeddings.jsonl          # Motion signature embeddings
│
├── sqlite_db/
│   └── vias.db                        # SQLite database
│
├── tests/
│   ├── test_health.py                 # Health check tests
│   ├── test_reid.py                   # Re-id tests
│   ├── test_tms.py                    # TMS tests
│   └── test_tracker.py                # Tracking tests
│
├── docker/
│   └── Dockerfile                     # Docker container config
│
└── VIAS/                              # Legacy folder (for compatibility)
    └── [mirrors main structure]
```

## 📝 Key Files by Purpose

### Entry Points
- **Backend**: `backend/main.py` - Run with `uvicorn backend.main:app`
- **Frontend**: `frontend/gradio_ui/app.py` - Run with `python app.py`
- **Startup**: `START_REALTIME.ps1` - Run with `.\START_REALTIME.ps1`

### Real-Time Features (NEW!)
- **Routes**: `backend/api/realtime_routes.py` - 8 new endpoints
- **UI**: `frontend/gradio_ui/app.py` - 4 interactive tabs
- **Main**: `backend/main.py` - Routes imported here

### API Endpoints
- **Original** (11): `backend/api/routes.py`
- **Real-Time** (8): `backend/api/realtime_routes.py`
- **Total**: 19 endpoints

### Services (Model Inference)
| Service | File | Purpose |
|---------|------|---------|
| Detector | `detector.py` | YOLOv10 person detection |
| Tracker | `tracker.py` | Multi-object tracking |
| Pose | `pose.py` | Pose estimation |
| Activity | `activity.py` | Activity recognition |
| Re-ID | `reid.py` | Body shape matching |
| Embeddings | `embedding_utils.py` | Generate face/body embeddings |
| FAISS | `faiss_store.py` | Vector similarity search |
| TMS | `tms.py` | Motion signature analysis |
| Query | `query_engine.py` | Natural language processing |
| Analytics | `analytics.py` | Compute metrics |

### Configuration Files
| File | Purpose |
|------|---------|
| `configs/settings.yaml` | System settings |
| `datasets/detection.yaml` | Detection dataset config |
| `requirements.txt` | Python dependencies |
| `docker-compose.yml` | Docker services |

### Documentation Files (NEW!)
| File | Content |
|------|---------|
| `START_HERE.md` | Getting started guide |
| `QUICKSTART_REALTIME.md` | Quick start in 3 steps |
| `README_REALTIME.md` | Complete API reference |
| `REALTIME_ANALYSIS.md` | Real-time features explained |
| `IMPLEMENTATION.md` | Architecture deep dive |
| `FILE_STRUCTURE.md` | This file |
| `SYSTEM_COMPLETE.txt` | Completion status |
| `COMPLETION_SUMMARY.txt` | What was implemented |

### Model Weights
| File | Size | Purpose |
|------|------|---------|
| `models/yolov10/yolov10n.pt` | ~50MB | Person detection |
| `models/arcface/...` | Varies | Face recognition |

### Database & Storage
| Location | Purpose |
|----------|---------|
| `sqlite_db/vias.db` | SQLite database |
| `faiss_store/` | Vector embeddings |
| `backend/uploads/` | Uploaded files |

## 🔄 Data Flow

```
User Upload (Frontend)
    ↓
Gradio UI (app.py)
    ↓
FastAPI Backend (main.py)
    ↓
Routes (routes.py + realtime_routes.py)
    ↓
Services (detector, tracker, etc.)
    ↓
Models (YOLOv10, ArcFace, etc.)
    ↓
Database (SQLite)
    ↓
FAISS Store (embeddings)
    ↓
Results Display (Frontend)
```

## 🚀 Startup Order

1. **Terminal 1**: Start Backend
   ```powershell
   cd d:\VIAS-main
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```

2. **Terminal 2**: Start Frontend
   ```powershell
   cd d:\VIAS-main
   python VIAS/frontend/gradio_ui/app.py
   ```

3. **Browser**: Open UI
   - Frontend: http://127.0.0.1:7860
   - API Docs: http://127.0.0.1:8000/docs

## 📊 File Statistics

- **Python Files**: 25+
- **Documentation Files**: 8 (NEW)
- **Model Weights**: 2+
- **Total Code Lines**: ~3000+
- **Total Documentation Lines**: ~2000+
- **Real-Time Endpoints**: 8 (NEW)

## 🔗 Related Documentation

- [START_HERE.md](START_HERE.md) - Overall getting started
- [QUICKSTART_REALTIME.md](QUICKSTART_REALTIME.md) - Quick reference
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical details
- [README_REALTIME.md](README_REALTIME.md) - API reference

## ✅ Completion Checklist

- ✅ Real-time API endpoints (8)
- ✅ Gradio frontend (4 tabs)
- ✅ Backend main.py updated
- ✅ Documentation (8 files)
- ✅ Error handling
- ✅ Async task processing
- ✅ Live metrics tracking
- ✅ SSE streaming support
