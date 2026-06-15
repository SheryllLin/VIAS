# Quick Start - VIAS Real-Time System

## 🚀 Get Started in 3 Steps

### Step 1: Start Backend
```powershell
cd d:\VIAS-main
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Step 2: Start Frontend (in new terminal)
```powershell
cd d:\VIAS-main
python VIAS/frontend/gradio_ui/app.py
```

Expected output:
```
Running on local URL:  http://127.0.0.1:7860
```

### Step 3: Open Browser
Go to: **http://127.0.0.1:7860**

## 📋 Next Steps

1. **Upload a Video**
   - Go to Tab 1: "📹 Video Processing"
   - Select a video file
   - Click "🚀 Process Video"
   - Watch real-time progress

2. **Register a Person**
   - Go to Tab 2: "👤 Person Analyzing"
   - Upload a reference image
   - Enter a Person ID
   - Click "📝 Register Person"

3. **Search for Person**
   - In Tab 2, enter the Person ID you registered
   - Click "🔎 Search"
   - See matching results with confidence scores

4. **Ask Questions**
   - Go to Tab 3: "❓ Query Engine"
   - Type: "How many people were detected?"
   - Click "💬 Query"
   - Get results with SQL citations

5. **View Analytics**
   - Go to Tab 4: "📊 Analytics & Status"
   - Click "📈 Refresh Analytics"
   - See system metrics and health

## ✅ Everything Working?

- ✅ Backend API responding
- ✅ Frontend UI loading
- ✅ All 4 tabs visible
- ✅ Buttons clickable
- ✅ Real-time updates flowing

## 🔧 Troubleshooting

### Frontend shows "Connection Error"
- Backend not running?
- Check: Is port 8000 in use?
- Kill and restart backend

### Video upload timeout
- File too large?
- Try a smaller video (< 100MB)
- Check system RAM

### No results in search
- Haven't registered any images yet
- Register reference images first

## 🎯 Key Features

| Feature | Location | What It Does |
|---------|----------|-------------|
| Video Processing | Tab 1 | Detects people, tracks motion, recognizes activities |
| Person Search | Tab 2 | Finds specific people using facial recognition |
| Natural Queries | Tab 3 | Ask questions in English, get SQL-backed answers |
| Analytics | Tab 4 | See system metrics and health status |

## 📚 More Documentation

- [README_REALTIME.md](README_REALTIME.md) - Full API reference
- [REALTIME_ANALYSIS.md](REALTIME_ANALYSIS.md) - Real-time features explained
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Architecture deep dive
- [START_HERE.md](START_HERE.md) - Overall getting started