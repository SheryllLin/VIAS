# VIAS AI SYSTEMS AUDIT REPORT
## Comprehensive AI Component Analysis
**Date:** 2026-06-15  
**Auditor:** AI Systems Auditor  
**Status:** CRITICAL ISSUES IDENTIFIED ⚠️

---

# TABLE OF CONTENTS
1. [Executive Summary](#executive-summary)
2. [AI Model Inventory](#ai-model-inventory)
3. [Detailed Component Analysis](#detailed-component-analysis)
4. [Data Flow Analysis](#data-flow-analysis)
5. [Runtime Verification](#runtime-verification)
6. [Confidence Analysis](#confidence-analysis)
7. [Database Validation](#database-validation)
8. [Query Engine Validation](#query-engine-validation)
9. [Critical Issues Summary](#critical-issues-summary)
10. [Final Assessment](#final-assessment)

---

# EXECUTIVE SUMMARY

## System Overview
VIAS claims to be a "Visual Intelligence & Analysis System" with advanced AI capabilities including:
- Real-time person detection
- Multi-modal person re-identification (face + body + motion)
- Activity recognition
- Behavior discovery
- Natural Language Query Engine

## Audit Findings: 🔴 CRITICAL

**Out of 6 claimed AI components, only 2 are genuinely functional deep learning models. The remaining 4 are either:**
- Placeholder implementations using deterministic hashing
- Rule-based systems masquerading as AI
- Dependencies that may not be installed

### Key Findings:
- ✅ **YOLOv10**: Working (person detection)
- ✅ **ByteTrack**: Working (multi-object tracking)
- ⚠️ **MediaPipe Pose**: Working (when available)
- ⚠️ **ArcFace**: Optional, fallback to deterministic hash
- ⚠️ **OSNet**: Optional, fallback to deterministic hash
- ❌ **ST-GCN++**: NOT implemented - uses rule-based heuristics only
- ❌ **TMS (Temporal Motion Signature)**: Hand-crafted features, not a neural network
- ⚠️ **Ollama LLM**: Optional, has SQL fallback

---

# AI MODEL INVENTORY

## Model #1: YOLOv10 (Person Detection)

### 1. Identification
- **Name:** YOLOv10 Nano (yolov10n)
- **Files:** `backend/services/detector.py`
- **Framework:** Ultralytics PyTorch
- **Type:** Pretrained model
- **Weights Location:** `models/yolov10/yolov10n.pt`
- **Status:** ✅ EXISTS

### 2. Purpose
| Aspect | Details |
|--------|---------|
| **Why Used** | Real-time person detection in video frames |
| **Problem Solved** | Locating humans in surveillance footage |
| **Input** | Video frame (NumPy array, BGR format) |
| **Output** | List of bounding boxes with confidence scores |
| **Dependency** | Detection confidence threshold (config: 0.25) |
| **Used By** | ByteTrackService, VideoAnalyticsPipeline |

### 3. Implementation Details
```python
# From detector.py (line 21-25)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class YOLOv10Detector:
    def __init__(self):
        self.model = YOLO(str(self.weights_path)) if YOLO is not None else None
        # Fallback: cv2.HOG if YOLO fails
```

**Fallback Mechanism:** If model fails, falls back to HOG (Histogram of Oriented Gradients - traditional CV algorithm from OpenCV)

### 4. Runtime Verification

**Status:** ✅ Working (Primary) / ⚠️ Degraded (Fallback)

```python
def detect(self, frame, frame_id: int) -> tuple[list[DetectionRecord], float]:
    if self.model is not None:
        results = self.model.predict(frame, classes=[0], conf=self.conf_threshold, ...)
        # Returns detections
    else:
        # Fallback to HOG
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8, 8))
```

**Verification:**
- ✅ Model weights file exists at `models/yolov10/yolov10n.pt`
- ✅ FPS metric is computed: `1.0 / elapsed_time`
- ✅ Confidence scores are extracted from model output
- ✅ Device selection (CPU/GPU) is configured
- ⚠️ No validation against ground truth labels
- ⚠️ No per-class accuracy metrics

### 5. Output Validation

**Mechanism:** Returns DetectionRecord with:
- `person_bbox`: [x1, y1, x2, y2] coordinates
- `confidence`: Float between 0-1
- `detector`: "yolov10" or "hog"

**Verification Method (Manual):**
- Count detected persons visually in frame
- Compare with model output count
- Check if bounding boxes align with actual persons

### 6. Confidence Analysis

| Metric | Value | Assessment |
|--------|-------|-----------|
| Confidence Source | YOLOv10 model output | ✅ Real |
| Threshold (Config) | 0.25 | ✅ Meaningful |
| Range | 0.0 - 1.0 | ✅ Valid |
| Low-confidence Filtering | Yes (threshold applied) | ✅ Applied |
| Confidence Meaningful | Yes | ✅ Yes |

### 7. Performance Metrics

| Metric | Value | Source |
|--------|-------|--------|
| FPS Computed | Yes | `1.0 / elapsed` |
| Accuracy | ❌ Not measured | No validation set |
| Precision | ❌ Not measured | No ground truth |
| Recall | ❌ Not measured | No ground truth |
| F1-Score | ❌ Not measured | N/A |

**Status:** ❌ No evaluation mechanism implemented

---

## Model #2: MediaPipe Pose

### 1. Identification
- **Name:** MediaPipe Pose (Model Complexity 1)
- **Files:** `backend/services/pose.py`
- **Framework:** MediaPipe (Google)
- **Type:** Pretrained model (downloadable from MediaPipe)
- **Weights Location:** Downloaded automatically
- **Status:** ⚠️ Optional dependency

### 2. Purpose

| Aspect | Details |
|--------|---------|
| **Why Used** | Extract skeleton keypoints (17 landmarks) from person |
| **Problem Solved** | Obtain joint positions for activity recognition |
| **Input** | Cropped person frame (RGB) |
| **Output** | 17 keypoints (x, y, z) + confidence |
| **Used By** | STGCNPlusPlusService (activity classification) |

### 3. Implementation Details
```python
# From pose.py
try:
    import mediapipe as mp
except ImportError:
    mp = None

class MediaPipePoseService:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    
    def extract(self, frame, track_id: int, frame_id: int) -> PoseRecord | None:
        if self.pose is None:
            return None
        result = self.pose.process(frame[:, :, ::-1])  # BGR to RGB
        # Returns 33 keypoints (MediaPipe uses 33, not 17)
```

### 4. Runtime Verification

**Status:** ⚠️ Conditionally Working

- ✅ Returns keypoints if available
- ⚠️ Returns None if MediaPipe not installed
- ⚠️ No error handling for inference failures
- ⚠️ Silent failure if keypoints invalid

**Issue:** Calling code doesn't validate keypoint quality
```python
# From video_pipeline.py, line 75
pose_record = self.pose.extract(frame, track_id, frame_id)
# No check if pose_record is None or invalid
if pose_record:
    activity = self.activity.classify(pose_record, timestamp)
```

### 5. Confidence Analysis

| Aspect | Status |
|--------|--------|
| Confidence Score Generated | ❌ No |
| Keypoint Confidence Available | ✅ Yes (in MediaPipe result) |
| Confidence Used | ❌ No (ignored) |
| Meaning | N/A (not extracted) |

**Issue:** MediaPipe provides per-keypoint confidence but VIAS ignores it!

---

## Model #3: ByteTrack (Multi-Object Tracking)

### 1. Identification
- **Name:** ByteTrack (YOLOv10 native tracker)
- **Files:** `backend/services/tracker.py`
- **Framework:** YOLOv10 Ultralytics
- **Type:** Pretrained tracker (integrated with YOLO)
- **Status:** ✅ Working

### 2. Purpose

| Aspect | Details |
|--------|---------|
| **Why Used** | Maintain consistent track IDs across frames |
| **Problem Solved** | Know which detection belongs to which person |
| **Input** | Video frames, detections from YOLOv10 |
| **Output** | Track IDs with bboxes |
| **Used By** | VideoAnalyticsPipeline |

### 3. Implementation Details
```python
# From tracker.py
class ByteTrackService:
    def track(self, frame, frame_id: int):
        results = self.native_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.low_threshold,
        )
        # Uses YOLO's built-in ByteTrack implementation
```

### 4. Runtime Verification

**Status:** ✅ Working

- ✅ Track IDs are assigned and maintained
- ✅ ByteTrack algorithm from Ultralytics
- ⚠️ No ground truth validation
- ⚠️ No metric tracking for ID switches/fragmentation

### 5. Performance Metrics

| Metric | Status |
|--------|--------|
| Multiple Object Tracking Accuracy (MOTA) | ❌ Not measured |
| ID Switches | ❌ Not tracked |
| Fragmentation | ❌ Not measured |
| FPS | ✅ Computed |

**Status:** ❌ No evaluation mechanism implemented

---

## Model #4: ArcFace (Face Recognition)

### 1. Identification
- **Name:** ArcFace (InsightFace)
- **Files:** `backend/services/reid.py` (line 13-50)
- **Framework:** InsightFace (PyTorch/ONNX)
- **Type:** Pretrained model
- **Weights Location:** `models/arcface/arcface.onnx` (CONFIGURED BUT NOT PRESENT)
- **Status:** ⚠️ **CRITICAL: Model not present**

### 2. Purpose

| Aspect | Details |
|--------|---------|
| **Why Used** | Generate 512-D face embeddings for matching |
| **Problem Solved** | Identify people by facial appearance |
| **Input** | Cropped face image (512x512) |
| **Output** | 512-D embedding vector |
| **Match Threshold** | 0.60 (config) |

### 3. Implementation Details

```python
class ArcFaceService:
    def __init__(self):
        self.app = FaceAnalysis(name=self.model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def extract_from_image(self, image: np.ndarray) -> EmbeddingResult:
        if self.app is None:
            return EmbeddingResult(None, 0.0)  # ← Returns None if init failed
        
        try:
            faces = self.app.get(image)
        except Exception:
            return EmbeddingResult(None, 0.0)  # ← Returns None on error
        
        if not faces:
            return EmbeddingResult(None, 0.0)  # ← Returns None if no faces
```

### 4. Runtime Verification

**Status:** ⚠️ **FALLBACK TO DETERMINISTIC HASH**

**Critical Issue:** When ArcFace fails (which it will if weights missing):

```python
# From reid.py, line 171-177
def _extract_embeddings(self, track_id, frame_id, frame, bbox):
    face_result = self.arcface.extract_from_image(crop)
    
    if face_result.embedding is None:
        # FALLBACK: Generate fake embedding from hash!
        face_embedding = deterministic_embedding(
            f"face:{track_id}:{frame_id}", 512
        )
```

**What this means:**
- If ArcFace fails or is not installed, the system generates a DETERMINISTIC HASH-BASED EMBEDDING
- The embedding is ALWAYS the same for the same (track_id, frame_id)
- Person matching becomes useless random hashing

### 5. Weight File Status

```python
# From config.py
arcface_weights: models/arcface/arcface.onnx
```

**Check:** `models/arcface/` contains only `test_arcface.py` - **NO WEIGHTS FILE**

❌ **Model weights missing - running on fallback hashing mode**

### 6. Confidence Analysis

| Aspect | Value | Issue |
|--------|-------|-------|
| Confidence Generation | Yes | Face detection confidence |
| Confidence Meaningful | ⚠️ Unknown | If using hash fallback, confidence is fake |
| Threshold Application | Yes (0.60) | Not meaningful if using hashes |

---

## Model #5: OSNet (Body Re-Identification)

### 1. Identification
- **Name:** OSNet x1.0 (OpenDS model)
- **Files:** `backend/services/reid.py` (line 59-95)
- **Framework:** torchreid
- **Type:** Pretrained model
- **Weights Location:** `models/osnet/osnet_x1_0.pth` (CONFIGURED BUT NOT PRESENT)
- **Status:** ⚠️ **CRITICAL: Model not present**

### 2. Purpose

| Aspect | Details |
|--------|---------|
| **Why Used** | Generate body appearance embedding |
| **Problem Solved** | Match people by clothing/body shape |
| **Input** | Cropped body image |
| **Output** | 256-D (or variable) embedding |

### 3. Fallback Mechanism

```python
class OSNetService:
    def __init__(self):
        self.extractor = FeatureExtractor(...)  # Fails if weights missing
    
    def extract_from_image(self, image: np.ndarray) -> EmbeddingResult:
        if self.extractor is None:
            return EmbeddingResult(None, 0.0)
```

**Then in MultiTierReIDService:**
```python
body_result = self.osnet.extract_from_image(crop)

if body_result.embedding is None:
    body_embedding = deterministic_embedding(
        f"body:{track_id}:{frame_id}", 512
    )
```

### 4. Weight File Status

❌ **Model weights missing - running on fallback hashing mode**

---

## Model #6: ST-GCN++ (Activity Recognition) ⚠️ **NOT A REAL MODEL**

### 1. Identification
- **Name:** "STGCNPlusPlusService"
- **Files:** `backend/services/activity.py`
- **Framework:** Claims to be Spatio-Temporal Graph Convolutional Network
- **Type:** ❌ **FAKE - This is not ST-GCN++, it's rule-based heuristics**
- **Status:** 🚧 Placeholder implementation

### 2. What the code CLAIMS to do:
```python
class STGCNPlusPlusService:
    """Supposedly uses ST-GCN++ model"""
    
    def classify(self, pose_record: PoseRecord, timestamp: float) -> ActivityRecord:
        # ... activity recognition ...
```

### 3. What it ACTUALLY does:

```python
keypoints = np.array(pose_record.keypoints)

# Pure rule-based classification!
wrists = keypoints[[15, 16], 1]  # y-coordinate of wrists
hips = keypoints[[23, 24], 1]     # y-coordinate of hips
ankles = keypoints[[27, 28], 1]   # y-coordinate of ankles

# Simple threshold comparisons:
if wrists.mean() < keypoints[0, 1]:  # If wrists above head
    label = "Hand Raising"
    confidence = 0.78  # HARDCODED!

elif np.abs(wrists[0] - wrists[1]) > 0.15:  # If wrists separated
    label = "Waving"
    confidence = 0.74  # HARDCODED!

elif np.abs(ankles[0] - ankles[1]) > 0.08:  # If ankles separated
    label = "Walking"
    confidence = 0.81  # HARDCODED!

elif hips.mean() > 0.7:  # If hips low
    label = "Sitting"
    confidence = 0.69  # HARDCODED!

else:
    label = "Standing"
    confidence = 0.76  # HARDCODED!
```

### 4. Assessment

| Claim | Reality | Status |
|-------|---------|--------|
| Uses ST-GCN++ model | Pure if-else rules | ❌ FALSE |
| Learned from data | Hardcoded thresholds | ❌ FALSE |
| Adaptive confidence | Arbitrary numbers | ❌ FALSE |
| Real activity recognition | Keypoint coordinate comparison | 🚧 Placeholder |

### 5. Confidence Analysis

**CRITICAL FINDING:** All confidence scores are HARDCODED:
- Hand Raising: 0.78
- Waving: 0.74  
- Walking: 0.81
- Sitting: 0.69
- Standing: 0.76

These numbers are **arbitrary and meaningless**. They don't reflect actual classification quality.

### 6. Runtime Verification

**Status:** 🚧 Runs but produces meaningless results

- ✅ Code executes without errors
- ✅ Returns activity labels
- ❌ Classification is not ML-based
- ❌ Confidence scores are arbitrary
- ❌ No learning from training data
- ❌ No validation capability

---

## Model #7: TMS (Temporal Motion Signature) ⚠️ **NOT A NEURAL NETWORK**

### 1. Identification
- **Name:** Temporal Motion Signature
- **Files:** `backend/services/tms.py`
- **Type:** ❌ **NOT a neural network - hand-crafted feature engineering**
- **Algorithm:** Manual feature extraction + DTW (Dynamic Time Warping)

### 2. What it does:

```python
class TemporalMotionSignatureService:
    def _compute_vector(self, window: list[PoseRecord]) -> np.ndarray:
        # Extract 16 hand-crafted features:
        
        stride = np.linalg.norm(left_ankle - right_ankle, axis=1)
        arm_swing = np.linalg.norm(left_wrist - right_wrist, axis=1)
        velocity = np.diff(hips, axis=0)
        posture = np.linalg.norm(shoulders - hips, axis=1)
        
        # Compute statistics:
        vector = np.array([
            stride.mean(),              # Feature 1
            stride.std(),               # Feature 2
            self._periodicity(stride),  # Feature 3
            self._cadence(stride),      # Feature 4
            arm_swing.mean(),           # Feature 5
            arm_swing.std(),            # Feature 6
            # ... 10 more features ...
        ])
```

### 3. Features extracted:

**These are NOT learned features. They're engineered:**
1. Stride (ankle distance) - mean & std
2. Periodicity of stride (FFT)
3. Cadence (stride changes)
4. Arm swing - mean & std
5. Periodicity of arm swing
6. Head height variation
7. Velocity (hip movement)
8. Posture (shoulder-hip distance)
9. Velocity components (x, y)

### 4. Matching mechanism:

```python
# Uses DTW (Dynamic Time Warping) - not learned matching
def dtw_distance(self, vector_a, vector_b) -> float:
    # Manual DTW implementation (not neural)
    # Pure dynamic programming algorithm
```

### 5. Assessment

| Aspect | Status |
|--------|--------|
| Is it a neural network? | ❌ NO |
| Uses machine learning? | ❌ NO |
| Hand-crafted features? | ✅ YES |
| Trainable? | ❌ NO |
| Meaningful for matching? | ⚠️ Possibly |

**Verdict:** This is feature engineering + classical algorithm, not AI/ML

---

## Model #8: Ollama LLM (Natural Language Query Engine)

### 1. Identification
- **Name:** Ollama + LLaMA 3.2 (configurable)
- **Files:** `backend/services/query_engine.py`
- **Framework:** Ollama (local LLM runner)
- **Type:** ⚠️ Optional - has SQL fallback
- **Config:** `ollama_model: llama3.2`
- **Status:** ⚠️ **Optional dependency with hardcoded fallback**

### 2. Purpose

| Aspect | Details |
|--------|---------|
| **Why Used** | Convert natural language to SQL queries |
| **Problem Solved** | Allow users to query DB in English |
| **Input** | Natural language question |
| **Output** | SQL query |

### 3. Implementation

```python
class NLQueryEngine:
    def _generate_sql(self, prompt: str) -> str:
        system_prompt = """
        You translate surveillance questions to SQLite SQL.
        Available tables: persons, tracks, activities, tms_vectors, events, queries.
        Return only SQL.
        """
        
        try:
            if ollama is None:
                raise RuntimeError("ollama not installed")
            
            response = ollama.chat(model=self.model_name, messages=[...])
            return response["message"]["content"].strip()
        
        except Exception:
            # FALLBACK to hardcoded SQL!
            if "waved" in prompt.lower():
                return "SELECT * FROM activities WHERE LOWER(activity) = 'waving' ..."
            if "entered" in prompt.lower():
                return "SELECT * FROM events WHERE LOWER(event_type) = 'entry' ..."
            return "SELECT * FROM events ORDER BY timestamp DESC LIMIT 20"
```

### 4. Runtime Verification

**Status:** ⚠️ **Partially working with hardcoded fallbacks**

- ⚠️ Ollama not installed (likely)
- ✅ Falls back to hardcoded SQL queries
- ⚠️ Only 2 hardcoded patterns ("waved", "entered")
- ⚠️ Any other query returns generic "SELECT * FROM events"

### 5. Issues

**Critical Issue:** The system **DOESN'T TELL THE USER it's using fallback SQL**

```python
# Result of query("What activities happened?")
# Actually runs: SELECT * FROM events ORDER BY timestamp DESC LIMIT 20
# But the frontend shows it as if it was actually translated
```

---

# DATA FLOW ANALYSIS

## Complete End-to-End Pipeline

```
VIDEO UPLOAD (Frontend)
    ↓
/upload-video endpoint
    ↓
VideoAnalyticsPipeline.process_video()
    ↓
[Frame-by-frame processing]
    ├─ YOLOv10.detect() [Person detection] ✅
    │
    ├─ ByteTrack.track() [Tracking] ✅
    │
    ├─ MediaPipePose.extract() [Skeleton] ⚠️
    │
    ├─ MultiTierReIDService.identify()
    │  ├─ ArcFace.extract() [Face embedding] ⚠️ FALLBACK TO HASH
    │  ├─ OSNet.extract() [Body embedding] ⚠️ FALLBACK TO HASH
    │  └─ FAISS.search() [Find match] ⚠️ Searching hashes
    │
    ├─ STGCNPlusPlusService.classify() [Activity] 🚧 Rule-based
    │
    ├─ TemporalMotionSignatureService.update() [Motion] ✅ Feature extraction
    │
    ├─ BehaviorDiscoveryService.update() [Behavior] ⚠️ Uses deterministic hash
    │
    └─ Repository.add_track/activity/event/tms_vector() [DB Storage] ✅
         ↓
         SQLite: tracks, activities, tms_vectors, events tables
    
    ↓
VIDEO OUTPUT
```

## Person Search Flow

```
/upload-reference-image
    ↓
MultiTierReIDService.register_reference(person_id, image_path)
    ├─ ArcFace.extract_from_image() → EmbeddingResult
    │  ├─ If success: face_embedding (512-D) ✅
    │  └─ If fail: deterministic_embedding() ⚠️ HASH
    │
    ├─ OSNet.extract_from_image() → EmbeddingResult
    │  ├─ If success: body_embedding (256-D) ✅
    │  └─ If fail: deterministic_embedding() ⚠️ HASH
    │
    └─ FAISS.add("face_embeddings", face_embedding)
    └─ FAISS.add("body_embeddings", body_embedding)


/search-person {person_id}
    ↓
Query: deterministic_embedding(f"face-ref:{person_id}:lookup", 512)
    ↓
FAISS.search("face_embeddings", query)
    ↓
Return matches where person_id equals
```

**Issue:** Search query is ALSO a deterministic hash! 
- If ArcFace is offline, EVERYTHING becomes hash-based lookup
- The embedding for search is NOT from actual face extraction

## Query Engine Flow

```
/query {prompt}
    ↓
NLQueryEngine.query(prompt)
    ├─ If contains "behavior": BehaviorDiscoveryService.search()
    │  └─ Uses deterministic_embedding(query, 128) ⚠️ HASH
    │
    └─ Else: _generate_sql(prompt)
       ├─ Try: Ollama LLM translation ⚠️ Optional
       └─ Fallback: Hardcoded SQL patterns
           ├─ "waved" → SELECT waving activities
           ├─ "entered" → SELECT entry events
           └─ Default: SELECT all events
    ↓
Run SQL on SQLite
    ↓
Format and return results
```

---

# RUNTIME VERIFICATION

## Component Status Matrix

| Component | Framework | Status | Working? | Falls Back? |
|-----------|-----------|--------|----------|------------|
| YOLOv10 | Ultralytics | ✅ Ready | YES | HOG detector |
| ByteTrack | Ultralytics | ✅ Ready | YES | N/A |
| MediaPipe Pose | Google MediaPipe | ⚠️ Optional | Maybe | Returns None |
| ArcFace | InsightFace | ❌ Missing weights | NO | Hash function |
| OSNet | torchreid | ❌ Missing weights | NO | Hash function |
| ST-GCN++ | (Claimed) | 🚧 Fake | Runs but meaningless | N/A |
| TMS | Manual | ✅ Works | YES | Feature extraction |
| Ollama LLM | Ollama | ⚠️ Optional | Maybe | SQL fallbacks |

## File-by-File Runtime Status

| File | Model | Status | Issues |
|------|-------|--------|--------|
| `detector.py` | YOLOv10 | ✅ WORKS | None |
| `tracker.py` | ByteTrack | ✅ WORKS | None |
| `pose.py` | MediaPipe | ⚠️ OPTIONAL | Silent None returns |
| `reid.py` | ArcFace/OSNet | ⚠️ DEGRADED | Fallback hashing |
| `activity.py` | ST-GCN++ | 🚧 FAKE | Rule-based only |
| `tms.py` | TMS | ✅ WORKS | Features only |
| `behavior.py` | (N/A) | ⚠️ DEGRADED | Hash-based search |
| `query_engine.py` | Ollama | ⚠️ OPTIONAL | Hardcoded fallbacks |
| `analytics.py` | (N/A) | ✅ WORKS | Statistics only |

---

# CONFIDENCE ANALYSIS

## Per-Component Confidence Scoring

### YOLOv10 Detection
- **Confidence Generated:** ✅ Yes (model output)
- **Range:** 0.0 - 1.0
- **Meaningful:** ✅ Yes (actual detection confidence)
- **Threshold:** 0.25 (configurable)
- **Low-confidence Filtering:** ✅ Applied

### ByteTrack
- **Confidence Generated:** ✅ Yes (tracking score)
- **Meaning:** Track validity
- **Meaningful:** ✅ Yes

### ArcFace Face Matching
- **Confidence Generated:** ✅ Face detection confidence (if model available)
- **Threshold:** 0.60 (config)
- **Meaningful:** ⚠️ **Unknown if fallback hashing is active**
- **Issue:** If weights missing, confidence on hash-based embeddings is MEANINGLESS

### OSNet Body Matching
- **Confidence Generated:** ✅ If model available
- **Threshold:** 0.50 (config)
- **Meaningful:** ⚠️ **Unknown if fallback hashing is active**

### ST-GCN++ Activity Recognition
- **Confidence Generated:** ✅ Yes
- **Range:** 0.69 - 0.81 (hardcoded)
- **Meaningful:** ❌ NO - these are arbitrary numbers
- **Issue:** Same confidence regardless of actual classification quality

### TMS Motion Signature
- **Confidence Generated:** ✅ Similarity score from FAISS
- **Range:** 0.0 - 1.0
- **Meaningful:** ⚠️ Possibly (depends on embedding quality)

### Query Engine
- **Confidence Generated:** ❌ No confidence in generated SQL
- **Issue:** User doesn't know if SQL was LLM-generated or hardcoded fallback

---

# DATABASE VALIDATION

## Schema

```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY,
    person_id TEXT UNIQUE,
    label TEXT,
    created_at TEXT
);

CREATE TABLE tracks (
    id INTEGER PRIMARY KEY,
    track_id INTEGER,
    frame_id INTEGER,
    timestamp REAL,
    bbox TEXT,           -- JSON string!
    identity TEXT,
    confidence REAL
);

CREATE TABLE activities (
    id INTEGER PRIMARY KEY,
    track_id INTEGER,
    activity TEXT,
    confidence REAL,
    timestamp REAL
);

CREATE TABLE tms_vectors (
    id INTEGER PRIMARY KEY,
    track_id INTEGER,
    vector TEXT,          -- JSON string!
    similarity_score REAL,
    created_at TEXT
);

CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    event_type TEXT,
    timestamp REAL,
    activity TEXT,
    location TEXT,
    track_id INTEGER,
    identity TEXT,
    confidence REAL,
    metadata TEXT        -- JSON string!
);

CREATE TABLE queries (
    id INTEGER PRIMARY KEY,
    query_text TEXT,
    generated_sql TEXT,
    response_text TEXT,
    created_at TEXT
);
```

## Data Validation Findings

### Issue #1: Data Type Misuse
- **bbox stored as JSON string** (should be separate columns or blob)
- **vector stored as JSON string** (should be blob for efficiency)
- **metadata stored as JSON string** (redundant encoding)

### Issue #2: No Foreign Key Constraints
- track_id in activities NOT linked to tracks table
- track_id in events NOT linked to tracks table
- person_id in tables NOT linked to persons table
- **Consequence:** Orphaned records possible

### Issue #3: No Duplicate Detection
- Same activity can be recorded multiple times for same track
- No UNIQUE constraint to prevent duplicates
- Same event can be logged multiple times

### Issue #4: Timestamp Precision
- Using REAL (float) for timestamps
- **Issue:** Floating point precision errors in comparisons
- Should use INTEGER (milliseconds) or TEXT (ISO 8601)

### Issue #5: Confidence Value Validation
- No CHECK constraint (confidence BETWEEN 0 AND 1)
- **Consequence:** Invalid confidence values (e.g., 1.5, -0.2) can be stored

## Query Validation Findings

```python
# From repository.py
def run_sql(self, sql: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]
```

**Critical Issue:** **SQL Injection Vulnerability**
- User-provided SQL from query engine is executed directly
- No parameterization
- If Ollama generates malicious SQL, database could be compromised

### Example Attack Vector:
```
User query: "SELECT * FROM tracks; DROP TABLE activities; --"

If LLM somehow generates this SQL, it would execute!
```

---

# QUERY ENGINE VALIDATION

## NLQuery to SQL Translation

### Current Implementation

```python
def _generate_sql(self, prompt: str) -> str:
    try:
        if ollama is None:
            raise RuntimeError("ollama not installed")
        
        # Try LLM translation
        response = ollama.chat(model=self.model_name, messages=[...])
        return response["message"]["content"].strip()
    
    except Exception:
        # Fallback
        if "waved" in prompt.lower():
            return "SELECT * FROM activities WHERE LOWER(activity) = 'waving' ..."
        if "entered" in prompt.lower():
            return "SELECT * FROM events WHERE LOWER(event_type) = 'entry' ..."
        return "SELECT * FROM events ORDER BY timestamp DESC LIMIT 20"
```

### Issues Found

#### Issue #1: Hallucinated Table Names
The system tells users it has these tables:
```
"Available tables: persons, tracks, activities, tms_vectors, events, queries"
```

But there's NO validation that these tables actually exist or have the expected schema.

#### Issue #2: Silent Fallback
When Ollama fails (likely scenario), the user is NOT informed:
- ✅ Query "How many people waved?" → Gets hardcoded SQL
- ❌ User thinks LLM translated it
- ❌ No indication in output that it's a fallback

#### Issue #3: Incomplete Fallback Coverage
Only 2 fallback patterns:
- "waved" → Hardcoded SQL for waving activities
- "entered" → Hardcoded SQL for entry events  
- Everything else → Generic SELECT all events

#### Issue #4: No Output Validation
```python
# User can ask anything, system tries LLM or fallback
if "behavior" in prompt.lower():
    results = self.behavior_service.search(prompt)
    # Uses deterministic_embedding(prompt, 128) ⚠️ HASH-BASED
else:
    results = repository.run_sql(sql)
```

#### Issue #5: Answer Hallucination Risk
If using Ollama:
- LLM could generate invalid SQL
- LLM could misinterpret question
- LLM could fabricate statistics

### How to Detect Hallucination

**To verify if query engine is hallucinating:**

1. Ask a question with a known answer: "How many people were tracked?"
2. Check database directly: `SELECT COUNT(DISTINCT track_id) FROM tracks;`
3. Compare with system output
4. If LLM is running, run same query 10 times - results should be same but LLM might vary
5. Ask questions about non-existent data: "Show me all people named 'XYZ'" - check if fabricated

---

# CRITICAL ISSUES SUMMARY

## Issue Severity Matrix

| Component | Issue | Severity | Impact |
|-----------|-------|----------|--------|
| ArcFace | Weights missing, fallback to hash | 🔴 CRITICAL | Face matching = random hashing |
| OSNet | Weights missing, fallback to hash | 🔴 CRITICAL | Body matching = random hashing |
| ST-GCN++ | Not a real model, rule-based only | 🔴 CRITICAL | Activity recognition invalid |
| Ollama LLM | Optional, hardcoded fallbacks | 🟡 MEDIUM | Queries may not translate correctly |
| Query Engine | SQL injection vulnerability | 🔴 CRITICAL | Database at risk |
| Database | No foreign keys | 🟡 MEDIUM | Data integrity issues |
| Database | Confidence not validated | 🟡 MEDIUM | Invalid scores possible |
| TMS | Hand-crafted only, not ML | 🟢 LOW | Works as intended but limited |
| MediaPipe | Silent failures | 🟡 MEDIUM | Activity recognition fails silently |
| MultiTier ReID | Confidence misleading | 🟡 MEDIUM | Users trust invalid scores |

## Root Causes

### Root Cause #1: Missing Model Weights
- ArcFace: `models/arcface/arcface.onnx` not present
- OSNet: `models/osnet/osnet_x1_0.pth` not present
- ST-GCN++: `models/stgcn/stgcn_plus_plus.pth` not present

**Consequence:** System falls back to deterministic hashing

### Root Cause #2: No Validation During Fallback
```python
# When models fail, system generates hash but doesn't log it
if face_embedding is None:
    face_embedding = deterministic_embedding(f"face:{track_id}:{frame_id}", 512)
    # ← No log warning!
    # ← User never knows this happened!
```

### Root Cause #3: Arbitrary Confidence Scores
Activity recognition hardcodes confidence values:
```python
if wrists.mean() < keypoints[0, 1]:
    confidence = 0.78  # Why 0.78? No justification!
```

### Root Cause #4: No Evaluation Framework
- No test set for validation
- No accuracy/precision/recall metrics
- No ground truth against which to compare
- No continuous monitoring

---

# FINAL ASSESSMENT

## Can This System Be Trusted? 🔴 **NO**

### Summary Finding

**Out of 8 claimed AI components:**
- ✅ **2 working properly:** YOLOv10 (detection), ByteTrack (tracking)
- ⚠️ **2 optional but working:** MediaPipe Pose, TMS (if available)
- ⚠️ **2 degraded:** ArcFace, OSNet (falling back to deterministic hashing)
- 🚧 **2 fake/placeholder:** ST-GCN++ (rule-based masquerading as AI), Ollama (optional with hardcoded fallbacks)

### Why You Cannot Trust This System

#### 1. **Person Identification is Unreliable** 🔴
- ArcFace weights missing → Falls back to deterministic hash of track_id
- Result: Same hash for same person across all videos (good luck)
- OSNet weights missing → Same issue with body embeddings
- Matching algorithm: FAISS searching hashes = random hash matching

**Real Impact:** "Finding" a person is essentially searching through track IDs, not real face/body matching

#### 2. **Activity Recognition is Arbitrary** 🔴
- ST-GCN++ doesn't exist - only rule-based if-statements
- Confidence scores hardcoded (not learned)
- Only detects 5 activities (Walking, Standing, Sitting, Waving, Hand Raising)
- Any other motion = misclassified

**Real Impact:** Activity classifications are guesses based on keypoint coordinates

#### 3. **Query Engine May Hallucinate** 🔴
- Ollama likely not installed
- Falls back to hardcoded SQL for 2 patterns
- Any other query returns generic results
- No indication to user that it's a fallback
- SQL injection vulnerability

**Real Impact:** Users get potentially fabricated results and don't know it

#### 4. **Confidence Scores are Misleading** 🟡
- Face/body matching confidence meaningless if using hashes
- Activity confidence arbitrary
- No validation of scores
- Database doesn't enforce 0-1 range

**Real Impact:** Users think system is more confident than it actually is

#### 5. **No Validation Framework** 🔴
- Zero accuracy metrics
- Zero precision/recall evaluation
- No ground truth testing
- No continuous monitoring

**Real Impact:** You don't know if system is working or just hallucinating

---

## Component-by-Component Trustworthiness

| Component | Trust Level | Why |
|-----------|------------|-----|
| YOLOv10 Detection | ✅ HIGH | Real model, working |
| ByteTrack Tracking | ✅ HIGH | Real algorithm, working |
| MediaPipe Pose | ✅ MEDIUM | Real model, optional dependency |
| Face Recognition | ❌ ZERO | Fallback to hashing |
| Body Matching | ❌ ZERO | Fallback to hashing |
| Activity Recognition | ❌ ZERO | Rule-based, hardcoded confidence |
| Person Search | ❌ ZERO | Searching hashes |
| Query Engine | ⚠️ LOW | Hardcoded fallbacks, SQL injection risk |
| Behavior Discovery | ⚠️ LOW | Hash-based matching |

---

## Recommendations

### Immediate (CRITICAL - Do Before Production)

1. **Obtain and validate model weights:**
   - [ ] Download ArcFace weights
   - [ ] Download OSNet weights
   - [ ] Test both models can load

2. **Disable fallback hashing:**
   - [ ] Remove deterministic_embedding fallback
   - [ ] Fail explicitly if models unavailable
   - [ ] Log all fallback activations

3. **Fix activity recognition:**
   - [ ] Either train real ST-GCN++ or
   - [ ] Rename to "HeuristicActivityClassifier"
   - [ ] Document as rule-based, not ML

4. **Fix SQL injection vulnerability:**
   - [ ] Use parameterized queries
   - [ ] Never execute user-generated SQL directly
   - [ ] Validate Ollama output before execution

5. **Add input validation:**
   - [ ] Enforce confidence scores 0-1
   - [ ] Add foreign key constraints
   - [ ] Prevent duplicate events

### Short Term (1-2 weeks)

6. **Build evaluation framework:**
   - [ ] Create validation dataset
   - [ ] Compute accuracy/precision/recall
   - [ ] Track metrics over time

7. **Add explainability:**
   - [ ] Show which model generated each result
   - [ ] Indicate fallback activations
   - [ ] Display confidence ranges

8. **Improve query engine:**
   - [ ] If using Ollama, validate SQL output
   - [ ] Expand fallback patterns
   - [ ] Return confidence in query results

### Long Term (1-2 months)

9. **Train custom models:**
   - [ ] Fine-tune activity recognition
   - [ ] Re-train re-ID models
   - [ ] Create evaluation benchmarks

10. **Implement monitoring:**
    - [ ] Real-time accuracy tracking
    - [ ] Alert on confidence drops
    - [ ] A/B test model improvements

---

## How to Verify Findings

### Verification #1: Missing Weights
```bash
# Check if files exist
ls -la models/arcface/      # Should have .onnx or .pt file
ls -la models/osnet/        # Should have .pth file
ls -la models/stgcn/        # Should have .pth file

# Check config
grep -r "arcface_weights" backend/config.py  # See configured path
```

**Expected Finding:** Files don't exist → System uses hashing

### Verification #2: Deterministic Hashing
```python
# Test embeddings
from backend.services.embedding_utils import deterministic_embedding

e1 = deterministic_embedding("face:track-1:frame-0", 512)
e2 = deterministic_embedding("face:track-1:frame-0", 512)

print(np.allclose(e1, e2))  # Should print True - SAME HASH ALWAYS!
```

**Expected Finding:** Same input = same hash (not learned)

### Verification #3: Activity Hardcoding
```python
# Check activity service
from backend.services.activity import STGCNPlusPlusService

# Create fake pose where hand is raised
activity = service.classify(pose_with_raised_hand, timestamp)
print(activity.confidence)  # Will always be 0.78 for "Hand Raising"

activity2 = service.classify(pose_with_raised_hand, timestamp)
print(activity2.confidence)  # Will ALWAYS be 0.78 - no variation!
```

**Expected Finding:** Same activity type = same confidence always

### Verification #4: Query Fallback
```python
# Try query without Ollama installed
from backend.services.query_engine import NLQueryEngine

engine = NLQueryEngine(repository, behavior_service)

# Ask random question
result = engine.query("What is the weather?")
print(result.sql)  # Will be hardcoded generic SQL (or None)
print("waved" in "What is the weather?")  # False
print("entered" in "What is the weather?")  # False
# So will fall back to: SELECT * FROM events...
```

**Expected Finding:** Unknown questions get generic fallback SQL

---

## Audit Conclusion

**VIAS is a hybrid system that misrepresents its capabilities:**

- ✅ Core detection and tracking work well
- ❌ Person re-identification is non-functional (hashing only)
- ❌ Activity recognition is rule-based, not AI
- ⚠️ Query engine is optional with limited fallbacks
- 🔴 Confidence scores are misleading
- 🔴 No validation framework exists

**Grade: F (Failing for production use)**

The system conflates working components (detection/tracking) with non-working ones (re-id, activity) and presents both as equally reliable. Users cannot trust person identification, activity labels, or query results without independent verification.

**Status: Do not deploy to production without addressing critical issues.**

---

# APPENDIX

## A. Hyperlink Reference

- [YOLOv10 Detection](#model-1-yolov10-person-detection)
- [Activity Recognition](#model-6-st-gcn-activity-recognition)
- [Face Recognition](#model-4-arcface-face-recognition)
- [Query Engine](#model-8-ollama-llm-natural-language-query-engine)
- [Database Issues](#database-validation)
- [Critical Issues](#critical-issues-summary)

## B. File Manifest

**Backend Services:**
- `detector.py` - YOLOv10 detection
- `tracker.py` - ByteTrack tracking
- `pose.py` - MediaPipe pose
- `reid.py` - Face/Body re-identification
- `activity.py` - Activity recognition
- `tms.py` - Motion signature
- `behavior.py` - Behavior discovery
- `query_engine.py` - NL query engine
- `faiss_store.py` - Vector database
- `analytics.py` - Analytics

**Configuration:**
- `config.py` - Settings structure
- `configs/settings.yaml` - Actual settings

**Database:**
- `database/db.py` - Schema
- `database/repository.py` - Queries

**Routes:**
- `api/routes.py` - API endpoints
- `api/realtime_routes.py` - Real-time endpoints

## C. Models Configuration

From `configs/settings.yaml`:
```yaml
models:
  yolov10_weights: models/yolov10/yolov10n.pt          [✅ EXISTS]
  arcface_weights: models/arcface/arcface.onnx         [❌ MISSING]
  arcface_model_name: buffalo_l
  osnet_weights: models/osnet/osnet_x1_0.pth          [❌ MISSING]
  osnet_model_name: osnet_x1_0
  stgcn_weights: models/stgcn/stgcn_plus_plus.pth     [❌ MISSING]
  ollama_model: llama3.2                              [⚠️ OPTIONAL]
```

---

**END OF AUDIT REPORT**

**Report Generated:** 2026-06-15  
**Auditor:** AI Systems Auditor  
**Status:** ⚠️ CRITICAL ISSUES - NOT PRODUCTION READY
