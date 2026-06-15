# VIAS AI AUDIT - TECHNICAL FINDINGS MATRIX

## Component Analysis Matrix

### Model #1: YOLOv10 Detector
```
Name:                   YOLOv10 Nano
Framework:              Ultralytics PyTorch
File:                   backend/services/detector.py
Status:                 ✅ WORKING
Real AI Model:          ✅ YES
Weights Present:        ✅ models/yolov10/yolov10n.pt
Falls Back:             ✅ HOG detector (if weights fail)
Confidence Scoring:     ✅ Real model output
Evaluation Present:     ❌ NO
```

### Model #2: ByteTrack
```
Name:                   ByteTrack
Framework:              Ultralytics (built into YOLOv10)
File:                   backend/services/tracker.py
Status:                 ✅ WORKING
Real AI Model:          ✅ YES
Confidence Scoring:     ✅ Track validity score
Evaluation Present:     ❌ NO (MOTA not computed)
```

### Model #3: MediaPipe Pose
```
Name:                   MediaPipe Pose
Framework:              Google MediaPipe
File:                   backend/services/pose.py
Status:                 ⚠️ OPTIONAL
Real AI Model:          ✅ YES
Dependency:             ⚠️ Optional (try/except)
Confidence Scoring:     ⚠️ Available but ignored
Evaluation Present:     ❌ NO
```

### Model #4: ArcFace
```
Name:                   ArcFace (InsightFace)
Framework:              InsightFace
File:                   backend/services/reid.py
Status:                 ❌ BROKEN
Real AI Model:          ✅ YES
Weights Present:        ❌ MISSING (config says models/arcface/arcface.onnx)
Actually Present:       ❌ Only test_arcface.py
Fallback Behavior:      🚧 deterministic_embedding() hashing
Confidence Scoring:     ⚠️ Fake if fallback active
Evaluation Present:     ❌ NO
```

### Model #5: OSNet
```
Name:                   OSNet x1.0
Framework:              torchreid
File:                   backend/services/reid.py
Status:                 ❌ BROKEN
Real AI Model:          ✅ YES
Weights Present:        ❌ MISSING (config says models/osnet/osnet_x1_0.pth)
Actually Present:       ❌ NO
Fallback Behavior:      🚧 deterministic_embedding() hashing
Confidence Scoring:     ⚠️ Fake if fallback active
Evaluation Present:     ❌ NO
```

### Model #6: ST-GCN++
```
Name:                   "STGCNPlusPlusService"
Framework:              (Claimed) PyTorch/TensorFlow
File:                   backend/services/activity.py
Status:                 🚧 FAKE
Real AI Model:          ❌ NO - This is rule-based if-else
Claims To Do:           Spatio-Temporal Graph Convolution
Actually Does:          Simple keypoint coordinate thresholds
Weights Present:        ❌ MISSING (config: models/stgcn/stgcn_plus_plus.pth)
Confidence Scoring:     ❌ Hardcoded (0.69-0.81)
Evaluation Present:     ❌ NO

Details:
  Walking:       ankles distance > 0.08 → confidence 0.81
  Waving:        wrists distance > 0.15 → confidence 0.74
  Hand Raising:  wrists above head → confidence 0.78
  Sitting:       hips y > 0.7 → confidence 0.69
  Standing:      everything else → confidence 0.76

Training:               ❌ NOT TRAINABLE
Learned Features:       ❌ NO
Data-Driven:            ❌ NO
```

### Model #7: TMS (Temporal Motion Signature)
```
Name:                   Temporal Motion Signature
Framework:              Custom hand-crafted + DTW
File:                   backend/services/tms.py
Status:                 ✅ WORKING
Real AI Model:          ❌ NO (hand-crafted features)
Trainable:              ❌ NO
Features Extracted:     ✅ 16 manual features (stride, arm swing, etc)
Matching Algorithm:     ✅ DTW (Dynamic Time Warping)
Confidence Scoring:     ✅ FAISS similarity score
Evaluation Present:     ❌ NO
```

### Model #8: Ollama LLM
```
Name:                   Ollama (LLaMA 3.2)
Framework:              Ollama (local LLM runner)
File:                   backend/services/query_engine.py
Status:                 ⚠️ OPTIONAL
Real AI Model:          ✅ YES (if installed)
Installed:              ⚠️ Likely NO (unspecified dependency)
Fallback Behavior:      ✅ Hardcoded SQL patterns
Fallback Coverage:      ⚠️ Only 2 patterns ("waved", "entered")
Confidence Scoring:     ❌ NO
Evaluation Present:     ❌ NO

Security Issue:         ⚠️ SQL injection risk (no parameterization)
```

---

## Critical Issues Checklist

### 🔴 CRITICAL (Must Fix Before Production)

- [ ] **Missing ArcFace weights**
  - Location: models/arcface/arcface.onnx
  - Current: Only test file exists
  - Impact: Face recognition falls back to hashing
  - Fix: Download/train weights

- [ ] **Missing OSNet weights**
  - Location: models/osnet/osnet_x1_0.pth
  - Current: Doesn't exist
  - Impact: Body matching falls back to hashing
  - Fix: Download/train weights

- [ ] **ST-GCN++ doesn't exist**
  - Current: Rule-based if-else statements
  - Claims: Uses deep learning
  - Impact: Activity recognition is arbitrary rules
  - Fix: Train real model OR rename and document as rules

- [ ] **SQL Injection Vulnerability**
  - Location: query_engine.py, repository.py
  - Issue: User SQL executed directly
  - Impact: Database could be compromised
  - Fix: Use parameterized queries, validate output

### 🟡 MEDIUM (Should Fix Soon)

- [ ] **Confidence Scores Hardcoded**
  - Issue: Activity recognition has fixed confidences
  - Impact: Users trust arbitrary numbers
  - Fix: Compute real confidence or document as estimates

- [ ] **No Evaluation Framework**
  - Issue: No accuracy/precision/recall metrics
  - Impact: Can't verify if system works
  - Fix: Create validation dataset, track metrics

- [ ] **Silent Fallbacks**
  - Issue: When models fail, system uses fallback without warning
  - Impact: User doesn't know which mode running
  - Fix: Log and display fallback activation

- [ ] **No Foreign Key Constraints**
  - Issue: Database lacks referential integrity
  - Impact: Orphaned records possible
  - Fix: Add constraints during schema redesign

### 🟢 LOW (Nice to Have)

- [ ] **Timestamp Precision Issues**
  - Issue: Using float for timestamps
  - Fix: Use INTEGER (milliseconds) or ISO 8601

- [ ] **JSON-in-Database**
  - Issue: bbox, vector, metadata stored as JSON strings
  - Fix: Use proper schema design

---

## Data Flow Issues

### Face Recognition Path
```
┌─────────────────────────────────────────────────┐
│ Frontend: Upload reference image                │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ Register: register_reference()                  │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ Try: ArcFace.extract() [WEIGHTS MISSING]        │
│ Result: Returns None                            │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ Fallback: deterministic_embedding()             │
│ Hash = SHA256(f"face-ref:{person_id}:...")     │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ Store in FAISS: face_embeddings                │
│ (Actually storing hash, not face embedding)    │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ Later: User searches for person                │
│ Query embedding = deterministic_embedding()    │
│ (Also a hash!)                                 │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ FAISS search: Find matching hashes             │
│ Result: Hash match = person "found"            │
│ ⚠️ This is NOT real face recognition!          │
└─────────────────────────────────────────────────┘
```

### Issue: Every stage uses same deterministic hash
- Register: hash(person_id) → stored in FAISS
- Search: hash(person_id) → matched in FAISS
- Result: Always finds the "same" person back, not real matching

---

## Confidence Score Breakdown

### Real Confidence (Can Trust)
- YOLOv10 detection: Model output probability ✅
- ByteTrack tracking: Tracking validity ✅
- MediaPipe pose: Per-keypoint confidence (if extracted) ✅
- TMS similarity: FAISS L2 distance ✅

### Arbitrary Confidence (Cannot Trust)
- Activity recognition: Hardcoded values ❌
- Face matching: If using hash fallback ⚠️
- Body matching: If using hash fallback ⚠️
- Query results: No confidence at all ❌

### Missing Confidence
- Behavior search: Uses hash, no meaning ❌
- Behavior summary: Text-based, no score ❌

---

## Evaluation Status

### What's Evaluated
- FPS (frames per second) ✅ Computed
- Detection count ✅ Logged
- Track count ✅ Logged
- Activity count ✅ Logged

### What's NOT Evaluated
| Metric | Present? | Why Important |
|--------|----------|--------------|
| Accuracy | ❌ NO | Know if detections correct |
| Precision | ❌ NO | Know false positive rate |
| Recall | ❌ NO | Know false negative rate |
| F1-Score | ❌ NO | Combined metric |
| MOTA | ❌ NO | Tracking quality |
| ID Switches | ❌ NO | Track consistency |
| Face Recognition Rate | ❌ NO | Re-ID works? |
| Activity Accuracy | ❌ NO | Activity labels correct? |
| Query Accuracy | ❌ NO | Answers correct? |

---

## Model Weight Inventory

```
Location: models/

yolov10/
├── yolov10n.pt                      ✅ EXISTS (53.9 MB)

arcface/
├── test_arcface.py                  ✅ EXISTS (test file)
├── arcface.onnx                     ❌ MISSING (configured but not present)

osnet/                               ❌ MISSING (directory doesn't exist)
├── osnet_x1_0.pth                   ❌ MISSING (configured but not present)

stgcn/                               ❌ MISSING (directory doesn't exist)
├── stgcn_plus_plus.pth              ❌ MISSING (configured but not present)
```

**Total Impact:** 3 out of 4 configurable model weights are MISSING

---

## SQL Injection Vulnerability

### Vulnerability Location
```python
# query_engine.py
def query(self, prompt: str) -> QueryResponse:
    # ...
    sql = self._generate_sql(prompt)  # May be from Ollama LLM
    results = self.repository.run_sql(sql)  # Executed directly!

# repository.py
def run_sql(self, sql: str) -> list[dict]:
    conn.execute(sql)  # ⚠️ NO PARAMETERIZATION!
```

### Attack Vector
```
User: "show me events; DROP TABLE tracks; --"

If Ollama somehow generates this SQL:
  - It executes directly
  - Table gets deleted
  - Data loss
```

### Fix Required
```python
# WRONG (current):
conn.execute(f"SELECT * FROM tracks WHERE id = {user_id}")

# RIGHT:
conn.execute("SELECT * FROM tracks WHERE id = ?", (user_id,))
```

---

## Fallback Coverage Analysis

### Query Engine Fallbacks
```
User Query              Fallback SQL
─────────────────────────────────────────────────
"How many waved?"      SELECT * FROM activities 
                       WHERE activity = 'waving'

"Who entered?"         SELECT * FROM events
                       WHERE event_type = 'entry'

"What happened?"       SELECT * FROM events
                       ORDER BY timestamp DESC
                       LIMIT 20

"Anything else..."     SELECT * FROM events
                       ORDER BY timestamp DESC
                       LIMIT 20
```

**Coverage:** 2 specific patterns + 1 generic fallback
**Issue:** 99% of queries probably get generic fallback

---

## Database Schema Issues

### Issue #1: Data Type Misuse
```sql
-- WRONG (current):
bbox TEXT,              -- JSON string stored as text
vector TEXT,            -- JSON string stored as text
metadata TEXT           -- JSON string stored as text

-- RIGHT:
bbox BLOB,              -- Binary serialized
vector BLOB,            -- Binary serialized
-- OR break into proper columns
```

### Issue #2: Missing Constraints
```sql
-- MISSING CONSTRAINTS:
ALTER TABLE activities ADD CHECK (confidence BETWEEN 0 AND 1);
ALTER TABLE tracks ADD FOREIGN KEY (identity) REFERENCES persons(person_id);
-- etc.
```

### Issue #3: No Uniqueness
```sql
-- Same activity can be logged multiple times:
INSERT INTO activities (track_id, activity, ...) VALUES (1, 'walking', ...);
INSERT INTO activities (track_id, activity, ...) VALUES (1, 'walking', ...);
-- Both inserted! (no UNIQUE constraint)
```

---

## Test Cases for Verification

### Test #1: Verify Hashing
```python
from backend.services.embedding_utils import deterministic_embedding
from backend.services.reid import MultiTierReIDService

# Create two embeddings with same input
e1 = deterministic_embedding("test", 512)
e2 = deterministic_embedding("test", 512)

# Should be identical (proof it's hashing, not ML)
assert np.allclose(e1, e2), "Embeddings differ - proof of real learning"
# This WILL pass, proving it's hashing
```

### Test #2: Verify Missing Weights
```python
from backend.services.reid import ArcFaceService

arcface = ArcFaceService()
print(f"Available: {arcface.available}")  # Will print False

# This triggers fallback to hashing
from backend.models.schemas import PoseRecord
identity = reid.identify(track_id=1, frame_id=1, pose_available=True, ...)
print(f"Match source: {identity.match_source}")  # Will be "none" or "tms"
```

### Test #3: Verify Activity Hardcoding
```python
from backend.services.activity import STGCNPlusPlusService

activity_svc = STGCNPlusPlusService()

# Create two identical poses
pose1 = PoseRecord(...)  # wrists above head
pose2 = PoseRecord(...)  # wrists above head (identical)

a1 = activity_svc.classify(pose1, 1.0)
a2 = activity_svc.classify(pose2, 2.0)

# Both will have confidence 0.78 (hardcoded)
assert a1.confidence == a2.confidence == 0.78
```

### Test #4: Verify Query Fallback
```python
from backend.services.query_engine import NLQueryEngine

engine = NLQueryEngine(repo, behavior_svc)

# Try unusual query
result = engine.query("What is the capital of France?")
print(result.sql)  # Will be generic "SELECT * FROM events"
# Ollama not available, so falls back
```

---

## Severity Scoring

### 🔴 CRITICAL (P0) - Production Blocker
1. Missing ArcFace weights → Person ID broken
2. Missing OSNet weights → Body matching broken
3. ST-GCN++ fake → Activity recognition invalid
4. SQL injection → Security risk

**Impact:** System cannot be trusted for any mission-critical use

### 🟡 MEDIUM (P1) - Blocking
5. Hardcoded confidence → Misleading scores
6. No evaluation → Can't verify quality
7. Silent fallbacks → User confusion
8. No foreign keys → Data integrity risk

**Impact:** Can use for development/testing but not production

### 🟢 LOW (P2) - Enhancement
9. Timestamp precision → Data quality
10. JSON-in-database → Schema design

**Impact:** Technical debt, doesn't block functionality

---

## Recommendation Matrix

| Issue | Priority | Effort | Impact | Recommended Action |
|-------|----------|--------|--------|-------------------|
| Missing weights | CRITICAL | HIGH | CRITICAL | Download/train |
| ST-GCN++ fake | CRITICAL | HIGH | CRITICAL | Train or document |
| SQL injection | CRITICAL | MEDIUM | CRITICAL | Parameterize |
| Hardcoded confidence | MEDIUM | LOW | MEDIUM | Add validation |
| No evaluation | MEDIUM | HIGH | MEDIUM | Create benchmark |
| Silent fallbacks | MEDIUM | LOW | MEDIUM | Add logging |

---

**For detailed analysis, see AI_SYSTEMS_AUDIT_REPORT.md**
