# ArcFace Trace Locations

## Summary
This document maps all locations where `deterministic_embedding()` fallback is used in the VIAS project.

---

## Files Using `deterministic_embedding()`

### 1. **backend/services/reid.py**
Main Re-identification service with ArcFace and OSNet integration.

#### Import (Line 18)
```python
from backend.services.embedding_utils import deterministic_embedding
```

#### Location 1: Register Reference - Face Fallback (Line 157)
```python
if face_result.embedding is None:
    logger.info(f"🔴 REGISTER: Face embedding is None, using deterministic fallback")
    face_result = EmbeddingResult(deterministic_embedding(f"face-ref:{person_id}:{image_path}", 512), 0.0)
```
**Trigger**: When ArcFace fails to detect/extract face from reference image.

#### Location 2: Register Reference - Body Fallback (Line 163)
```python
if body_result.embedding is None:
    logger.info(f"🔴 REGISTER: Body embedding is None, using deterministic fallback")
    body_result = EmbeddingResult(deterministic_embedding(f"body-ref:{person_id}:{image_path}", 512), 0.0)
```
**Trigger**: When OSNet fails to extract body embedding from reference image.

#### Location 3: Video Processing - Face Fallback (Line 205)
```python
if face_embedding is None:
    logger.info(f"🔴 EXTRACT: Face embedding is None, using deterministic fallback")
    face_embedding = deterministic_embedding(f"face:{track_id}:{frame_id}", 512)
```
**Trigger**: When ArcFace fails to detect/extract face from video frame.

#### Location 4: Video Processing - Body Fallback (Line 210)
```python
if body_embedding is None:
    logger.info(f"🔴 EXTRACT: Body embedding is None, using deterministic fallback")
    body_embedding = deterministic_embedding(f"body:{track_id}:{frame_id}", 512)
```
**Trigger**: When OSNet fails to extract body embedding from video frame.

---

### 2. **backend/api/routes.py**
API endpoint handlers.

#### Import (Line 16)
```python
from backend.services.embedding_utils import deterministic_embedding
```

#### Location 5: Search Person Endpoint (Line 106)
```python
if request.person_id:
    query = deterministic_embedding(f"face-ref:{request.person_id}:lookup", 512)
    results = get_store().search("face_embeddings", query, top_k=request.top_k)
```
**Trigger**: Always uses fallback for person search by ID (NOT using real ArcFace).
**Issue**: This endpoint NEVER uses real face embeddings, even if ArcFace is available.

---

### 3. **backend/services/behavior.py**
Behavior pattern discovery service.

#### Import (Line 8)
```python
from backend.services.embedding_utils import deterministic_embedding
```

#### Location 6: Behavior Pattern Storage (Line 21)
```python
embedding = deterministic_embedding(summary, 128).reshape(1, -1)
```
**Trigger**: Always uses fallback (behavior patterns, not face/body).
**Note**: This is intentional - behavior embeddings use text summaries, not images.

#### Location 7: Behavior Search (Line 30)
```python
embedding = deterministic_embedding(query, 128)
```
**Trigger**: Always uses fallback (text query for behavior search).
**Note**: This is intentional - searching behavior patterns by text description.

---

### 4. **backend/services/embedding_utils.py**
Utility module containing the fallback function.

#### Function Definition (Line 8)
```python
def deterministic_embedding(seed_text: str, dimension: int) -> np.ndarray:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    values = np.frombuffer((digest * ((dimension // len(digest)) + 1))[:dimension], dtype=np.uint8)
    vector = values.astype(np.float32)
    vector = (vector - vector.mean()) / (vector.std() + 1e-6)
    return vector
```
**How it works**: 
1. SHA-256 hash of input text
2. Convert hash bytes to uint8 array
3. Repeat to fill dimension
4. Convert to float32 and normalize (zero mean, unit std)

---

## Execution Flow Trace

### When ArcFace is Working (Real Embeddings)
```
1. ArcFaceService.__init__()
   ├── FaceAnalysis available ✅
   ├── buffalo_l model loaded ✅
   └── self.app = <FaceAnalysis object>

2. extract_from_image(image)
   ├── self.app.get(image) → faces detected ✅
   ├── best_face.embedding extracted ✅
   └── return EmbeddingResult(real_embedding, confidence)

3. register_reference() / _extract_embeddings()
   ├── face_result.embedding is NOT None ✅
   ├── Skip deterministic_embedding() ✅
   └── Use real embedding
```

### When ArcFace is NOT Working (Fallback)
```
1. ArcFaceService.__init__()
   ├── FaceAnalysis is None ❌
   └── self.app = None

2. extract_from_image(image)
   ├── self.app is None ❌
   └── return EmbeddingResult(None, 0.0)

3. register_reference() / _extract_embeddings()
   ├── face_result.embedding is None ❌
   ├── Call deterministic_embedding() ❌
   └── Use SHA-256 hash fallback
```

---

## Logging Added

All logging uses emoji prefixes for easy identification:

- 🟢 **Green**: Success (real AI embeddings)
- 🔴 **Red**: Fallback (deterministic_embedding used)
- 🟡 **Yellow**: In-progress operation
- 🔵 **Blue**: Information

### Example Log Output (ArcFace Working)
```
🟢 ARCFACE INIT: FaceAnalysis available, loading model...
🟢 ARCFACE INIT: Model loaded successfully (buffalo_l)
🟡 ARCFACE EXTRACT: Running face detection...
🟢 ARCFACE EXTRACT: 1 face(s) detected
🟢 ARCFACE EXTRACT: Real embedding generated (dim=512, conf=0.9234)
🔵 REGISTER REFERENCE: person_id=john_doe, image=uploads/john.jpg
🟢 REGISTER: Real face embedding generated
```

### Example Log Output (ArcFace NOT Working)
```
🔴 ARCFACE INIT: FaceAnalysis is None (insightface not installed)
🔴 ARCFACE EXTRACT: app is None, returning None embedding
🔵 REGISTER REFERENCE: person_id=john_doe, image=uploads/john.jpg
🔴 REGISTER: Face embedding is None, using deterministic fallback
```

---

## Test Scripts

### test_arcface.py
Standalone verification script that:
1. Initializes ArcFaceService
2. Loads a test image
3. Extracts embedding
4. Analyzes whether it's real or fallback
5. Reports verdict with detailed metrics

**Usage**:
```bash
python test_arcface.py
```

---

## Critical Issue: Search Person Endpoint

**File**: `backend/api/routes.py` (Line 106)

**Problem**: The `/search-person` endpoint ALWAYS uses `deterministic_embedding()`, even when ArcFace is available.

**Current Code**:
```python
if request.person_id:
    query = deterministic_embedding(f"face-ref:{request.person_id}:lookup", 512)
    results = get_store().search("face_embeddings", query, top_k=request.top_k)
```

**Impact**: 
- Person search will NOT work correctly even if ArcFace is installed
- Hash-based query will never match real face embeddings
- This endpoint needs to be fixed separately

---

## Summary Table

| Location | File | Line | Type | Always Fallback? |
|----------|------|------|------|------------------|
| 1 | reid.py | 157 | Face (Reference) | No - only if ArcFace fails |
| 2 | reid.py | 163 | Body (Reference) | No - only if OSNet fails |
| 3 | reid.py | 205 | Face (Video) | No - only if ArcFace fails |
| 4 | reid.py | 210 | Body (Video) | No - only if OSNet fails |
| 5 | routes.py | 106 | Search Query | **YES - BROKEN** |
| 6 | behavior.py | 21 | Behavior Pattern | YES - intentional |
| 7 | behavior.py | 30 | Behavior Query | YES - intentional |

---

## Next Steps

1. **Run test_arcface.py** to verify ArcFace status
2. **Check logs** when running backend to see initialization
3. **Test reference registration** and watch for fallback logging
4. **Fix routes.py** search endpoint (separate task)
5. **Install insightface** if not already installed:
   ```bash
   pip install insightface onnxruntime
   ```
