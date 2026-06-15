# VIAS AI AUDIT - EXECUTIVE SUMMARY

**Status: 🔴 CRITICAL ISSUES IDENTIFIED - NOT PRODUCTION READY**

---

## Quick Findings

### Components Breakdown

| Component | Status | Real? | Working? |
|-----------|--------|-------|----------|
| YOLOv10 Detection | ✅ PASS | YES | YES |
| ByteTrack Tracking | ✅ PASS | YES | YES |
| MediaPipe Pose | ⚠️ OPTIONAL | YES | Maybe |
| ArcFace Face Recognition | ❌ FAIL | YES | NO (weights missing) |
| OSNet Body Matching | ❌ FAIL | YES | NO (weights missing) |
| ST-GCN++ Activity | ❌ FAKE | NO | Runs but meaningless |
| TMS Motion | ✅ WORKS | NO (hand-crafted) | YES |
| Ollama Query Engine | ⚠️ OPTIONAL | YES | Has fallbacks |

---

## Critical Findings

### 🔴 Finding #1: Person Recognition is Broken
- **What claimed:** Multi-modal face/body recognition using ArcFace + OSNet
- **What's happening:** Falls back to deterministic hashing
- **Impact:** Person matching = random hash matching, not real recognition
- **Fix:** Obtain and load ArcFace + OSNet weights

### 🔴 Finding #2: Activity Recognition is Fake
- **What claimed:** Uses "ST-GCN++" deep learning model
- **What's happening:** Simple if-else rules based on keypoint coordinates
- **Confidence:** Hardcoded (0.69-0.81) regardless of quality
- **Impact:** Activity labels are guesses, not AI-based
- **Fix:** Either train real ST-GCN++ or rename to "RuleBasedActivityClassifier"

### 🔴 Finding #3: Query Engine Can Hallucinate
- **What claimed:** LLM translates English to SQL
- **What's happening:** Ollama likely not installed, uses hardcoded SQL for 2 patterns only
- **Impact:** Most queries get generic results or wrong SQL
- **Security Risk:** SQL injection vulnerability
- **Fix:** Validate SQL output, use parameterized queries

### 🟡 Finding #4: Confidence Scores Misleading
- **Issue:** Users trust 95% confidence in face match that's really a hash comparison
- **Impact:** False sense of accuracy
- **Fix:** Add validation, show which model generated each score

### 🟡 Finding #5: No Evaluation Framework
- **Missing:** Accuracy, Precision, Recall, F1-Score, MOTA, ID switches
- **Impact:** No way to know if system works
- **Fix:** Create validation dataset, track metrics

---

## How to Verify (Simple Tests)

### Test #1: Missing Model Weights
```bash
ls -la models/arcface/      # Should show .onnx file - MISSING
ls -la models/osnet/        # Should show .pth file - MISSING
```

### Test #2: Hashing Instead of Real Recognition
```python
from backend.services.embedding_utils import deterministic_embedding

# Same input always gives same hash
e1 = deterministic_embedding("face:1:0", 512)
e2 = deterministic_embedding("face:1:0", 512)
print(e1 == e2)  # TRUE - proves it's hashing!
```

### Test #3: Hardcoded Activity Confidence
```python
# Activity service always returns same confidence for same activity
activity1.classify(pose, timestamp).confidence  # 0.78
activity2.classify(pose, timestamp).confidence  # 0.78 (always!)
```

### Test #4: Query Fallback
```python
# Try a query about weather (no hardcoded pattern)
result = engine.query("What is the weather?")
print(result.sql)  # Generic SELECT * FROM events
```

---

## Component Trustworthiness

| What You Can Trust | What You Cannot Trust |
|-------------------|----------------------|
| ✅ Person detection (YOLOv10) | ❌ Face identification |
| ✅ Person tracking (ByteTrack) | ❌ Activity recognition |
| ✅ Pose estimation (MediaPipe) | ❌ Body matching |
| ✅ TMS features | ⚠️ Query results |

---

## What's Actually Working?

The core detection and tracking pipeline is solid:

1. **Frame-by-frame person detection** ✅ (YOLOv10)
2. **Consistent tracking across frames** ✅ (ByteTrack)
3. **Skeleton pose extraction** ✅ (MediaPipe)
4. **Manual feature extraction** ✅ (TMS)

What's **NOT working:**
5. **Re-identifying people** ❌ (Hash fallback)
6. **Recognizing activities** ❌ (Rule-based only)
7. **Querying results** ⚠️ (Limited fallbacks)

---

## Production Readiness: 🔴 NOT READY

### Before Deploying (MUST DO)

- [ ] **Critical:** Load ArcFace + OSNet model weights
- [ ] **Critical:** Fix SQL injection vulnerability
- [ ] **Critical:** Replace activity recognition with real model or document as rule-based
- [ ] **High:** Add confidence score validation
- [ ] **High:** Create evaluation framework
- [ ] **High:** Remove misleading claims from documentation

### Current State

- ✅ Can detect and track people
- ❌ **Cannot identify people** (hashing only)
- ❌ **Cannot recognize activities** (rules only)
- ⚠️ **Can query but results unreliable** (fallbacks)

---

## Grade: F (Failing)

**System is not trustworthy for production:**
- Only 2 of 8 AI components actually work
- Person identification completely broken
- Activity recognition is rule-based masquerading as AI
- No evaluation/validation framework
- Misleading confidence scores
- Security vulnerability (SQL injection)

**Verdict:** High-quality detection/tracking, but unreliable higher-level AI features.

---

## Next Steps

### Immediate (1 week)
1. Get missing model weights
2. Fix SQL injection
3. Test all components

### Short term (2 weeks)
4. Replace fake ST-GCN++ with real model or document as rules
5. Add evaluation metrics
6. Update documentation

### Long term (1 month)
7. Fine-tune models
8. Add continuous monitoring
9. Implement A/B testing

---

**For full details, see: AI_SYSTEMS_AUDIT_REPORT.md**
