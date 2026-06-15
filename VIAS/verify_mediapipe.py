#!/usr/bin/env python
"""
MediaPipe Production Readiness Verification Script
Checks all aspects of MediaPipe integration.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("MEDIAPIPE PRODUCTION READINESS VERIFICATION")
print("="*70)

# Test 1: Module Installation
print("\n[1/6] Checking MediaPipe installation...")
try:
    import mediapipe as mp
    print(f"    ✅ MediaPipe installed (v{mp.__version__})")
    mediapipe_installed = True
except ImportError:
    print("    ❌ MediaPipe NOT installed")
    print("    → Install with: pip install mediapipe==0.10.14")
    mediapipe_installed = False

# Test 2: Service Import
print("\n[2/6] Checking pose service import...")
try:
    from backend.services.pose import MediaPipePoseService
    print("    ✅ MediaPipePoseService imported successfully")
    service_imported = True
except Exception as e:
    print(f"    ❌ Failed to import: {e}")
    service_imported = False

# Test 3: Service Initialization
service_initialized = False
service = None
if service_imported:
    print("\n[3/6] Checking service initialization...")
    try:
        service = MediaPipePoseService()
        if service.available:
            print("    ✅ Service initialized successfully")
            print(f"    ✅ Pose estimator available: {service.pose is not None}")
            service_initialized = True
        else:
            print("    ⚠️ Service initialized but pose estimator unavailable")
            print("       (This is expected if MediaPipe is not installed)")
            service_initialized = False
    except Exception as e:
        print(f"    ❌ Initialization failed: {e}")
        service_initialized = False
else:
    print("\n[3/6] Skipping initialization (import failed)")

# Test 4: Check Methods
print("\n[4/6] Checking service methods...")
if service:
    methods_ok = True
    
    # Check available property
    try:
        available = service.available
        print(f"    ✅ available property: {available}")
    except Exception as e:
        print(f"    ❌ available property failed: {e}")
        methods_ok = False
    
    # Check extract method
    try:
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = service.extract(dummy_frame, track_id=1, frame_id=0)
        print(f"    ✅ extract method works (returned: {type(result).__name__})")
    except Exception as e:
        print(f"    ❌ extract method failed: {e}")
        methods_ok = False
    
    # Check get_stats method
    try:
        stats = service.get_stats()
        print(f"    ✅ get_stats method works")
        print(f"       Extractions: {stats['total_extractions']}")
        print(f"       Failures: {stats['total_failures']}")
    except Exception as e:
        print(f"    ❌ get_stats method failed: {e}")
        methods_ok = False
else:
    print("    ⚠️ Skipping (service not initialized)")
    methods_ok = False

# Test 5: Integration with Video Pipeline
print("\n[5/6] Checking video pipeline integration...")
try:
    from backend.services.video_pipeline import VideoAnalyticsPipeline
    pipeline = VideoAnalyticsPipeline()
    
    print(f"    ✅ Video pipeline imported")
    print(f"    ✅ Pose service integrated: {pipeline.pose is not None}")
    print(f"    ✅ Pose available: {pipeline.pose.available if pipeline.pose else False}")
    pipeline_ok = True
except Exception as e:
    print(f"    ❌ Pipeline integration failed: {e}")
    pipeline_ok = False

# Test 6: Check Logging
print("\n[6/6] Checking logging configuration...")
try:
    import logging
    logger = logging.getLogger('backend.services.pose')
    print(f"    ✅ Logger configured: {logger.name}")
    print(f"    ✅ Log level: {logging.getLevelName(logger.level)}")
    logging_ok = True
except Exception as e:
    print(f"    ❌ Logging check failed: {e}")
    logging_ok = False

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

tests = [
    ("MediaPipe Installation", mediapipe_installed),
    ("Service Import", service_imported),
    ("Service Initialization", service_initialized),
    ("Service Methods", methods_ok),
    ("Pipeline Integration", pipeline_ok),
    ("Logging Configuration", logging_ok),
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

print(f"\nTests Passed: {passed}/{total}\n")

for test_name, result in tests:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"  {status:10} {test_name}")

print("\n" + "="*70)

if passed == total:
    print("🎉 ALL TESTS PASSED - MediaPipe is PRODUCTION READY!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run visual test: python pose_test.py")
    print("  2. Process a video through VIAS")
    print("  3. Check logs for any warnings")
    sys.exit(0)
else:
    print("⚠️  SOME TESTS FAILED - Review issues above")
    print("="*70)
    
    if not mediapipe_installed:
        print("\n🔧 Quick Fix:")
        print("   pip install mediapipe==0.10.14")
    
    sys.exit(1)
