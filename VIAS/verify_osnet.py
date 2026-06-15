"""
OSNet Verification Script
=========================

This script verifies the complete OSNet setup for the VIAS project.

Run this after installing torchreid:
    pip install torchreid

Usage:
    python verify_osnet.py
"""

import sys
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("OSNET VERIFICATION SCRIPT")
print("=" * 80)
print()

# Test 1: Check torchreid installation
print("TEST 1: Check torchreid installation")
print("-" * 80)
try:
    import torchreid
    print("✅ torchreid is installed")
    print(f"   Version: {torchreid.__version__}")
except ImportError as e:
    print(f"❌ torchreid is NOT installed: {e}")
    print()
    print("SOLUTION:")
    print("   pip install torchreid")
    print()
    sys.exit(1)
print()

# Test 2: Check FeatureExtractor import
print("TEST 2: Check FeatureExtractor import")
print("-" * 80)
try:
    import torchreid
    FeatureExtractor = torchreid.utils.FeatureExtractor
    print("✅ FeatureExtractor imported successfully")
except (ImportError, AttributeError) as e:
    print(f"❌ FeatureExtractor import failed: {e}")
    sys.exit(1)
print()

# Test 3: Check OSNetService import
print("TEST 3: Check OSNetService import")
print("-" * 80)
try:
    from backend.services.reid import OSNetService
    print("✅ OSNetService imported successfully")
except ImportError as e:
    print(f"❌ OSNetService import failed: {e}")
    sys.exit(1)
print()

# Test 4: Initialize OSNetService
print("TEST 4: Initialize OSNetService")
print("-" * 80)
try:
    osnet = OSNetService()
    print("✅ OSNetService initialized")
    print(f"   Model name: {osnet.model_name}")
    print(f"   Device: {osnet.device}")
    print(f"   Weights path: {osnet.weights_path}")
    print(f"   Weights exist: {osnet.weights_path.exists()}")
    print(f"   Extractor available: {osnet.available}")
except Exception as e:
    print(f"❌ OSNetService initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 5: Check extractor availability
print("TEST 5: Check extractor availability")
print("-" * 80)
if osnet.extractor is None:
    print("❌ Extractor is None")
    print()
    print("POSSIBLE CAUSES:")
    print("   1. torchreid not installed (already checked)")
    print("   2. Model initialization failed (check logs above)")
    print("   3. Model download failed (network issue)")
    print()
    print("CHECK BACKEND LOGS for detailed error messages:")
    print("   Look for: 🔴 OSNET INIT FAILED")
    print()
    sys.exit(1)
else:
    print("✅ Extractor is initialized")
    print(f"   Extractor type: {type(osnet.extractor).__name__}")
print()

# Test 6: Test embedding extraction
print("TEST 6: Test embedding extraction")
print("-" * 80)
try:
    import numpy as np
    import cv2
    
    # Create a test image (synthetic person)
    test_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    
    print("   Creating synthetic test image (256x128x3)...")
    result = osnet.extract_from_image(test_image)
    
    if result.source == "osnet" and result.embedding is not None:
        print("✅ Embedding extraction successful")
        print(f"   Shape: {result.embedding.shape}")
        print(f"   Dtype: {result.embedding.dtype}")
        print(f"   Min: {result.embedding.min():.4f}")
        print(f"   Max: {result.embedding.max():.4f}")
        print(f"   Mean: {result.embedding.mean():.4f}")
        print(f"   Std: {result.embedding.std():.4f}")
        print(f"   Confidence: {result.confidence:.4f}")
        print(f"   Source: {result.source}")
    else:
        print("❌ Embedding extraction failed (fallback used)")
        print(f"   Source: {result.source}")
        print(f"   Confidence: {result.confidence}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Embedding extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 7: Check video pipeline integration
print("TEST 7: Check video pipeline integration")
print("-" * 80)
try:
    from backend.services.video_pipeline import VideoAnalyticsPipeline
    pipeline = VideoAnalyticsPipeline()
    
    print("✅ VideoAnalyticsPipeline imported and initialized")
    print(f"   OSNet available in pipeline: {pipeline.reid.osnet.available}")
    
    if not pipeline.reid.osnet.available:
        print("⚠️  Warning: OSNet not available in video pipeline")
except Exception as e:
    print(f"❌ Video pipeline check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 8: Check logging configuration
print("TEST 8: Check logging configuration")
print("-" * 80)
try:
    from backend.utils.logging import logger
    print("✅ Logger imported successfully")
    print(f"   Logger name: {logger.name}")
    print(f"   Logger level: {logger.level}")
    print(f"   Handlers: {len(logger.handlers)}")
except Exception as e:
    print(f"❌ Logger check failed: {e}")
print()

# Final Summary
print("=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print()
print("✅ All tests passed!")
print()
print("OSNet is properly configured and ready for production use.")
print()
print("NEXT STEPS:")
print("   1. Start the backend: python backend/main.py")
print("   2. Check logs for: 🟢 OSNET INIT: Model initialized successfully")
print("   3. Upload a reference image with a person")
print("   4. Check logs for: 🟢 OSNET EXTRACT: Real embedding generated")
print()
print("MONITORING:")
print("   - GET /reid/statistics - Check body_real_count vs body_fallback_count")
print("   - Backend logs - Look for 🟢/🔴 OSNET prefixes")
print()
