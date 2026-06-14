from __future__ import annotations

import json
import time

from backend.services.video_pipeline import VideoAnalyticsPipeline


def benchmark() -> None:
    start = time.perf_counter()
    pipeline = VideoAnalyticsPipeline()
    elapsed = time.perf_counter() - start
    print(json.dumps({"initialization_seconds": elapsed, "pipeline": pipeline.__class__.__name__}, indent=2))


if __name__ == "__main__":
    benchmark()
