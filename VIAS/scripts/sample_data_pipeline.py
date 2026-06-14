from __future__ import annotations

import json
from pathlib import Path

from backend.services.video_pipeline import VideoAnalyticsPipeline


def main(video_path: str) -> None:
    pipeline = VideoAnalyticsPipeline()
    result = pipeline.process_video(Path(video_path))
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/sample_data_pipeline.py <video_path>")
    main(sys.argv[1])
