try:
    from insightface.app import FaceAnalysis
except ImportError as exc:
    raise ImportError(
        "insightface is not installed or cannot be resolved. "
        "Install it with `pip install insightface` and ensure your environment is correct."
    ) from exc

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

print("ArcFace loaded successfully")