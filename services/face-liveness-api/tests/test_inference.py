"""Real-model integration tests.

These run the *actual* EfficientNet-B2 + BiLSTM pipeline on frames extracted
from the bundled Oulu sample videos and are marked ``real``. They are the only
tests that prove the shipped model works end-to-end.
"""

from __future__ import annotations

import struct

import cv2
import numpy as np
import pytest

SEQ = struct.Struct(">I")


def _extract_video_frames(path: str, n: int = 150, jpeg_quality: int = 92):
    """Read n frames from a video and encode them as JPEG bytes (as a browser would)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    if total <= n:
        indices = list(range(total))
        while len(indices) < n:
            indices.append(indices[-1])
    else:
        indices = np.linspace(0, total - 1, n, dtype=int).tolist()
    out = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if success:
            out.append(buf.tobytes())
    cap.release()
    return out


@pytest.mark.real
@pytest.mark.slow
def test_real_video_scores(real_pipeline, videos_dir):
    """The real Oulu samples must produce the expected decisions."""
    import os
    cases = [
        ("1_1_14_1_real_oulu.avi", "live"),
        ("1_1_14_3_printer2_oulu.avi", "spoof"),
    ]
    for filename, expected in cases:
        path = os.path.join(videos_dir, filename)
        frames = _extract_video_frames(path)
        assert len(frames) == 150, f"{filename}: expected 150 frames, got {len(frames)}"
        result = real_pipeline.predict(frames)
        assert result.prediction == expected, (
            f"{filename}: expected {expected}, got {result.prediction} "
            f"(score={result.anomaly_score:.4f}, thr={result.threshold})"
        )
        assert result.frames_processed == 150
        assert 0.0 <= result.anomaly_score <= 1.0
        assert result.threshold == 0.5
        assert result.inference_ms > 0


@pytest.mark.real
@pytest.mark.slow
def test_full_websocket_flow_real_model(fastapi_client, videos_dir, models_dir):
    """Complete flow over WebSocket against the real model."""
    import os
    # Restore the real pipeline (protocol tests may have installed a stub).
    from app import main
    from app.ml import LivenessPipeline, PipelineConfig
    if not isinstance(main.PIPELINE, LivenessPipeline):
        main.PIPELINE = LivenessPipeline(
            models_dir=models_dir, device="cpu", config=PipelineConfig(batch_size=8),
        )
        main.PIPELINE.load()

    frames = _extract_video_frames(os.path.join(videos_dir, "1_1_14_1_real_oulu.avi"))
    session = fastapi_client.post("/sessions").json()
    sid, token = session["session_id"], session["ws_token"]

    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        for i, payload in enumerate(frames):
            ws.send_bytes(SEQ.pack(i) + payload)
        result = None
        for _ in range(len(frames) + 10):
            msg = ws.receive_json()
            if msg["type"] in ("result", "error"):
                result = msg
                break
    assert result is not None
    assert result["type"] == "result", f"expected result, got {result}"
    assert result["prediction"] == "live"
    assert result["frames_processed"] == 150
