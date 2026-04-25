import asyncio
import numpy as np

from config.stream_config import (
    CHUNK_SIZE,
    SAMPLE_RATE,
    N_CHANNELS,
    FRAME_INTERVAL,
    QUEUE_MAXSIZE,
)

# 40 frames × 50ms = 2 seconds of buffer — covers normal network jitter.
# Agent 3 uses this as the reference for maximum end-to-end latency.
ECG_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

_drop_count: int = 0


# ---------------------------------------------------------------------------
# Synthetic ECG generation
# ---------------------------------------------------------------------------

_HEART_RATE   = 75                                     # bpm
_BEAT_PERIOD  = int(SAMPLE_RATE * 60 / _HEART_RATE)   # samples per beat (~400)

# Lead amplitude scaling — simplified anatomical projection coefficients
# Order matches LEADS_ORDER: [I, II, III, aVR, aVL, aVF, V1..V6]
_LEAD_GAINS = np.array(
    [1.0, 1.3, 0.3, -1.0, 0.5, 0.8, 0.4, 0.6, 0.9, 1.1, 1.0, 0.7],
    dtype=np.float32,
)


def _build_qrs_template(length: int) -> np.ndarray:
    t = np.linspace(-0.5, 0.5, length)
    p  =  0.15 * np.exp(-0.5 * ((t + 0.30) / 0.050) ** 2)
    q  = -0.10 * np.exp(-0.5 * ((t + 0.05) / 0.015) ** 2)
    r  =  1.00 * np.exp(-0.5 * ((t       ) / 0.020) ** 2)
    s  = -0.20 * np.exp(-0.5 * ((t - 0.04) / 0.015) ** 2)
    tw =  0.30 * np.exp(-0.5 * ((t - 0.25) / 0.080) ** 2)
    return (p + q + r + s + tw).astype(np.float32)


_QRS_TEMPLATE = _build_qrs_template(length=min(100, _BEAT_PERIOD))
_sample_cursor: int = 0


def generate_next_frame() -> np.ndarray:
    """Return ndarray (12, CHUNK_SIZE) float32 with synthetic ECG morphology."""
    global _sample_cursor

    amplitudes = np.zeros(CHUNK_SIZE, dtype=np.float32)
    for i in range(CHUNK_SIZE):
        pos = (_sample_cursor + i) % _BEAT_PERIOD
        if pos < len(_QRS_TEMPLATE):
            amplitudes[i] = _QRS_TEMPLATE[pos]

    noise = np.random.normal(0.0, 0.02, (N_CHANNELS, CHUNK_SIZE)).astype(np.float32)
    frame = _LEAD_GAINS[:, np.newaxis] * amplitudes[np.newaxis, :] + noise

    _sample_cursor = (_sample_cursor + CHUNK_SIZE) % _BEAT_PERIOD
    return frame  # shape (12, 25)


# ---------------------------------------------------------------------------
# Async generator — runs as background task inside the FastAPI process
# ---------------------------------------------------------------------------

async def stream_generator() -> None:
    """Push ECG frames into ECG_QUEUE at 20fps.

    Drop strategy: discard the INCOMING frame when the queue is full.
    put_nowait() raises QueueFull → we catch and drop.
    For a live signal, losing a new frame is correct — we never want to
    replay stale data to a slow consumer.
    """
    global _drop_count
    while True:
        frame = generate_next_frame()
        try:
            ECG_QUEUE.put_nowait(frame)
        except asyncio.QueueFull:
            _drop_count += 1
        await asyncio.sleep(FRAME_INTERVAL)


def get_drop_count() -> int:
    """Cumulative dropped-frame count — exposed for telemetry endpoint."""
    return _drop_count
