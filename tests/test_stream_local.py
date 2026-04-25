"""
Day 1 validation harness: simulate_stream.py produces (12, 25) float32 frames at 20fps.

Run from project root:
    python tests/test_stream_local.py
"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulate_stream import ECG_QUEUE, stream_generator, get_drop_count
from config.stream_config import (
    CHUNK_SIZE,
    N_CHANNELS,
    FRAME_INTERVAL,
    QUEUE_MAXSIZE,
)


async def main() -> int:
    print(f"Producer cadence    : {1 / FRAME_INTERVAL:.1f} fps")
    print(f"Queue maxsize       : {QUEUE_MAXSIZE} frames "
          f"({QUEUE_MAXSIZE * FRAME_INTERVAL:.2f}s buffer)")
    print()

    producer = asyncio.create_task(stream_generator())

    n_target = 40
    t0 = time.perf_counter()
    frames = []
    for _ in range(n_target):
        frames.append(await ECG_QUEUE.get())
    elapsed = time.perf_counter() - t0

    producer.cancel()
    try:
        await producer
    except asyncio.CancelledError:
        pass

    sample = frames[0]
    expected_bytes = N_CHANNELS * CHUNK_SIZE * 4

    print(f"Frames captured     : {len(frames)}")
    print(f"Elapsed             : {elapsed:.2f}s "
          f"(expected ~{n_target * FRAME_INTERVAL:.2f}s)")
    print(f"Effective fps       : {len(frames) / elapsed:.2f}")
    print(f"Frame shape         : {sample.shape} "
          f"(expected ({N_CHANNELS}, {CHUNK_SIZE}))")
    print(f"Frame dtype         : {sample.dtype} (expected float32)")
    print(f"Bytes per frame     : {sample.nbytes} (expected {expected_bytes})")
    print(f"Frames dropped      : {get_drop_count()} (expected 0 in healthy run)")

    assert sample.shape == (N_CHANNELS, CHUNK_SIZE), "shape contract violated"
    assert sample.dtype == np.float32,               "dtype contract violated"
    assert sample.nbytes == expected_bytes,          "byte size contract violated"

    print("\n[OK] Day 1 stream contracts validated.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
