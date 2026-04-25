"""
Day 1 contract test for data_loader.py.

Runs in two modes:
  - Without MIMIC-III: validates the public API surface and contracts on a
    synthetic tensor (no wfdb call).
  - With MIMIC-III: pass an env var MIMIC_RECORD pointing to a record path
    (without extension). e.g.
        MIMIC_RECORD=/data/mimic3wdb/30/3000063/3000063 python tests/test_data_loader.py
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import TENSOR_LENGTH, validate_shape, to_gpu
from config.stream_config import N_CHANNELS


def test_validate_shape_synthetic() -> None:
    t = torch.zeros((N_CHANNELS, TENSOR_LENGTH), dtype=torch.float32)
    validate_shape(t)
    b = torch.zeros((4, N_CHANNELS, TENSOR_LENGTH), dtype=torch.float32)
    validate_shape(b)
    print(f"[OK] validate_shape accepts (12, 5000) and (B, 12, 5000) float32")


def test_validate_rejects_bad_shape() -> None:
    t = torch.zeros((11, TENSOR_LENGTH), dtype=torch.float32)
    try:
        validate_shape(t)
    except AssertionError:
        print("[OK] validate_shape rejects 11 channels")
        return
    raise SystemExit("validate_shape failed to reject bad channel count")


def test_validate_rejects_bad_dtype() -> None:
    t = torch.zeros((N_CHANNELS, TENSOR_LENGTH), dtype=torch.float64)
    try:
        validate_shape(t)
    except AssertionError:
        print("[OK] validate_shape rejects float64")
        return
    raise SystemExit("validate_shape failed to reject bad dtype")


def test_to_gpu_noop_on_cpu() -> None:
    t = torch.zeros((N_CHANNELS, TENSOR_LENGTH), dtype=torch.float32)
    out = to_gpu(t)
    assert out.shape == t.shape
    print(f"[OK] to_gpu returns tensor with same shape "
          f"(cuda available: {torch.cuda.is_available()})")


def test_load_real_record_if_provided() -> None:
    record = os.environ.get("MIMIC_RECORD")
    if not record:
        print("[SKIP] MIMIC_RECORD not set — skipping real-record test")
        return
    from data_loader import load_mimic_record
    t = load_mimic_record(record)
    validate_shape(t)
    print(f"[OK] loaded {record}: shape={tuple(t.shape)} dtype={t.dtype}")


if __name__ == "__main__":
    test_validate_shape_synthetic()
    test_validate_rejects_bad_shape()
    test_validate_rejects_bad_dtype()
    test_to_gpu_noop_on_cpu()
    test_load_real_record_if_provided()
    print("\n[OK] data_loader Day 1 contracts validated.")
