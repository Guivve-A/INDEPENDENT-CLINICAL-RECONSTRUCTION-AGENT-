"""
MIMIC-III WFDB → tensor (12, 5000) float32 @ 500Hz.

Public surface used by Agent 2 (training) and Agent 3 (validation harness):
    load_mimic_record(path)          → tensor (12, 5000)
    load_batch(paths)                → tensor (B, 12, 5000)
    to_gpu(tensor)                   → moves to ROCm GPU if available
    validate_shape(tensor)           → hard contract assertion
"""
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

try:
    import wfdb
except ImportError:
    wfdb = None

from config.stream_config import N_CHANNELS, SAMPLE_RATE

TENSOR_LENGTH = 5000  # 10 seconds at 500Hz — canonical (batch, 12, 5000)

WFDB_LEAD_NAMES_LOWER = [
    "i", "ii", "iii", "avr", "avl", "avf",
    "v1", "v2", "v3", "v4", "v5", "v6",
]


def load_mimic_record(record_path: str,
                      length: int = TENSOR_LENGTH,
                      target_fs: int = SAMPLE_RATE) -> torch.Tensor:
    """Load a MIMIC-III WFDB record, resample to target_fs, return (12, length) float32.

    Multi-segment records are handled transparently by wfdb.rdrecord.
    Missing leads are zero-filled. Header PHI is never returned.
    """
    if wfdb is None:
        raise ImportError("wfdb-python not installed. pip install wfdb")

    record = wfdb.rdrecord(record_path)
    fs = int(record.fs)
    sigs = record.p_signal.astype(np.float32)              # (n_samples, n_channels)
    sig_names = [s.lower().strip() for s in record.sig_name]

    if fs != target_fs:
        x = torch.from_numpy(sigs.T).unsqueeze(0)          # (1, n_ch, n_samp)
        new_len = int(sigs.shape[0] * target_fs / fs)
        x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
        sigs = x.squeeze(0).T.numpy()                      # (n_samp', n_ch)

    canon = np.zeros((N_CHANNELS, length), dtype=np.float32)
    n_avail = min(sigs.shape[0], length)
    for i, lead in enumerate(WFDB_LEAD_NAMES_LOWER):
        if lead in sig_names:
            idx = sig_names.index(lead)
            canon[i, :n_avail] = sigs[:n_avail, idx]

    return torch.from_numpy(canon)


def load_batch(record_paths: List[str], length: int = TENSOR_LENGTH) -> torch.Tensor:
    """Stack multiple records → tensor (B, 12, length) float32."""
    return torch.stack([load_mimic_record(p, length) for p in record_paths], dim=0)


def to_gpu(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to ROCm GPU (torch.cuda is the ROCm interface) if available."""
    if torch.cuda.is_available():
        return tensor.to("cuda", non_blocking=True)
    return tensor


def validate_shape(tensor: torch.Tensor, length: int = TENSOR_LENGTH) -> None:
    """Hard contract validation. Used by Agent 2 and Agent 3 sanity checks."""
    assert tensor.dim() in (2, 3), f"Expected 2D or 3D tensor, got {tensor.dim()}D"
    if tensor.dim() == 2:
        c, t = tensor.shape
    else:
        _, c, t = tensor.shape
    assert c == N_CHANNELS, f"Expected {N_CHANNELS} channels, got {c}"
    assert t == length, f"Expected {length} samples, got {t}"
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
