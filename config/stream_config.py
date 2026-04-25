"""
stream_config.py — Constantes canónicas del protocolo de stream ECG.
Fuente de verdad única compartida por todos los módulos Python.
Mirror TypeScript: frontend/lib/streamConfig.ts  (mantener sincronizado).

Protocolo acordado Agente 1 ↔ Agente 3 (2026-04-25):
  - Frames ECG   : WebSocket BINARY opcode, float32 LE, C-order, shape (12,25)
  - Control/JSON : WebSocket TEXT opcode, JSON UTF-8
"""

CHUNK_SIZE     = 25      # samples per WebSocket binary frame per channel
SAMPLE_RATE    = 500     # Hz (MIMIC-III standard)
N_CHANNELS     = 12      # ECG leads
FPS            = 20      # frames per second to browser
FRAME_INTERVAL = 1.0 / FPS   # 0.050 s between frames
QUEUE_MAXSIZE  = 40      # frames in asyncio.Queue = 2 s of signal at 20fps
                         # Timeout de reconstrucción (Día 3) DEBE ser > 2 s

# Derivados — NO redefinir en otros módulos, importar desde aquí
FRAME_BYTES    = N_CHANNELS * CHUNK_SIZE * 4   # 1200 bytes (float32 = 4 B)
DTYPE          = "float32"

LEADS_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]
