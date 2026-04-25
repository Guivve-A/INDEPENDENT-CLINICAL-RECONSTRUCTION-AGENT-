CHUNK_SIZE     = 25      # samples per WebSocket binary frame
SAMPLE_RATE    = 500     # Hz (MIMIC-III standard)
N_CHANNELS     = 12      # ECG leads
FRAME_INTERVAL = 0.050   # seconds between frames (1 / 20fps)
QUEUE_MAXSIZE  = 40      # frames = 2 seconds of buffer at 20fps

LEADS_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]
