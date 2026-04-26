"""
main.py — Servidor FastAPI + WebSocket para streaming de ECG.

Protocolo acordado con Agente 1 (Día 1):
    - Formato  : binary float32, row-major
    - Shape    : (12, 25) por frame
    - Bytes    : 1200 por frame (12 × 25 × 4)
    - FPS      : 20  (50 ms entre frames)
    - Endpoint : ws://<host>:8000/stream

Estado Día 1: usa SyntheticEmitter como fuente de datos.
Día 3: SyntheticEmitter se reemplaza por el stream real de Agente 1 +
       invocación del modelo de Agente 2 cuando la FSM lo indique.
"""

import asyncio
import sys
import os
import struct
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# Permite importar desde la raíz del proyecto tanto en dev como en Docker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from synthetic_emitter import SyntheticEmitter
from config.stream_config import (
    N_CHANNELS, CHUNK_SIZE, FPS, FRAME_INTERVAL, FRAME_BYTES, QUEUE_MAXSIZE
)

# ---------------------------------------------------------------------------
# Constantes de protocolo — importadas desde config/stream_config.py
# NO redefinir aquí. Si cambia el protocolo, cambiar en stream_config.py.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ECG Reconstruction Agent — Backend",
    description="Servidor de streaming ECG con WebSocket binario (float32).",
    version="0.1.0-day1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://0.0.0.0:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints HTTP
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Verificación de estado del servidor y parámetros del protocolo."""
    return {
        "status": "ok",
        "agent_state": "MONITORING",
        "protocol": {
            "encoding": "binary_float32_row_major",
            "frame_shape": [N_CHANNELS, CHUNK_SIZE],
            "frame_bytes": FRAME_BYTES,
            "fps": FPS,
            "sample_rate_hz": 500,
        },
    }


# ---------------------------------------------------------------------------
# WebSocket /stream
# ---------------------------------------------------------------------------

@app.websocket("/stream")
async def stream_ecg(websocket: WebSocket) -> None:
    """
    Emite frames ECG binarios al cliente a 20 FPS.

    Cada frame = 1200 bytes:
        bytes [0..99]    → canal 0  (25 muestras float32)
        bytes [100..199] → canal 1
        ...
        bytes [1100..1199] → canal 11

    Backpressure: `asyncio.sleep` cede el event loop entre frames.
    Si el cliente no consume a tiempo, send_bytes lanzará excepción y
    el handler termina limpiamente — nunca se acumula cola ilimitada.
    """
    await websocket.accept()
    client = websocket.client
    print(f"[stream] cliente conectado: {client}")

    emitter = SyntheticEmitter(
        n_channels=N_CHANNELS,
        sample_rate=500,
        chunk_size=CHUNK_SIZE,
        heart_rate_bpm=72.0,
        seed=42,
    )

    try:
        while True:
            # Genera frame (12, 25) float32
            frame: np.ndarray = emitter.next_frame()

            # Serializa como bytes raw (row-major, little-endian float32)
            payload: bytes = frame.tobytes()
            assert len(payload) == FRAME_BYTES, (
                f"Frame size mismatch: {len(payload)} != {FRAME_BYTES}"
            )
            # --- DIAGNÓSTICO CAPA 2: INYECTADO AQUÍ ---
            # Solo imprimimos 1 de cada 20 frames (aprox 1 vez por segundo) para no inundar la consola
            if np.random.rand() < 0.05:
                 unpacked_floats = struct.unpack('<2f', payload[:8])
                 print(f"[WS-SEND] bytes len: {len(payload)}, primeros 8 bytes (2 float32): {unpacked_floats}")
            # ------------------------------------------
            await websocket.send_bytes(payload)

            # Cede el event loop: permite atender otros WebSockets durante la espera
            await asyncio.sleep(FRAME_INTERVAL)

    except WebSocketDisconnect:
        print(f"[stream] cliente desconectado: {client}")

    except Exception as exc:
        print(f"[stream] error con {client}: {type(exc).__name__}: {exc}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Arranque directo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Usar: python main.py
    # O directamente: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,       # reload=True útil en dev local (no en Docker)
    )
