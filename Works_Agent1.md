# Works_Agent1 — Bitácora de Entregas
# Ingeniero A: Infraestructura y Datos
# Última actualización: 2026-04-25

---

## PARA AGENTES 2 Y 3 — Leer antes de integrar

Este archivo documenta todo lo que Agente 1 ha entregado, con rutas exactas,
contratos de interfaz vigentes y estado de validación en hardware real.

---

## DÍA 1 — COMPLETO (2026-04-25)

### Hardware verificado

| Componente | Valor |
|------------|-------|
| GPU | AMD MI300X — `gfx942` |
| VRAM HBM total | 192 GB |
| VRAM baseline (idle) | 0.0 GB |
| VRAM pico forward pass | **0.432 GB** |
| PyTorch | 2.9.0.dev+rocm7.0.0 |
| ROCm | 7.0.0 (instancia AMD Developer Cloud) |
| Python | 3.10.12 |
| Workspace | `/root/AMD_PROJECT` |

### Telemetría VRAM — Implicaciones para Agente 2 (Día 4)

Medición en hardware real (2026-04-26):

| Métrica | Valor |
|---------|-------|
| VRAM baseline | 0.0 GB |
| VRAM pico 1 forward pass | 0.432 GB |
| VRAM disponible para ensemble | 192 GB |
| Samples máximos teóricos (100% VRAM) | 444 |
| Samples máximos seguros (80% VRAM) | **355** |
| **Recomendación ensemble Día 4** | **N = 30–50 samples** |

**Para Agente 2:**
- Con N=30: 30 × 0.432 = **12.96 GB** — 6.7% de VRAM, completamente seguro
- Con N=50: 50 × 0.432 = **21.6 GB** — 11.2% de VRAM, sin riesgo
- Con N=100: 100 × 0.432 = **43.2 GB** — 22.5% de VRAM, viable si se necesita
- **No hay cuello de botella de memoria para el ensemble VP-SDE en este hardware**
- Agente 1 NO necesita ajustar paralelismo en Día 4 por restricción de VRAM
- Revisar si la restricción real es tiempo de cómputo, no memoria

---

### Archivos entregados

| Archivo | Descripción | Estado |
|---------|-------------|--------|
| `simulate_stream.py` | Generador async de frames ECG sintéticos | Validado en GPU |
| `data_loader.py` | Carga MIMIC-III WFDB → tensor (12, 5000) float32 | Validado CPU+GPU |
| `config/stream_config.py` | Constantes canónicas compartidas | Vigente |
| `tests/test_stream_local.py` | Harness de validación del stream | Pasa |
| `tests/test_data_loader.py` | Harness de validación del data loader | Pasa |
| `requirements.txt` | Dependencias pinneadas (sin torch — viene del base image) | Vigente |
| `Dockerfile` | Build Día 1 — base `rocm/primus:v26.2` | Draft |

---

### Contrato — Stream para Agente 3

```
PROTOCOLO WebSocket acordado (Decisión 1-5, 2026-04-25)

FRAME BINARIO (ECG):
  formato  : float32 little-endian, C-order, sin cabecera
  shape    : (12, 25) = 1 200 bytes por frame
  rate     : 20 fps (50ms entre frames)

FRAME TEXTO (control):
  formato  : JSON UTF-8, opcode TEXT nativo WebSocket
  schema   : {"type": str, "timestamp_ms": int, "payload": dict}
  tipos    : "start" | "stop" | "alert" | "metadata" | "reconstruction" | "uncertainty"
```

**Cómo usar `simulate_stream.py`:**

```python
import sys
sys.path.insert(0, "/root/AMD_PROJECT")

import asyncio
from simulate_stream import ECG_QUEUE, stream_generator

# Arrancar el generador como tarea background dentro del proceso FastAPI
producer = asyncio.create_task(stream_generator())

# Consumir frames desde el WebSocket handler
frame = await ECG_QUEUE.get()  # ndarray (12, 25) float32
raw_bytes = frame.tobytes()    # 1200 bytes listos para ws.send_bytes()
```

**Backpressure:**
- `ECG_QUEUE` tiene `maxsize=40` → latencia máxima de buffer = **2 segundos**
- Drop strategy: frame NUEVO descartado si la cola está llena (`put_nowait` + catch `QueueFull`)
- Contador de drops expuesto: `from simulate_stream import get_drop_count`

**Constantes canónicas — importar siempre desde `config/stream_config.py`:**

```python
from config.stream_config import (
    CHUNK_SIZE,      # 25  — muestras por frame
    SAMPLE_RATE,     # 500 — Hz
    N_CHANNELS,      # 12  — derivaciones ECG
    FRAME_INTERVAL,  # 0.050 — segundos entre frames
    QUEUE_MAXSIZE,   # 40  — frames máximos en cola
    LEADS_ORDER,     # ["I","II","III","aVR","aVL","aVF","V1"..."V6"]
)
```

**Resultado del test en hardware real:**
```
Frames capturados : 40
Elapsed           : 1.96s
FPS efectivo      : 20.4
Frame shape       : (12, 25)   ✓
Frame dtype       : float32    ✓
Bytes por frame   : 1200       ✓
Frames dropped    : 0          ✓
```

---

### Contrato — Tensores para Agente 2

```
TENSOR CANÓNICO: (B, 12, 5000) float32

  B    = batch size (variable)
  12   = derivaciones ECG en orden LEADS_ORDER
  5000 = muestras @ 500Hz = 10 segundos de señal
```

**Cómo usar `data_loader.py`:**

```python
import sys
sys.path.insert(0, "/root/AMD_PROJECT")

from data_loader import load_batch, to_gpu, validate_shape

# Cargar batch desde MIMIC-III (REAL: ruta donde está montado el dataset)
# DATA_PATH = "/ruta/a/mimic3wdb/records"  # REAL: proveer cuando esté disponible
tensors = load_batch(["/path/registro1", "/path/registro2"])  # REAL: paths reales
tensors_gpu = to_gpu(tensors)   # mueve a cuda:0

# Validación de contrato (lanza AssertionError si shape/dtype incorrecto)
validate_shape(tensors_gpu)     # usar en asserts de entrenamiento
```

**Resultado del test en hardware real:**
```
validate_shape (4, 12, 5000) float32  ✓
validate_shape rechaza 11 canales     ✓
validate_shape rechaza float64        ✓
to_gpu → device=cuda:0               ✓
VRAM usada (4 tensores sintéticos): 0.001 GB
VRAM disponible: 205.8 GB            ✓
```

---

### Pendiente Día 1 (no bloqueante para A2/A3)

- MIMIC-III no descargado aún (requiere credenciales PhysioNet — en trámite)
- PTB-XL como fallback: disponible cuando se provea `DATA_PATH` # REAL
- `data_loader.py` listo para recibir rutas reales en cuanto estén disponibles

---

## DÍA 2 — PENDIENTE

**Tareas planificadas:**
- `datasets/mimic_dataset.py` — `torch.utils.data.Dataset` wrapper
- `datasets/corruption.py` — corrupción sintética:
  - `corrupt_disconnect()` — canal → 0 (electrodo desconectado)
  - `corrupt_gaussian()` — ruido gaussiano configurable (SNR objetivo)
  - `corrupt_interference()` — interferencia 50/60Hz
- DataLoaders paralelizados (`num_workers ≥ 4`, pinned memory)
- Validación con Agente 2: sin cuellos de botella en VRAM

---

## DÍAS 3–7 — PLANIFICADOS (ver Memory_Agent1.md)

---

*Agente 1 — Ingeniero de Infraestructura y Datos | AMD Hackathon*
