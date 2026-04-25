# Works_Agent3 — Bitácora de Entregas
# Ingeniero C: Agente Autónomo y Full-Stack
# Última actualización: 2026-04-25

---

## PARA AGENTES 1 Y 2 — Leer antes de integrar

Este archivo documenta todo lo que Agente 3 ha entregado, con rutas exactas,
contratos de interfaz vigentes y estado de validación en hardware real.

---

## ENTORNO REAL CONFIRMADO (de Works_Agent1.md)

| Componente | Valor real en instancia |
|------------|------------------------|
| GPU | AMD MI300X — `gfx942` |
| VRAM total | 205.8 GB |
| PyTorch | 2.9.0.dev+rocm7.0.0 |
| ROCm | 7.0.0 |
| Python | 3.10.12 |
| Workspace | `/root/AMD_PROJECT` |

---

## DÍA 1 — EN PROGRESO (2026-04-25)

### Archivos entregados

| Archivo | Descripción | Estado |
|---------|-------------|--------|
| — | Pendiente de reporte | — |

### Contratos de entrada esperados de Agente 1

```
STREAM WebSocket (ya acordado):
  formato  : float32 little-endian, C-order
  shape    : (12, 25) = 1200 bytes/frame
  rate     : 20 fps
  Queue    : ECG_QUEUE maxsize=40 (2s de buffer)
  import   : from simulate_stream import ECG_QUEUE, stream_generator
```

### Métricas de rendimiento frontend

```
FPS mínimo Día 1    : 30 FPS con datos sintéticos
FPS objetivo demo   : 144 FPS en hardware final
Pendiente — Agente 3 actualizará este bloque al completar Día 1
```

### Puertos a abrir en UFW cuando sea necesario

```bash
sudo ufw allow 8000   # FastAPI backend
sudo ufw allow 3000   # Next.js frontend
```

### Pendientes Día 1

- [ ] FastAPI + Uvicorn en puerto 8000 con handler WebSocket /stream
- [ ] `synthetic_emitter.py` — 12 canales QRS sintético a 500Hz
- [ ] Next.js en puerto 3000 con webgl-plot, 12 líneas independientes
- [ ] Buffer circular TypeScript por canal
- [ ] Canvas a ≥30 FPS con datos sintéticos sin artefactos visuales

---

## DÍA 2 — PENDIENTE

**Tareas planificadas:**
- `backend/heuristics.py` — tres detectores (varianza, baseline, FFT con torch.fft.rfft)
- `backend/agent_fsm.py` — FSM completa (MONITORING → INTERCEPTING → RECONSTRUCTING → INTEGRATING|ALERTING)
- `backend/inference_bridge.py` — mock stub (reemplazar con Agente 2 en Día 3)
- Validar: detección de corrupción en <100ms

---

## DÍAS 3–7 — PLANIFICADOS (ver Memory_Agent3.md)

---

*Agente 3 — Ingeniero de Agente Autónomo y Full-Stack | AMD Hackathon*
