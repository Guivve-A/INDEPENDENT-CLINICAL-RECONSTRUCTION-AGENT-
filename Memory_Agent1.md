# Memory_Agent1 — Ingeniero de Infraestructura y Datos
# Agente Autónomo de Reconstrucción Clínica | AMD Hackathon

---

## IDENTIDAD Y ROL

- **Rol:** Ingeniero A — Infraestructura y Datos
- **Responsabilidades核心:**
  1. Garantizar funcionamiento óptimo del hardware AMD MI300X bajo ROCm 7.2.0
  2. Gestionar ingesta y transformación de datos fisiológicos MIMIC-III
  3. Empaquetar el sistema completo en Docker unificado

---

## CANAL DE SOPORTE HUMANO

El usuario (Guillermo) actúa como operador humano del sprint y puede ejecutar
cualquier acción externa que un agente de IA no puede realizar directamente.

**Puedes solicitarle ayuda para:**
- Provisionar y configurar instancias en **AMD Developer Cloud** (zona, tipo de instancia, claves SSH)
- Habilitar o instalar **ROCm** en la instancia (drivers, verificación de hardware con `rocminfo`)
- Configurar **Docker** en la instancia remota (daemon, permisos de dispositivo `/dev/kfd`, `/dev/dri`)
- Descargar el dataset **MIMIC-III** desde PhysioNet (credenciales, acuerdo de uso, comandos wget/rsync)
- Ajustes de **NUMA / kernel** en la instancia (acceso root, archivos de configuración del sistema)
- Gestionar **credenciales y variables de entorno** (claves API, tokens, archivos `.env`)
- Cualquier operación que requiera **acceso físico o privilegios de administrador** en la nube

**Regla obligatoria — Instrucciones Paso a Paso:**
Cuando necesites la intervención del usuario, debes entregarle una guía completa con:

1. **Objetivo** — qué se logrará con estos pasos y por qué es necesario
2. **Prerrequisitos** — qué debe tener listo antes de empezar
3. **Pasos numerados** — cada comando exacto, cada campo de formulario, cada opción de menú
4. **Verificación** — cómo confirmar que el paso fue exitoso antes de continuar al siguiente
5. **Qué reportar de vuelta** — la salida o dato exacto que necesitas que te comunique

> NO emitas instrucciones vagas como "configura el contenedor".
> Emite instrucciones ejecutables, por ejemplo:
> "Ejecuta: `docker run --device=/dev/kfd --device=/dev/dri --group-add video rocm/primus:v26.2 bash`
>  luego comparte la salida completa de `rocminfo | grep 'Agent Type'`"

---

## STACK TECNOLÓGICO

| Componente        | Especificación                          |
|-------------------|-----------------------------------------|
| GPU               | AMD MI300X                              |
| Runtime           | ROCm 7.2.0                              |
| Contenedor base   | `rocm/primus:v26.2`                     |
| Dataset           | MIMIC-III (registros multi-segmento)    |
| Librería ECG      | `wfdb-python`                           |
| Tensor shape      | `(batch, 12, 5000)` — 12 canales, 5000 puntos |
| Backend API       | FastAPI + WebSocket (gestionado por Agente 3) |
| Formato empaquetado | Docker + `requirements.txt`           |

---

## PROTOCOLO DE ACOPLE CON OTROS AGENTES

### → RECIBO DE Agente 2
- **Día 3:** Especificaciones del modelo para perfilar en ROCm (kernel fusion targets, capas críticas)
- **Día 4:** Requerimientos de paralelismo para muestreo múltiple de incertidumbre (Monte Carlo Dropout o similar)

### → ENTREGO A Agente 2
- **Día 1:** Tensores limpios `(batch, 12, 5000)` y DataLoaders listos en GPU — validados sin errores de shape
- **Día 2:** Pipeline de corrupción sintética funcional; verificación de que no hay cuellos de botella en VRAM

### → RECIBO DE Agente 3
- **Día 5:** Especificaciones del entorno backend para construir el Docker final (puertos, variables de entorno, rutas de montaje)

### → ENTREGO A Agente 3
- **Día 1:** Streaming de datos fisiológicos simulados como tensores para pruebas del servidor FastAPI/WebSocket

---

## PROTOCOLO DE ACOPLE CON AGENTE 3 — WIRE PROTOCOL WebSocket
> Decisión acordada Día 1 (2026-04-25). Ambos agentes implementan este contrato.

### Decisión 1 — Formato de frames ECG
**ACEPTADO: Binary float32, sin cabecera, little-endian**
- Shape por mensaje: `(12, 25)` → 300 × float32 = **1 200 bytes por frame**
- Layout: row-major (C-order), canal → tiempo
- Sin bytes de prefijo de tipo en frames binarios
- Razón: nuestros tensores GPU ya son float32; cero conversión en el pipeline

### Decisión 2 — N_chunk
**ACEPTADO: `CHUNK_SIZE = 25` muestras**
- 25 samples × (1/500Hz) = **50ms por frame → 20 frames/segundo**
- Alineación exacta: 5 000 / 25 = 200 frames por segmento MIMIC-III completo
- Constante configurable en `config/stream_config.py` para facilitar ajuste en Día 3

### Decisión 3 — Canal de control/eventos
**CONTRAPROPUESTA ACEPTADA: distinción nativa TEXT/BINARY de WebSocket**
- **Frames BINARY** → datos ECG float32 `(12, 25)`
- **Frames TEXT** → mensajes de control JSON UTF-8
- No se usa type-prefix byte ni canal separado
- Razón: el protocolo WebSocket ya distingue ambos tipos a nivel de opcode; todas las librerías (websockets, Starlette) lo manejan nativamente sin overhead extra
- Formato JSON de control:
  ```json
  {"type": "start|stop|alert|metadata|reconstruction|uncertainty",
   "timestamp_ms": 1745000000000,
   "payload": {}}
  ```

### Decisión 4 — Modelo de proceso de simulate_stream.py
**MISMO PROCESO FASTAPI, importado como módulo**
- `simulate_stream.py` expone un **async generator** (`async def ecg_stream()`)
- Se conecta al WebSocket handler de Agente 3 via `asyncio.Queue`
- No corre como subprocess ni socket separado
- Razón: igual al patrón que Agente 3 usa para inferencia GPU (`asyncio.run_in_executor`); evita complejidad de IPC; el simulador genera NumPy/torch sintético, no ROCm inference, por lo que `await asyncio.sleep(FRAME_INTERVAL)` entre frames no bloquea el event loop
- Señal de arranque/parada: mensaje JSON TEXT `{"type": "start"}` / `{"type": "stop"}`

### Constantes canónicas (compartidas con Agente 3)
```python
# config/stream_config.py
CHUNK_SIZE    = 25      # muestras por frame WebSocket
SAMPLE_RATE   = 500     # Hz (MIMIC-III estándar)
N_CHANNELS    = 12      # derivaciones ECG
FRAME_INTERVAL = 0.050  # segundos entre frames
LEADS_ORDER   = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
```

### Resumen del contrato de bytes en el wire
```
ws://host:PORT/ws/ecg

FRAME BINARIO (ECG):
  bytes[0:1200] = float32 array, shape (12,25), little-endian, C-order

FRAME TEXTO (control):
  UTF-8 JSON → {"type": str, "timestamp_ms": int, "payload": dict}
```

### Decisión 5 — Backpressure: ECG_QUEUE maxsize=40
**Acordado 2026-04-25. Corrección aplicada antes de crear simulate_stream.py.**

- `asyncio.Queue()` sin argumento tiene `maxsize=0` → ilimitada → viola Principio 3 de Agente 3
- `ECG_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)` donde `QUEUE_MAXSIZE = 40`
- **Latencia máxima de buffer:** 40 frames × 50ms = **2 segundos**
- **Estrategia de drop:** `put_nowait()` + captura de `QueueFull` → descarta el frame NUEVO
  - Correcto para señal en tiempo real: nunca replantear datos obsoletos a un consumer lento
  - Drop no silencioso en producción: `_drop_count` expuesto via `get_drop_count()` para telemetría
- La constante `QUEUE_MAXSIZE` vive en `config/stream_config.py` — Agente 3 la importa desde ahí

> **Notificación a Agente 3:** Actualizar `Memory_Agent3.md` reemplazando
> "(formato pendiente coordinar Día 1)" por este protocolo acordado.
> Usar `QUEUE_MAXSIZE = 40` de `config/stream_config.py` como referencia
> de latencia máxima del buffer (2 segundos).

---

## PLAN DE EJECUCIÓN — SPRINT 7 DÍAS

### Día 1 — Provisionamiento e Ingesta Base
**Tareas:**
- Provisionar instancia MI300X
- Deshabilitar balanceo NUMA (`numactl --interleave=all` o config kernel)
- Lanzar contenedor: `docker run --device=/dev/kfd --device=/dev/dri rocm/primus:v26.2`
- Implementar rutinas `wfdb-python` para cargar registros MIMIC-III multi-segmento
- Convertir registros a tensores shape `(batch, 12, 5000)`
- Generar stream de tensores simulados para Agente 3

**Validación:**
- Agente 2 lee tensores sin errores de shape
- Agente 3 puede consumir el stream simulado sin errores

**Artefactos de salida:**
- `data_loader.py` — carga MIMIC-III → tensores GPU
- `simulate_stream.py` — generador de tensores fisiológicos para WebSocket

---

### Día 2 — DataLoaders Paralelizados y Corrupción Sintética
**Tareas:**
- Diseñar DataLoaders con workers paralelos (`torch.utils.data.DataLoader`, `num_workers >= 4`)
- Implementar funciones de corrupción sintética:
  - `corrupt_disconnect()` — simula desconexión de electrodo (canal → 0 o NaN)
  - `corrupt_gaussian()` — ruido gaussiano configurable (SNR objetivo)
  - `corrupt_interference()` — interferencia de red eléctrica (60Hz/50Hz artifact)
- Pinned memory para transferencias CPU→GPU eficientes

**Validación (con Agente 2):**
- Pipeline no genera cuellos de botella en VRAM del MI300X
- Throughput de batches medido y dentro de objetivo

**Artefactos de salida:**
- `datasets/mimic_dataset.py`
- `datasets/corruption.py`

---

### Día 3 — Perfilado y Optimización de Inferencia
**Tareas:**
- Perfilar pase directo de la red usando ROCm Profiler (`rocprof`)
- Aplicar kernel fusion donde sea posible
- Evaluar exportación a ONNX-RT para reducir latencia CPU→GPU
- Medir y documentar latencias por capa

**Validación (con Agente 3):**
- Latencia de inferencia permite mantener flujo en tiempo real (objetivo: < umbral definido por Agente 3)

**Artefactos de salida:**
- `profiling/rocm_profile_report.json`
- `inference/optimized_forward.py`

---

### Día 4 — Ajuste de Paralelismo y Telemetría de Hardware
**Tareas:**
- Ajustar configuración de paralelismo para que muestreo múltiple de incertidumbre (Agente 2) no sature VRAM
- Estrategias: gradient checkpointing, batch size dinámico, streams HIP concurrentes
- Extraer telemetría de hardware vía ROCm/MAD:
  - Utilización GPU, VRAM usada, temperatura, bandwidth de memoria

**Artefactos de salida:**
- `telemetry/hardware_metrics.py`
- `config/parallelism_config.yaml`

---

### Día 5 — Docker Unificado y Pruebas de Estrés
**Tareas:**
- Construir Dockerfile unificado integrando:
  - Código de Agente 1 (data pipeline)
  - Código de Agente 2 (modelo)
  - Código de Agente 3 (FastAPI/WebSocket)
- Ejecutar pruebas de estrés sobre el WebSocket (conexiones concurrentes, latencia bajo carga)
- Verificar reproducibilidad del build

**Artefactos de salida:**
- `Dockerfile`
- `docker-compose.yml`
- `tests/stress_websocket.py`

---

### Día 6 — Reporte de Rendimiento y Auditoría de Datos
**Tareas:**
- Procesar logs del profiler ROCm → reporte de rendimiento legible
- Auditar pipeline de anonimización de datos clínicos MIMIC-III:
  - Verificar que ningún identificador PHI escapa al pipeline
  - Conformidad con uso educativo/investigación del dataset

**Artefactos de salida:**
- `reports/performance_report.md`
- `reports/data_privacy_audit.md`

---

### Día 7 — Congelamiento y Entrega Final
**Tareas:**
- Congelar todas las dependencias en `requirements.txt` con versiones exactas
- Verificar que el sistema levanta con un solo comando:
  ```bash
  docker-compose up --build
  ```
- Documentar comando único de arranque en README
- Tag final del contenedor Docker

**Validación final:**
- Build limpio desde cero en < 10 minutos
- Pipeline end-to-end funciona sin intervención manual

**Artefactos de salida:**
- `requirements.txt` (versiones pinneadas)
- `README.md` (instrucción de arranque único)

---

## ESTRUCTURA DE ARCHIVOS ESPERADA

```
AMD_PROJECT/
├── Memory_Agent1.md          ← este archivo
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── data_loader.py
├── simulate_stream.py
├── datasets/
│   ├── mimic_dataset.py
│   └── corruption.py
├── inference/
│   └── optimized_forward.py
├── profiling/
│   └── rocm_profile_report.json
├── telemetry/
│   └── hardware_metrics.py
├── config/
│   └── parallelism_config.yaml
├── tests/
│   └── stress_websocket.py
└── reports/
    ├── performance_report.md
    └── data_privacy_audit.md
```

---

## RESTRICCIONES Y PRINCIPIOS OPERATIVOS

1. **Shape canónico de tensores:** Siempre `(batch, 12, 5000)` — nunca cambiar sin notificar a Agente 2 y Agente 3.
2. **VRAM primero:** Toda decisión de diseño prioriza no saturar la VRAM del MI300X.
3. **Datos clínicos:** MIMIC-III se usa solo bajo términos de licencia PhysioNet. Ningún dato PHI sale del pipeline.
4. **Reproducibilidad:** Seeds fijadas, versiones de dependencias pinneadas desde Día 1.
5. **Interfaz de acople limpia:** Los tensores entregados a Agente 2 y Agente 3 deben pasar siempre una validación de shape antes de ser publicados.
6. **Un solo comando de arranque:** El sistema final debe levantarse con `docker-compose up --build` sin pasos manuales adicionales.

---

## ESTADO ACTUAL

| Día | Estado     | Notas                        |
|-----|------------|------------------------------|
| 1   | EN CURSO   | Protocolo WS acordado · simulate_stream.py · data_loader.py · Dockerfile · tests · pendiente: provisión MI300X + MIMIC-III (delegado al operador humano) |
| 2   | PENDIENTE  |                              |
| 3   | PENDIENTE  |                              |
| 4   | PENDIENTE  |                              |
| 5   | PENDIENTE  |                              |
| 6   | PENDIENTE  |                              |
| 7   | PENDIENTE  |                              |

---

*Última actualización: 2026-04-25 (backpressure Queue) | Agente 1 — Ingeniero de Infraestructura y Datos*
