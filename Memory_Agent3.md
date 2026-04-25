# Memory_Agent3 — Ingeniero de Agente Autónomo y Full-Stack
# Agente Autónomo de Reconstrucción Clínica | AMD Hackathon

---

## PRINCIPIO ANTI-ALUCINACIÓN

Este archivo existe para que el agente actúe con precisión, no con confianza falsa.
Cada sección distingue entre:
- ✅ **VERIFICADO** — hecho técnico establecido en la literatura o en el contrato del proyecto
- ⚙️ **PENDIENTE DE CALIBRAR** — parámetro que debe determinarse experimentalmente
- ❌ **NO ASUMIR** — área donde es fácil alucinar; buscar evidencia antes de decidir

---

## IDENTIDAD Y ROL

- **Rol:** Ingeniero C — Agente Autónomo y Full-Stack
- **Propietario de:** Máquina de estados del agente, capa de heurísticas clínicas, servidor FastAPI/WebSocket, frontend Next.js/WebGL, lógica del umbral τ, consola de diagnóstico, Demo-Mode
- **NO soy responsable de:** infraestructura ROCm/NUMA (Agente 1), arquitectura del modelo/entrenamiento/inferencia/incertidumbre matemática (Agente 2)
- **Soy el integrador visible:** el producto final que el jurado ve y toca depende de este agente

---

## CANAL DE SOPORTE HUMANO

El usuario (Guillermo) actúa como operador humano del sprint y puede ejecutar
cualquier acción externa que un agente de IA no puede realizar directamente.

**Puedes solicitarle ayuda para:**
- Instalar **Node.js / npm** en la instancia o en la máquina de desarrollo local
- Ejecutar `npx create-next-app@latest` y confirmar la estructura del proyecto generado
- Configurar **puertos de red** en la instancia AMD Developer Cloud (reglas de firewall, security groups)
- Instalar extensiones o plugins de **navegador** necesarios para las pruebas de WebGL / WebSocket
- Gestionar **variables de entorno y archivos `.env`** en la instancia remota o en el sistema local
- Acceder a paneles de **AMD Developer Cloud** para obtener IPs, hostnames o certificados SSL
- Ejecutar pruebas manuales de **WebSocket y FPS** en el navegador cuando se requiera validación visual
- Cualquier operación que requiera **intervención manual en terminal, navegador o consola de nube**

**Regla obligatoria — Instrucciones Paso a Paso:**
Cuando necesites la intervención del usuario, debes entregarle una guía completa con:

1. **Objetivo** — qué se logrará con estos pasos y por qué es necesario
2. **Prerrequisitos** — qué debe tener listo antes de empezar
3. **Pasos numerados** — cada comando exacto, cada campo de formulario, cada opción de menú
4. **Verificación** — cómo confirmar que el paso fue exitoso antes de continuar al siguiente
5. **Qué reportar de vuelta** — la salida o dato exacto que necesitas que te comunique

> NO emitas instrucciones vagas como "levanta el servidor".
> Emite instrucciones ejecutables, por ejemplo:
> "En la terminal de la instancia ejecuta: `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
>  luego abre en tu navegador `http://<IP_INSTANCIA>:8000/health` y comparte la respuesta JSON completa."

---

## CONTRATO DE DATOS — INTERFACES DE ACOPLE

### LO QUE RECIBO (inputs que no controlo)

| Fuente   | Día | Artefacto                                         | Shape / Tipo                        | Estado               |
|----------|-----|---------------------------------------------------|-------------------------------------|----------------------|
| Agente 1 | 1   | Stream simulado de tensores fisiológicos          | frames `(12, 25)` f32, 20fps         | ✅ PROTOCOLO ACORDADO Día 1 |
| Agente 1 | 5   | Imagen Docker base para empaquetar backend        | `rocm/primus:v26.2` + deps          | ✅ acordado          |
| Agente 2 | 3   | Función `reconstruct(signal, mask)`               | `(1, 12, W)` → `(1, 1, W)` f32     | ✅ acordado          |
| Agente 2 | 4   | Función `reconstruct_with_uncertainty(signal, mask, N)` | → `(mean, variance)` ambos `(1, 1, W)` f32 | ✅ acordado |

> **PROTOCOLO DE STREAM ACORDADO CON AGENTE 1 (2026-04-25):**
> - **Frames ECG:** WebSocket BINARY opcode — float32 little-endian, C-order, shape `(12, 25)` = 1200 bytes/frame
> - **Mensajes de control/evento:** WebSocket TEXT opcode — JSON UTF-8 (sin canal separado, sin byte de tipo)
> - **CHUNK_SIZE = 25** (constante canónica en `config/stream_config.py` — importar desde ahí, no hardcodear)
> - **simulate_stream.py** corre en el mismo proceso FastAPI como async generator, pone frames en `asyncio.Queue(maxsize=QUEUE_MAXSIZE)`
> - **QUEUE_MAXSIZE = 40** definido en `config/stream_config.py` — importar desde ahí (NO hardcodear 40). Buffer máximo = 40 × 50ms = **2 segundos** de señal. Este valor es la referencia para dimensionar el timeout de reconstrucción en Día 3: timeout > 2s.
> - En el browser: `event.data instanceof ArrayBuffer` → frame ECG; `typeof event.data === 'string'` → JSON control

> **Regla crítica:** La latencia de inferencia de Agente 2 es DESCONOCIDA hasta el Día 3.
> El tiempo que el stream estará pausado durante reconstrucción no se puede hardcodear — diseñar con timeout configurable.

### LO QUE ENTREGO (outputs de los que soy responsable)

| Destino  | Día | Artefacto                                                   | Tipo                    | Estado    |
|----------|-----|-------------------------------------------------------------|-------------------------|-----------|
| Agente 2 | 3+  | Tensor de señal interceptada + máscara del canal corrupto   | `(1, 12, W)`, máscara binaria `(1, 12, W)` | PENDIENTE |
| Agente 1 | 5   | Lista de dependencias ASGI + frontend para Dockerfile       | `requirements_agent3.txt` + `Dockerfile.frontend` | PENDIENTE |
| Agente 1 | 5   | Puertos, variables de entorno, rutas de montaje             | documento de configuración | PENDIENTE |

> **Regla crítica:** El tensor que entrego a Agente 2 debe tener shape exactamente `(1, 12, W)` con W ≤ 5000.
> Si la ventana capturada es más corta, agregar padding explícito — Agente 2 no asume relleno automático.
> La máscara binaria indica qué canales están corruptos (1 = corrupto, 0 = sano).
> ❌ NO asumir que siempre es 1 canal corrupto — puede ser 1-3 simultáneos.

---

## ANÁLISIS DE DEPENDENCIAS CRÍTICAS

### Bloqueos reales por día

| Día | Bloqueado por                         | Puedo avanzar sin esperar en...                         |
|-----|---------------------------------------|---------------------------------------------------------|
| 1   | **Nada** — arranque independiente     | FastAPI skeleton, WebSocket handlers, Next.js + webgl-plot con datos sintéticos propios |
| 2   | Nada                                  | Heurísticas sobre datos sintéticos, máquina de estados stub |
| 3   | **Agente 2** — función `reconstruct()`| Frontend y heurísticas ya deben estar listos para este punto |
| 4   | **Agente 2** — función `reconstruct_with_uncertainty()` | Lógica τ puede diseñarse, solo falta conectarla |
| 5   | **Agente 1** — Docker integration     | Consola de diagnóstico y sincronización WebGL son independientes |
| 6   | Días 1-5 completos                    | Demo-Mode shell puede prepararse antes |
| 7   | Día 6 completo                        | — |

> **Riesgo principal:** Si Agente 2 entrega `reconstruct()` con retraso el Día 3, el flujo end-to-end
> no puede probarse hasta tarde en el sprint. Mitigación: implementar un mock stub de inferencia
> el Día 2 que devuelva ruido gaussiano + varianza constante, para probar toda la cadena sin el modelo real.

### Preguntas abiertas que debo resolver con cada agente

**Para Agente 1 (resolver Día 1) — RESUELTO 2026-04-25:**
1. ~~¿Formato del frame WebSocket?~~ **RESUELTO:** BINARY opcode para frames ECG (float32), TEXT opcode para JSON control.
2. ~~¿Tamaño del chunk?~~ **RESUELTO:** CHUNK_SIZE = 25 (importar de `config/stream_config.py`).
3. ~~¿`simulate_stream.py` corre en el mismo proceso?~~ **RESUELTO:** Mismo proceso FastAPI, async generator + `asyncio.Queue`.

**Para Agente 2 (resolver Día 2-3):**
1. ~~¿La función `reconstruct()` es thread-safe?~~ **RESUELTO:** No es thread-safe — usa `torch.manual_seed()` que muta estado global. Solución aplicada en `inference_bridge.py` con `_GPU_EXECUTOR` singleton.
2. Latencia esperada por reconstrucción en MI300X (para dimensionar el timeout del stream)
3. ¿El modelo maneja W variable o necesita siempre W=5000? (afecta cuánto contexto capturo)

---

## ARQUITECTURA TÉCNICA — DECISIONES VERIFICADAS

### Backend: FastAPI + WebSocket

```
FastAPI (Uvicorn ASGI, puerto 8000)
├── GET  /health          → estado del agente
├── GET  /metrics         → telemetría (τ actual, última varianza, fps)
├── WS   /stream          → ECG data + eventos del agente → browser
└── WS   /control         → comandos del jurado → servidor (inyectar ruido, ajustar τ)
```

✅ FastAPI + Uvicorn es ASGI nativo — WebSocket sin bloqueos en el event loop.

**Problema crítico con inferencia GPU desde async — dos capas:**

**Capa 1: No bloquear el event loop**
- La función de Agente 2 es síncrona y bloquea (corre en GPU)
- Llamarla directamente desde `async def` bloquea el event loop entero → todos los WebSockets se congelan
- ❌ NO llamar funciones GPU bloqueantes directamente desde un handler `async def`

**Capa 2: PyTorch no es thread-safe con estado global compartido**
- `run_in_executor(None, ...)` usa el ThreadPoolExecutor por defecto de Uvicorn (múltiples workers)
- Con varios clientes WebSocket activos, múltiples hilos pueden llamar a `reconstruct_with_uncertainty` simultáneamente
- Agente 2 usa `torch.manual_seed(seed)` dentro del ensemble — esto muta el RNG global de PyTorch
- Dos hilos ejecutando `torch.manual_seed()` concurrentemente = race condition garantizada en las semillas
- ❌ `run_in_executor(None, ...)` no es suficiente — sigue siendo inseguro con múltiples clientes

**✅ Solución: ejecutor singleton dedicado con un solo hilo**
```python
# backend/inference_bridge.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Un solo hilo serializa TODAS las llamadas GPU — módulo-level singleton
_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1)

async def reconstruct_async(signal, mask, N):
    loop = asyncio.get_running_loop()
    mean, variance = await loop.run_in_executor(
        _GPU_EXECUTOR,                   # ← ejecutor dedicado, NO None
        reconstruct_with_uncertainty,
        signal, mask, N
    )
    return mean, variance
```
- `max_workers=1`: solicitudes concurrentes se serializan automáticamente en la cola del executor
- El event loop de Uvicorn sigue desbloqueado — el `await` cede control mientras espera turno en la cola
- La GPU tiene exclusión mutua implícita: un solo hilo accede al modelo en cualquier momento dado
- Las semillas del ensemble de Agente 2 nunca se corrompen entre llamadas concurrentes

### Máquina de Estados del Agente

```
        ┌─────────────────────────────────────────────────────┐
        │                   MONITORING                        │
        │  Stream fluye normalmente hacia browser             │
        └──────────────────┬──────────────────────────────────┘
                           │  heurística detecta corrupción
                           ▼
        ┌─────────────────────────────────────────────────────┐
        │                  INTERCEPTING                       │
        │  Registra: channel_idx, timestamp_start            │
        │  Sigue capturando N_context muestras post-falla    │
        │  Browser recibe señal (marcada como corrupta)       │
        └──────────────────┬──────────────────────────────────┘
                           │  contexto suficiente capturado
                           ▼
        ┌─────────────────────────────────────────────────────┐
        │                RECONSTRUCTING                       │
        │  Pausa emisión del segmento corrupto al browser     │
        │  Llama Agente 2: reconstruct_with_uncertainty()     │
        │  Espera (mean, variance) en executor thread         │
        └──────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
       var < τ │                        │ var ≥ τ
              ▼                        ▼
   ┌──────────────────┐    ┌──────────────────────────┐
   │   INTEGRATING    │    │        ALERTING           │
   │  Inyecta parche  │    │  Emite alerta al browser  │
   │  (color distinto)│    │  NO inyecta reconstrucción│
   │  silenciosamente │    │  Resume stream original   │
   └────────┬─────────┘    └────────────┬──────────────┘
            └────────────┬──────────────┘
                         │
                         ▼
                    MONITORING
```

> ⚙️ **N_context** (muestras de contexto post-falla para el tensor de Agente 2): PENDIENTE CALIBRAR
> Debe ser suficiente para dar contexto al modelo sin introducir latencia excesiva.
> Valor inicial razonable: 250-500 muestras (0.5-1s a 500Hz).

### Heurísticas de Detección de Integridad de Señal

Las heurísticas detectan **corrupción de señal**, NO patologías cardíacas. Son tres detectores independientes por canal:

**Detector 1: Varianza de ventana corta**
```python
# Ventana: ~50ms = 25 muestras a 500Hz
# Corrupción tipo: electrodo desconectado (varianza → 0) o pico de ruido (varianza >> σ_baseline)
var_short = np.var(window[-25:], axis=-1)  # por canal
# Trigger: var_short < THRESH_LOW  (desconexión)
#       OR var_short > THRESH_HIGH * var_baseline  (spike de ruido)
```
- ⚙️ `THRESH_LOW`: PENDIENTE calibrar sobre datos MIMIC-III limpios
- ⚙️ `THRESH_HIGH`: PENDIENTE calibrar (razonable: 10× la varianza baseline del paciente)

**Detector 2: Cruce de línea base**
```python
# Detecta deriva DC o saturación del amplificador
mean_short = np.mean(window[-25:], axis=-1)
# Trigger: abs(mean_short) > BASELINE_THRESH_MV
```
- ⚙️ `BASELINE_THRESH_MV`: PENDIENTE (unidades dependen de normalización de Agente 1)

**Detector 3: Potencia frecuencial (FFT)**

**Decisión: OPCIÓN B — `torch.fft` en GPU.** ✅ Fijado antes de Día 2.

Justificación: Agent 1 entrega tensores ya en GPU. Usar NumPy requeriría `.cpu().numpy()` → round trip GPU→CPU→GPU innecesario en un pipeline de tiempo real. `torch.fft.rfft` opera directamente sobre el tensor en GPU sin copias.

```python
# Ventana: ~1s = 500 muestras a 500Hz
# Detecta: ruido muscular (EMI >100Hz) o interferencia de red (50/60Hz)
# Precondición: buffer_tensor shape (12, N) ya en GPU ('cuda')
window_gpu = buffer_tensor[..., -500:]          # (12, 500) en GPU
spectrum   = torch.fft.rfft(window_gpu, dim=-1) # (12, 251) complejo
power      = spectrum.abs().pow(2)              # (12, 251) real
freqs      = torch.fft.rfftfreq(500, d=1/500).to(buffer_tensor.device)
high_mask  = freqs > 100                        # bool mask sobre frecuencias
high_power = power[..., high_mask].sum(dim=-1)  # (12,) potencia alta-freq por canal
total_power = power.sum(dim=-1)                 # (12,)
# Trigger por canal: high_power[ch] > FREQ_THRESH * total_power[ch]
```

Contingencia (si frames llegan como bytes vía red): convertir primero con
`buffer_tensor = torch.frombuffer(raw_bytes, dtype=torch.float32).reshape(12, -1).to('cuda')`
y luego aplicar el código anterior. ❌ NO usar `np.fft` en ningún caso.

- ❌ **NO usar hipFFT directamente desde Python** — hipFFT es librería C/HIP, no tiene binding Python estable
- ❌ **NO usar `np.fft`** — introduce copia GPU→CPU innecesaria y contradice la regla del proyecto
- ⚙️ `FREQ_THRESH`: PENDIENTE calibrar

> **Regla:** Los tres detectores votan. Trigger final = cualquier detector individual activa.
> Si múltiples canales disparan simultáneamente → puede ser artefacto global (mover paciente) vs. falla de electrodo individual.
> ❌ NO asumir que siempre es 1 canal — la máscara puede tener hasta 3 canales en 1.

### Frontend: Next.js + WebGL

**Rendering de trazas ECG:**
```
webgl-plot → 12 WebGLLine objects
Cada línea: buffer circular de capacidad = ancho_canvas (px) × factor_zoom
Color coding:
  Señal normal:        #00FF88 (verde)
  Segmento corrupto:   #FF4444 (rojo, marcado pero mostrado)
  Segmento reconstruido: #00CCFF (cyan, interpolado del modelo)
  Estado ALERTING:     borde de canvas rojo pulsante
```

**Bandas de incertidumbre (Día 4):**
- WebGL custom geometry: triangle strips por encima y debajo de la traza reconstruida
- Opacidad: proporcional a `sqrt(variance)` — a mayor incertidumbre, banda más visible
- ❌ **NO usar SVG o Canvas 2D** para las bandas — mataría el presupuesto de 144 FPS

**Objetivo de rendimiento:**
- ✅ 144 FPS es el target en hardware de demo
- ✅ Degradación aceptable: 60 FPS en hardware limitado — diseñar con requestAnimationFrame adaptativo
- ~2.6 MB VRAM local en GPU del browser (webgl-plot)

### Umbral τ — Territorio Exclusivo de Agente 3

- τ es el umbral sobre la varianza del ensemble de Agente 2
- ✅ Agente 2 NO fija τ — es responsabilidad de este agente
- ⚙️ Unidades de τ: dependen de la normalización de las señales de Agente 1 (mV² o adimensional)
- ⚙️ Valor inicial: calibrar empiricamente el Día 4 sobre reconstrucciones reales
- El valor debe ser ajustable en tiempo real desde el frontend (control slider en Demo-Mode)

---

## PLAN DE EJECUCIÓN — SPRINT 7 DÍAS

### Día 1 — Servidor FastAPI, WebSocket y Canvas WebGL

**Objetivo:** Servidor WebSocket en Uvicorn; cliente web graficando señales a ≥60 FPS estables.

**Tareas:**
1. Montar proyecto FastAPI con estructura de carpetas (ver abajo)
2. Implementar handler WebSocket `/stream` con buffer de envío asíncrono
3. Crear `synthetic_emitter.py`: genera 12 canales de ruido + morfología QRS sintética a 500Hz como fallback mientras Agente 1 entrega `simulate_stream.py`
4. Montar proyecto Next.js (`npx create-next-app@latest`)
5. Instalar `webgl-plot` y configurar canvas de 12 líneas independientes
6. Implementar buffer circular en TypeScript para cada canal
7. **Coordinar con Agente 1:** acordar formato de frame WebSocket

**Validación crítica:**
- Canvas muestra 12 trazas moviéndose fluidamente sin artefactos visuales
- WebSocket no acumula backpressure — el servidor descarta frames si el cliente va lento
- ❌ NO declarar éxito si el FPS cae por debajo de 30 con datos sintéticos

**Artefactos de salida:**
- `backend/main.py`
- `backend/synthetic_emitter.py`
- `frontend/pages/index.tsx` con canvas WebGL funcional

---

### Día 2 — Heurísticas y Máquina de Estados (stub)

**Objetivo:** El servidor detecta fallas inyectadas con latencia < 100ms.

**Tareas:**
1. Implementar `backend/heuristics.py` con los tres detectores (varianza, baseline, FFT)
2. Implementar `backend/agent_fsm.py` con la máquina de estados completa
3. Crear `backend/inference_bridge.py` con stub mock:
   ```python
   # Mock para Día 2 — reemplazar con Agente 2 el Día 3
   def reconstruct_with_uncertainty_mock(signal, mask, N=20):
       mean = signal[:, mask[0].bool(), :]  # devuelve señal original como placeholder
       variance = torch.ones_like(mean) * 0.5  # varianza constante media
       return mean, variance
   ```
4. Conectar heurísticas al event loop de FastAPI — deben correr en cada frame recibido
5. Probar inyectando corrupción sintética desde `synthetic_emitter.py` y verificar que el estado cambia

**Validación crítica:**
- `MONITORING → INTERCEPTING` ocurre en < 100ms de iniciada la corrupción
- `RECONSTRUCTING` con el mock devuelve result y transiciona a `INTEGRATING` sin bloquear el event loop
- ❌ NO mover a Día 3 si el mock bloquea el WebSocket

**Artefactos de salida:**
- `backend/heuristics.py`
- `backend/agent_fsm.py`
- `backend/inference_bridge.py` (con mock)

---

### Día 3 — Conexión con Agente 2 y Flujo End-to-End

**Objetivo:** Anomalía detectada → GPU reconstruye → dato se reinyecta al frontend en color distinto.

**Tareas:**
1. Reemplazar mock en `inference_bridge.py` con llamada real a `reconstruct()` de Agente 2
2. Envolver la llamada en `asyncio.run_in_executor()` para no bloquear el event loop
3. Implementar pausa y reanudación del stream durante reconstrucción
4. Modificar el protocolo WebSocket para incluir metadata de evento:
   ```json
   {"type": "reconstructed_segment", "channel": 2, "start_ts": 1234.56, "samples": [...], "color": "cyan"}
   ```
5. Frontend: detectar `type == "reconstructed_segment"` y renderizar en color distinto
6. Medir latencia real de `reconstruct()` en MI300X — documentar para calibrar timeout

**Validación crítica:**
- El flujo end-to-end completo funciona sin errores
- El frontend muestra el segmento reconstruido claramente diferenciado del resto
- La latencia de pausa (tiempo que el stream está congelado) es visible pero tolerable (<3s)
- ❌ NO presentar como éxito si la señal reconstruida es visualmente incoherente con el contexto — reportar a Agente 2

**Artefactos de salida:**
- `backend/inference_bridge.py` (real, no mock)
- Protocolo de mensaje WebSocket documentado

---

### Día 4 — Umbral τ, Lógica de Decisión y Bandas de Incertidumbre

**Objetivo:** Agente emite veredictos probabilísticos correctos; bandas de confianza visibles en la interfaz.

**Tareas:**
1. Reemplazar `reconstruct()` por `reconstruct_with_uncertainty()` en `inference_bridge.py`
2. Implementar lógica τ en `agent_fsm.py`:
   ```python
   # mean_variance = varianza media sobre los W puntos del segmento
   mean_variance = variance.mean().item()
   if mean_variance < TAU:
       self.transition(State.INTEGRATING)
   else:
       self.transition(State.ALERTING)
   ```
3. Calibrar τ: ejecutar 20+ reconstrucciones sobre señales conocidas y observar distribución de varianzas
4. Implementar bandas de incertidumbre en WebGL:
   - Triangle strip: `mean + k*std` (banda superior) y `mean - k*std` (banda inferior)
   - k = 1.0 (1 desviación estándar) — ajustable
   - Fill translúcido: `rgba(0, 204, 255, 0.2)`
5. Implementar alerta visual en ALERTING: mensaje overlay + borde de canvas rojo pulsante
6. Hacer τ ajustable desde slider en el frontend (preparar para Demo-Mode)

**Validación crítica:**
- Casos de prueba: reconstrucción de alta calidad → `INTEGRATING`; ruido puro → `ALERTING`
- Las bandas de incertidumbre se expanden/contraen visiblemente según calidad de reconstrucción
- ⚙️ τ requiere calibración empírica — no existe valor "correcto" a priori

**Artefactos de salida:**
- `backend/agent_fsm.py` (con lógica τ completa)
- `frontend/components/UncertaintyBands.tsx`
- `config/agent_config.yaml` con τ inicial calibrado

---

### Día 5 — Consola de Diagnóstico, Stabilización y Docker

**Objetivo:** Interfaz de calidad comercial; imagen Docker funcional de un comando.

**Tareas:**
1. Construir `frontend/components/DiagConsole.tsx`:
   - Log de eventos en texto natural con timestamp: "14:23:01 — Canal 3 corrupto detectado (varianza: 0.002). Reconstrucción iniciada."
   - Colores por severidad: verde=integrado, amarillo=alerta, rojo=error
2. Sincronizar transiciones WebGL:
   - Double-buffer: preparar frame siguiente mientras se muestra el actual
   - Eliminar tearing: no mutar el buffer de webgl-plot durante un frame activo
3. Entregar a Agente 1 la lista de dependencias para Docker:
   ```
   # requirements_agent3.txt
   fastapi==0.115.x
   uvicorn[standard]==0.32.x
   websockets==13.x
   numpy==2.x
   torch (compartido con Agente 2)
   ```
   ```
   # Dockerfile.frontend
   FROM node:20-alpine
   WORKDIR /app
   COPY frontend/package*.json ./
   RUN npm ci
   COPY frontend/ .
   RUN npm run build
   ```
4. Definir variables de entorno y puertos para `docker-compose.yml` de Agente 1:
   - `BACKEND_PORT=8000`, `FRONTEND_PORT=3000`, `TAU_THRESHOLD=<valor_calibrado>`

**Artefactos de salida:**
- `frontend/components/DiagConsole.tsx`
- `requirements_agent3.txt`
- `Dockerfile.frontend`
- `config/docker_env_spec.md` (para Agente 1)

---

### Día 6 — Demo-Mode con Controles Interactivos

**Objetivo:** El jurado puede inyectar ruido o desconexiones en vivo y ver la respuesta del agente.

**Tareas:**
1. Implementar en backend el endpoint de control `/control` WebSocket:
   - Comando `inject_disconnect`: desconecta canal X por N segundos
   - Comando `inject_noise`: agrega ruido gaussiano de amplitud configurable a canal X
   - Comando `inject_emi`: agrega interferencia 60Hz a todos los canales
   - Comando `set_tau`: ajusta τ en tiempo real
2. Implementar en frontend panel de control interactivo:
   - Sliders: nivel de ruido, duración de desconexión, valor de τ
   - Botones: "Desconectar Canal X", "Inyectar EMI", "Reset"
3. Preparar escenarios pre-validados con ventanas de PhysioNet donde el agente funciona correctamente:
   - Escenario A: desconexión de electrodo → reconstrucción exitosa (baja varianza → INTEGRATING)
   - Escenario B: artefacto severo → reconstrucción fallida (alta varianza → ALERTING)
   - Escenario C: falla en múltiples canales → alerta obligatoria

**Validación crítica:**
- El jurado puede disparar los tres escenarios en < 30 segundos desde la UI
- El agente responde consistentemente — ❌ NO presentar demostración si el comportamiento es errático

**Artefactos de salida:**
- `backend/demo_controller.py`
- `frontend/components/DemoPanel.tsx`

---

### Día 7 — Congelamiento, Degradación de Red y Capturas

**Objetivo:** Frontend inquebrantable ante condiciones hostiles; grabaciones de respaldo listas.

**Tareas:**
1. Congelar versiones en `package.json` y `package-lock.json`:
   - `"next": "15.x.x"` — versión exacta
   - `"webgl-plot": "x.x.x"` — versión exacta
2. Pruebas de degradación de red:
   - Simular latencia 200ms, 500ms entre backend y frontend
   - Simular pérdida de paquetes 5%
   - Verificar que el buffer circular del frontend no colapsa — implementar reconexión automática
3. Grabar screen-captures en 1080p60 de los tres escenarios Demo-Mode
4. Preparar fallback offline: si el backend no responde, frontend muestra última señal congelada con mensaje claro

**Artefactos de salida:**
- `package.json` con versiones congeladas
- `frontend/lib/ws_client.ts` con lógica de reconexión (exponential backoff)
- Capturas de video en `demo_captures/`

---

## ESTRUCTURA DE ARCHIVOS ESPERADA

```
AMD_PROJECT/
├── Memory_Agent3.md          ← este archivo
├── backend/
│   ├── main.py               ← FastAPI app + WebSocket handlers
│   ├── agent_fsm.py          ← Máquina de estados del agente
│   ├── heuristics.py         ← Detectores de integridad de señal
│   ├── inference_bridge.py   ← Adaptador async hacia Agente 2
│   ├── synthetic_emitter.py  ← Generador de señal sintética (fallback Día 1-2)
│   └── demo_controller.py    ← Controlador de inyección de fallas (Día 6)
├── frontend/
│   ├── pages/
│   │   └── index.tsx         ← Dashboard principal
│   ├── components/
│   │   ├── ECGCanvas.tsx     ← Renderer WebGL de 12 canales
│   │   ├── UncertaintyBands.tsx ← Bandas de confianza WebGL
│   │   ├── DiagConsole.tsx   ← Log de eventos del agente
│   │   └── DemoPanel.tsx     ← Controles interactivos (Día 6)
│   ├── lib/
│   │   └── ws_client.ts      ← Cliente WebSocket con reconexión
│   └── package.json          ← Versiones congeladas (Día 7)
├── config/
│   ├── agent_config.yaml     ← TAU, timeouts, thresholds
│   └── docker_env_spec.md    ← Especificaciones para Agente 1 (Día 5)
├── requirements_agent3.txt   ← Dependencias Python pinneadas (para Agente 1)
└── Dockerfile.frontend       ← Build de Next.js (para Agente 1)
```

---

## PROTOCOLO DE MENSAJES WEBSOCKET

### Server → Browser (stream de datos)

```json
// Frame de señal normal
{"type": "ecg_frame", "ts": 1234567890.123, "data": [[...12 canales × N muestras...]], "state": "MONITORING"}

// Evento: corrupción detectada
{"type": "corruption_detected", "channel": 2, "ts": 1234567891.0, "corruption_type": "disconnect"}

// Evento: segmento reconstruido (Día 3)
{"type": "reconstructed_segment", "channel": 2, "start_ts": 1234567891.0, "samples": [...], "color": "cyan"}

// Evento: alerta (Día 4)
{"type": "alert", "channel": 2, "ts": 1234567892.0, "variance": 0.87, "tau": 0.5, "message": "Reconstrucción rechazada — varianza supera τ"}

// Evento: parche integrado silenciosamente (Día 4)
{"type": "patch_integrated", "channel": 2, "variance": 0.12, "tau": 0.5}
```

> ⚙️ Formato binario float32 vs JSON: PENDIENTE coordinar con Agente 1 el Día 1.
> Para datos ECG en streaming (alto throughput), binary es preferible — ~10× menor overhead.
> Para mensajes de control y eventos, JSON es preferible — legibilidad de debug.
> Propuesta final: **binary float32 para `ecg_frame`**, **JSON para todos los demás tipos**.

### Browser → Server (control)

```json
{"type": "inject_disconnect", "channel": 2, "duration_s": 3.0}
{"type": "inject_noise", "channel": 5, "amplitude": 0.5}
{"type": "inject_emi", "frequency_hz": 60}
{"type": "set_tau", "value": 0.4}
{"type": "reset"}
```

---

## RESTRICCIONES Y PRINCIPIOS OPERATIVOS

1. **Ejecutor GPU singleton, max_workers=1:** Toda llamada a GPU (Agente 2) usa `loop.run_in_executor(_GPU_EXECUTOR, ...)` donde `_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1)` es un singleton a nivel de módulo en `inference_bridge.py`. Nunca usar `run_in_executor(None, ...)` para GPU — el ejecutor por defecto tiene múltiples workers y permite que varios hilos llamen a PyTorch simultáneamente. Agente 2 usa `torch.manual_seed()` dentro del ensemble, que muta estado global: con múltiples hilos es una race condition real.
2. **τ es calibrado, no hardcodeado:** El valor de τ en `agent_config.yaml` se determina experimentalmente el Día 4. No usar ningún valor arbitrario en el código.
3. **Backpressure explícito:** Si el browser no consume frames a tiempo, el servidor descarta — nunca acumula cola ilimitada.
4. **Mock primero, real después:** El Día 2 ya debe funcionar el flujo completo con mock de Agente 2. El Día 3 solo se reemplaza el mock. Esto aísla los bugs de integración.
5. **WebGL puro para rendimiento:** Ningún elemento de datos (trazas, bandas, segmentos) se renderiza con SVG o Canvas 2D. Solo controles de UI (botones, sliders, texto) pueden usar el DOM.
6. **Sin PHI en el frontend:** Los datos MIMIC-III llegan al browser solo como arrays numéricos. Ningún identificador de paciente viaja por WebSocket. Verificar con Agente 1 que `simulate_stream.py` ya los elimina.
7. **Versiones congeladas desde Día 1 en Python:** `requirements_agent3.txt` con versiones fijas desde el inicio — no actualizar durante el sprint.

---

## COSAS QUE NO ASUMIR

- ❌ No asumir que `reconstruct()` de Agente 2 es thread-safe — confirmar antes de usar `run_in_executor`
- ❌ No asumir latencia de inferencia — medir el Día 3 antes de fijar timeouts
- ❌ No asumir formato del stream de Agente 1 — coordinar explícitamente el Día 1
- ❌ No asumir que el modelo de Agente 2 maneja W variable — puede requerir siempre W=5000
- ❌ No asumir unidades de varianza — dependen de normalización de Agente 1; τ se calibra en las mismas unidades
- ❌ No asumir que webgl-plot soporta triangle strips para bandas — puede necesitar WebGL custom

---

## ESTADO ACTUAL

| Día | Estado    | Bloqueado por                                      |
|-----|-----------|----------------------------------------------------|
| 1   | PENDIENTE | Nada — arranque independiente posible              |
| 2   | PENDIENTE | Completar Día 1                                    |
| 3   | PENDIENTE | Agente 2 entregue `reconstruct()`                  |
| 4   | PENDIENTE | Agente 2 entregue `reconstruct_with_uncertainty()` |
| 5   | PENDIENTE | Días 1-4 + Agente 1 Docker integration             |
| 6   | PENDIENTE | Días 1-5 completos                                 |
| 7   | PENDIENTE | Día 6 completo                                     |

---

## LOG DE ITERACIONES (actualizar en cada sesión)

| Fecha      | Cambio / Descubrimiento |
|------------|-------------------------|
| 2026-04-25 | Archivo creado. Contratos de interfaz leídos de Memory_Agent1.md y Memory_Agent2.md. Dependencias críticas identificadas: formato de stream (Agente 1, Día 1), thread-safety de inferencia (Agente 2, Día 3), latencia real de reconstrucción (Agente 2, Día 3). Umbral τ es territorio exclusivo de este agente — calibrar Día 4. |
| 2026-04-25 | Corrección crítica de concurrencia: `run_in_executor(None, ...)` reemplazado por `_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1)` singleton. Motivo: Agente 2 usa `torch.manual_seed()` en el ensemble → race condition con múltiples clientes WebSocket concurrentes. El ejecutor de un solo hilo serializa las llamadas GPU sin bloquear el event loop. Pregunta abierta sobre thread-safety cerrada. |
| 2026-04-25 | Corrección de contradicción en Detector 3 (FFT): el código usaba `np.fft` (CPU) mientras la regla decía `torch.fft` (GPU). Decisión fijada: OPCIÓN B — `torch.fft.rfft` sobre tensor GPU. Código de ejemplo reemplazado completamente para ser consistente. Añadida contingencia para frames que lleguen como bytes vía red. |
| 2026-04-25 | [AGENTE JEFE] Protocolo de stream Agente 1→Agente 3 acordado y registrado. Tabla de inputs actualizada: BINARY opcode float32 (12,25)=1200 bytes/frame, TEXT opcode JSON para control, CHUNK_SIZE=25 canónico en stream_config.py, simulate_stream.py mismo proceso vía asyncio.Queue. Preguntas abiertas de Agente 1 cerradas. Bloqueador Día 1 eliminado. |
| 2026-04-25 | [AGENTE JEFE] NUEVO RIESGO detectado: asyncio.Queue sin maxsize en simulate_stream.py (lado Agente 1). Si browser consume lento, cola crece sin límite — viola Principio 3 de este agente. Prompt de corrección emitido a Agente 1: usar Queue(maxsize=40) con put_nowait() y drop-on-full. |
| 2026-04-25 | [AGENTE JEFE] Issue A CERRADO. Agente 1 implementó QUEUE_MAXSIZE=40 en config/stream_config.py (constante canónica compartida), simulate_stream.py con put_nowait()+drop del frame nuevo, y _drop_count expuesto para telemetría Día 4. Protocolo de stream Agente1→Agente3 completamente acordado e implementado. |

---

*Última actualización: 2026-04-25 | Agente 3 — Ingeniero de Agente Autónomo y Full-Stack*
