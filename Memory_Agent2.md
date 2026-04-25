# Memory_Agent2 — Ingeniero de Matemática y Modelado
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

- **Rol:** Ingeniero B — Matemática y Modelado
- **Propietario de:** Arquitectura del modelo, entrenamiento, inferencia generativa, cuantificación de incertidumbre, manuscrito IEEE
- **NO soy responsable de:** infraestructura ROCm (Agente 1), servidor FastAPI/WebSocket/frontend (Agente 3)

---

## CANAL DE SOPORTE HUMANO

El usuario (Guillermo) actúa como operador humano del sprint y puede ejecutar
cualquier acción externa que un agente de IA no puede realizar directamente.

**Puedes solicitarle ayuda para:**
- Instalar y verificar **PyTorch con backend ROCm** en la instancia (versión exacta, comando pip, test de `torch.cuda.is_available()`)
- Configurar el **entorno Python / conda / venv** en la instancia remota
- Acceder y descargar registros de **MIMIC-III desde PhysioNet** (credenciales, aceptación de licencia, comandos de descarga)
- Crear una cuenta o proyecto en **Overleaf** para el manuscrito IEEE, o instalar LaTeX localmente
- Gestionar **credenciales y tokens** necesarios para acceder a servicios externos (IEEE Xplore API, etc.)
- Ejecutar comandos de **verificación de hardware** en la instancia (`rocminfo`, `rocm-smi`, `hipinfo`)
- Cualquier operación que requiera **intervención manual en la nube o en sistemas de archivos remotos**

**Regla obligatoria — Instrucciones Paso a Paso:**
Cuando necesites la intervención del usuario, debes entregarle una guía completa con:

1. **Objetivo** — qué se logrará con estos pasos y por qué es necesario
2. **Prerrequisitos** — qué debe tener listo antes de empezar
3. **Pasos numerados** — cada comando exacto, cada campo de formulario, cada opción de menú
4. **Verificación** — cómo confirmar que el paso fue exitoso antes de continuar al siguiente
5. **Qué reportar de vuelta** — la salida o dato exacto que necesitas que te comunique

> NO emitas instrucciones vagas como "instala PyTorch con ROCm".
> Emite instrucciones ejecutables, por ejemplo:
> "Ejecuta: `pip install torch==2.10.0+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2`
>  luego comparte la salida de `python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"`"

---

## CONTRATO DE DATOS — INTERFACES DE ACOPLE

### LO QUE RECIBO (inputs que no controlo)

| Fuente    | Día | Artefacto                                | Shape / Tipo             | Estado    |
|-----------|-----|------------------------------------------|--------------------------|-----------|
| Agente 1  | 1   | DataLoader GPU listo                     | `(batch, 12, 5000)` f32/bf16 | ✅ definido |
| Agente 1  | 2   | Tensores con corrupción sintética        | mismo shape, canal(es) corrompidos | ✅ definido |
| Agente 3  | 3+  | Tensor de señal interceptada (corrupción real) | `(1, 12, W)` W ≤ 5000 | ✅ acordado |

> **Regla crítica:** Nunca asumir shape distinto a `(batch, 12, 5000)` sin confirmación escrita de Agente 1.
> Si W < 5000 (ventana parcial), el modelo debe manejar padding explícito — no asumir relleno automático.

### LO QUE ENTREGO (outputs de los que soy responsable)

| Destino   | Día | Artefacto                                | Shape / Tipo             | Estado    |
|-----------|-----|------------------------------------------|--------------------------|-----------|
| Agente 1  | 3   | Módulo modelo para profiling ROCm        | `model.py` + `inference.py` | PENDIENTE |
| Agente 1  | 4   | Requerimientos de VRAM para ensemble     | documento numérico       | PENDIENTE |
| Agente 3  | 3   | Función `reconstruct(signal, mask)` → tensor | `(1, 1, W)` f32     | PENDIENTE |
| Agente 3  | 4   | Función `reconstruct_with_uncertainty()` → [mean, var] | `(1,1,W)` + `(1,1,W)` | PENDIENTE |

> **Regla crítica:** La entrega a Agente 3 debe ser una función con firma estable.
> Cambiar la firma requiere notificación explícita — Agente 3 depende de ella para su lógica de decisión.

> **Padding interno (transparente para Agente 3):** Agente 3 puede enviar W < 5000 (ventana parcial).
> El padding a 5000 y el recorte de la salida son **responsabilidad exclusiva de `inference.py`**.
> Invariante garantizado: `output.shape[-1] == signal.shape[-1]` — misma W que la entrada, sin padding residual.
> ❌ Agente 3 nunca debe hacer padding por su cuenta ni conocer el detalle de implementación.

---

## STACK TÉCNICO — VERIFICADO

| Componente            | Tecnología / Especificación                         | Estado  |
|-----------------------|-----------------------------------------------------|---------|
| Framework             | PyTorch 2.10 sobre ROCm 7.2.0                       | ✅      |
| Precisión de entrenamiento | BF16 (bfloat16)                                | ✅      |
| Arquitectura          | U-Net 1D con convolución causal dilatada            | ✅      |
| Proceso estocástico   | VP-SDE (Variance Preserving SDE)                    | ✅      |
| Función de pérdida    | Score-matching (denoising score matching)           | ✅      |
| Solucionador inverso  | Euler-Maruyama (predictor) + Langevin (corrector)   | ✅      |
| Cuantificación incertidumbre | Ensemble de semillas estocásticas (primario) | ✅      |
| Cuantificación incertidumbre | FLARE (Fisher-Laplace Approx.) (secundario) | ⚙️ confirmar paper fuente |
| Métricas de validación | MMD, DTW vs. baselines deterministas              | ✅      |
| Publicación objetivo  | IEEE BioCAS / TBioCAS                               | ✅      |

---

## MATEMÁTICA DEL MODELO — VERIFICADA

### VP-SDE (Variance Preserving)

Ecuación directa (perturbación):
```
dx = -½ β(t) x dt + √β(t) dW
```
- `f(x,t) = -½ β(t) x`  ← coeficiente de deriva
- `g(t) = √β(t)`         ← coeficiente de difusión
- `β(t)`: schedule de ruido ⚙️ **PENDIENTE CALIBRAR** (lineal o coseno — evaluar ambos)

Solución marginal (distribución en tiempo t dado x₀):
```
p(x_t | x_0) = N(x_t; √ᾱ(t) x_0, (1 - ᾱ(t)) I)
donde ᾱ(t) = exp(-½ ∫₀ᵗ β(s)ds)
```
✅ Esta fórmula es matemáticamente exacta para VP-SDE.

### Score-Matching Loss (Denoising)

```python
# Conceptualmente:
loss = E_t E_{x0} E_{eps} [ || s_theta(x_t, t, context) - score_target ||² ]
# donde score_target = -eps / sqrt(1 - alpha_bar(t))
# y x_t = sqrt(alpha_bar(t)) * x0 + sqrt(1 - alpha_bar(t)) * eps
# eps ~ N(0, I)
```
✅ Esta es la forma estándar de denoising score matching (Song et al., 2020).

### Solucionador Inverso: Predictor-Corrector

**Predictor (Euler-Maruyama, un paso)**:
```
x_{t-Δt} = x_t + [f(x_t,t) - g(t)² · s_θ(x_t,t)] · (-Δt) + g(t)·√Δt · z
z ~ N(0, I)
```

**Corrector (Langevin, M pasos por nivel de ruido)**:
```
x ← x + ε · s_θ(x, t) + √(2ε) · z
z ~ N(0, I)
```
- `ε`: step size del corrector ⚙️ **PENDIENTE CALIBRAR**
- `M`: número de pasos Langevin por nivel ⚙️ **PENDIENTE CALIBRAR** (típico: 1-5)
- Número total de steps de discretización `T` ⚙️ **PENDIENTE CALIBRAR** (típico: 100-1000)

✅ Este esquema es el PC sampler de Song et al. (2021) "Score-Based Generative Modeling through SDEs".

### Condicionamiento Cruzado (11 canales → 1 canal dañado)

```
# El canal dañado recibe condicionamiento de los 11 canales sanos
# Implementación: cross-attention o concatenación en el espacio latente
input_model = concat([corrupted_channel, healthy_channels * mask], dim=1)
# Shape: (batch, 12, 5000) donde el canal dañado está enmascarado
```
- La máscara `mask` proviene de Agente 3 (indica qué canal está corrupto)
- ❌ **NO ASUMIR** que siempre es 1 canal corrupto — puede ser 1-3 simultáneos

### Cuantificación de Incertidumbre: Ensemble de Semillas

```python
# Método primario (ensemble estocástico)
reconstructions = []
for seed in range(N_ENSEMBLE):  # N_ENSEMBLE ⚙️ PENDIENTE (típico: 10-50)
    torch.manual_seed(seed)
    rec = solver.sample(signal, mask, context)  # (1, 1, W)
    reconstructions.append(rec)

stack = torch.stack(reconstructions)  # (N, 1, 1, W)
mean = stack.mean(dim=0)              # (1, 1, W) → entrega a Agente 3
variance = stack.var(dim=0)           # (1, 1, W) → entrega a Agente 3
```
✅ Este es el método más robusto y menos propenso a errores de implementación.

> **VRAM warning:** N_ENSEMBLE × tamaño del modelo × batch = puede saturar MI300X.
> Coordinar con Agente 1 antes de fijar N_ENSEMBLE. Alternativa: inferencia secuencial vs. paralela.

---

## ARQUITECTURA U-NET 1D — ESPECIFICACIONES

### Estructura General

```
Encoder (downsampling):
  ConvBlock(12, 64, k=7, dil=1)   → (batch, 64, 5000)
  ConvBlock(64, 128, k=7, dil=2)  → (batch, 128, 2500)
  ConvBlock(128, 256, k=7, dil=4) → (batch, 256, 1250)
  ConvBlock(256, 512, k=7, dil=8) → (batch, 512, 625)

Bottleneck:
  ConvBlock(512, 512, k=7, dil=16) → (batch, 512, 625)
  [Aquí entra el condicionamiento cross-channel]

Decoder (upsampling con skip connections):
  UpConv(512+512, 256) → (batch, 256, 1250)
  UpConv(256+256, 128) → (batch, 128, 2500)
  UpConv(128+128, 64)  → (batch, 64, 5000)
  Conv(64, 1, k=1)     → (batch, 1, 5000)  ← canal reconstruido
```

> ⚙️ **PENDIENTE CALIBRAR:** Número exacto de capas, canales, y dilataciones.
> La estructura arriba es un punto de partida razonable — ajustar según VRAM disponible.

### Convolución Causal vs. Estándar

- **Causal:** el output en tiempo `t` no ve el futuro (`t+1, t+2...`)
- **Para reconstrucción clínica:** causalidad NO es obligatoria si tenemos acceso a la ventana completa
- ✅ Usar convoluciones estándar (no causales) durante entrenamiento — más fácil de optimizar
- ❌ **NO implementar causalidad** sin confirmar con Agente 3 si el sistema opera en streaming puro

### Condicionamiento de Tiempo t (difusión)

```python
# Embedding del timestep t (Transformer-style sinusoidal o aprendido)
t_emb = timestep_embedding(t, dim=256)   # (batch, 256)
# Se inyecta en cada bloque del U-Net via FiLM o adición
```
✅ Estándar en modelos de difusión (DDPM, Score-SDE).

---

## PLAN DE EJECUCIÓN — SPRINT 7 DÍAS

### Día 1 — Arquitectura Base y Verificación de Forward Pass

**Objetivo:** U-Net inicializado, pases directos sin errores en MI300X.

**Tareas concretas:**
1. Implementar bloques `ResBlock1D` con dilatación configurable
2. Implementar `UNet1D` con encoder, bottleneck, decoder
3. Implementar `VPSDECoefficients`: `beta(t)`, `alpha_bar(t)`, `marginal_prob(x0, t)`
4. Probar forward pass:
   ```python
   x = torch.randn(4, 12, 5000, dtype=torch.bfloat16).to('cuda')
   t = torch.rand(4).to('cuda')
   out = model(x, t)
   assert out.shape == (4, 1, 5000)
   ```

**Validación crítica:**
- Sin errores de OOM en MI300X con batch=4, BF16
- Shape de salida exactamente `(batch, 1, 5000)`
- Agente 1 puede cargar el módulo desde `model/unet1d.py` sin ImportError

**Artefactos de salida:**
- `model/unet1d.py`
- `model/vpsde.py`
- `tests/test_forward_pass.py`

---

### Día 2 — Loss de Score-Matching y Entrenamiento BF16

**Objetivo:** Pérdida empírica descendiendo; red asimila distribución QRS.

**Tareas concretas:**
1. Implementar `score_matching_loss(model, x0, context_mask)`:
   - Samplear `t ~ Uniform(t_min, t_max)` — ⚙️ `t_min=1e-5`, `t_max=1.0`
   - Samplear `eps ~ N(0,I)`
   - Computar `x_t = sqrt(alpha_bar(t)) * x0 + sqrt(1-alpha_bar(t)) * eps`
   - Loss = `||model(x_t, t, context) - (-eps/sqrt(1-alpha_bar(t)))||²`
2. Implementar condicionamiento cruzado:
   - `context`: los 11 canales sanos, shape `(batch, 11, 5000)`
   - Concatenar con canal dañado enmascarado antes del encoder
3. Iniciar loop de entrenamiento con `torch.optim.AdamW`, `lr=2e-4`
4. Habilitar BF16 vía `torch.autocast('cuda', dtype=torch.bfloat16)`

**Validación:**
- `loss` desciende monótonamente en primeras 100 iteraciones
- No hay NaN/Inf en gradientes — si aparecen: reducir `lr` o revisar `t_min`
- ❌ **NO declarar éxito** si la pérdida oscila sin descender claro

**Artefactos de salida:**
- `training/loss.py`
- `training/train.py`
- `checkpoints/day2_checkpoint.pt`

---

### Día 3 — Solucionador SDE Inverso e Integración con Agente 3

**Objetivo:** Inferencia generativa funcional; Agente 3 puede llamar `reconstruct()`.

**Tareas concretas:**
1. Implementar `PCsampler` (Predictor-Corrector):
   ```python
   class PCsampler:
       def sample(self, shape, context, mask, T=500, M=1):
           # autocast cubre las activaciones internas del U-Net (BF16)
           # el acumulador x permanece en f32 — comportamiento intencionado de autocast
           with torch.autocast('cuda', dtype=torch.bfloat16):
               x = torch.randn(shape, device='cuda')  # f32, pero ops del modelo en BF16
               for t in reversed(range(1, T + 1)):
                   # Predictor: Euler-Maruyama step
                   x = self.euler_maruyama_step(x, t, context, mask)
                   # Corrector: M pasos Langevin
                   for _ in range(M):
                       x = self.langevin_step(x, t, context, mask)
           return x.float()  # garantizar f32 a Agente 3; no-op si x ya es f32
   ```

   > **Matiz VRAM (reportar a Agente 1 con precisión):**
   > - `autocast` ahorra VRAM en **activaciones intermedias** del U-Net dentro del bucle — el cuello de botella real.
   > - Los **pesos del modelo permanecen en f32** (master weights del AdamW). Para ahorrar VRAM en pesos haría falta `model.to(torch.bfloat16)` — NO hacer sin coordinación con Agente 1.
   > - El acumulador `x` es f32 y su shape `(1,1,W)` es trivial — no contribuye al problema de VRAM.

2. Calibrar T y M:
   - Probar T=100, T=500, T=1000 — medir calidad visual vs. latencia
   - Objetivo latencia: < 2 segundos por reconstrucción en MI300X
3. Exponer función pública para Agente 3 — padding, autocast y recorte incluidos internamente:
   ```python
   def reconstruct(signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
       """
       signal: (1, 12, W), mask: (1, 12, W) — W puede ser <= 5000
       returns: (1, 1, W) — misma longitud W que la entrada, sin padding residual
       """
       W_original = signal.shape[-1]

       if W_original < 5000:
           pad_len = 5000 - W_original
           signal = torch.nn.functional.pad(signal, (0, pad_len))  # zeros al final
           mask   = torch.nn.functional.pad(mask,   (0, pad_len))

       with torch.autocast('cuda', dtype=torch.bfloat16):
           output = self.sampler.sample((1, 1, 5000), context=signal, mask=mask)

       return output[..., :W_original].float()  # recortar al W original; garantizar f32
   ```

   > **Por qué `(1, 1, 5000)` como shape del sampler:** el modelo reconstruye 1 canal a lo largo de
   > los 5000 puntos de la ventana completa. El shape del acumulador de ruido inicial es el de la salida.
   > Pasar `signal.shape[-1:]` (= `(W,)`) era un bug doble: shape incorrecto Y sin padding.

**Entrega a Agente 1:**
- `inference/inference.py` con forward pass listo para profiling ROCm

**Artefactos de salida:**
- `inference/sampler.py`
- `inference/inference.py`
- `tests/test_reconstruction_visual.py` (genera gráficas para validación visual)

---

### Día 4 — Cuantificación de Incertidumbre y Coordinación VRAM

**Objetivo:** Salida `[mean, variance]` lista; ensemble no satura VRAM.

**Tareas concretas:**
1. Implementar `reconstruct_with_uncertainty(signal, mask, N=20)`:
   - **Mismo wrapper de padding que `reconstruct()`** — guardar `W_original`, pad a 5000, recortar salida.
   - N inferencias con semillas distintas sobre el tensor ya paddeado `(1, 12, 5000)`
   - Calcular media y varianza por punto sobre el stack recortado a `W_original`
   - Retornar `(mean, variance)` ambos shape `(1, 1, W)` — misma W que la entrada
   ```python
   def reconstruct_with_uncertainty(signal, mask, N=20):
       W_original = signal.shape[-1]
       if W_original < 5000:
           pad_len = 5000 - W_original
           signal = torch.nn.functional.pad(signal, (0, pad_len))
           mask   = torch.nn.functional.pad(mask,   (0, pad_len))

       reconstructions = []
       for seed in range(N):
           torch.manual_seed(seed)
           with torch.autocast('cuda', dtype=torch.bfloat16):
               rec = self.sampler.sample((1, 1, 5000), context=signal, mask=mask)
           reconstructions.append(rec[..., :W_original].float())  # recortar aquí

       stack    = torch.stack(reconstructions)   # (N, 1, 1, W_original)
       mean     = stack.mean(dim=0)              # (1, 1, W_original)
       variance = stack.var(dim=0)               # (1, 1, W_original)
       return mean, variance
   ```
2. Medir uso de VRAM para N=10, 20, 50:
   ```python
   # Reportar a Agente 1:
   torch.cuda.memory_allocated() / 1e9  # GB
   torch.cuda.max_memory_allocated() / 1e9
   ```
3. Coordinar con Agente 1: si VRAM < umbral, aumentar N; si VRAM satura, reducir N o usar inferencia secuencial
4. Exponer contrato final a Agente 3:
   ```python
   # Agente 3 espera exactamente esta firma:
   mean, variance = reconstruct_with_uncertainty(signal, mask, N=N_ENSEMBLE)
   # mean:     (1, 1, W) — señal imputada
   # variance: (1, 1, W) — incertidumbre por punto temporal
   ```

**⚙️ PENDIENTE DE CALIBRAR (no asumir valores):**
- N_ENSEMBLE óptimo: determinar después de medir VRAM
- Umbral τ para la lógica de Agente 3: es RESPONSABILIDAD DE AGENTE 3 — no hardcodear aquí

**Artefactos de salida:**
- `inference/uncertainty.py`
- `reports/vram_ensemble_report.md` (tabla N vs. VRAM vs. latencia)

---

### Día 5 — Validación contra Patologías MIMIC-III

**Objetivo:** Métricas MMD y DTW que demuestran superioridad vs. baselines.

**Tareas concretas:**
1. Preparar test-set separado (registros MIMIC-III NO vistos en entrenamiento):
   - Incluir: bloqueos AV, fibrilación auricular, taquicardia ventricular
   - ❌ **NO usar registros del train-set para reportar métricas**
2. Implementar baselines deterministas para comparación:
   - Interpolación lineal
   - Spline cúbica
   - Media de canales vecinos
3. Calcular métricas:
   ```python
   # MMD (Maximum Mean Discrepancy) — mide distancia distribucional
   # Usar kernel RBF con bandwidth = mediana de distancias
   from sklearn.metrics.pairwise import rbf_kernel
   
   # DTW (Dynamic Time Warping) — mide similitud morfológica
   from dtaidistance import dtw
   ```
4. Tabla comparativa: Modelo Difusión vs. Interpolación vs. Spline

**✅ Métricas verificadas:**
- MMD más bajo = distribución generada más cercana a la real
- DTW más bajo = morfología más similar
- ❌ **NO reportar MSE como métrica principal** — es inadecuado para señales estocásticas

**Artefactos de salida:**
- `evaluation/metrics.py`
- `evaluation/evaluate.py`
- `reports/metrics_table.md`

---

### Día 6 — Ablaciones y Manuscrito IEEE

**Objetivo:** Secciones matemáticas del paper redactadas; ablaciones documentadas.

**Tareas concretas:**

**Ablaciones (comparar métodos de incertidumbre):**
1. Ensemble estocástico de semillas (método primario implementado)
2. Monte Carlo Dropout (insertar `nn.Dropout(p=0.1)` en inference mode)
3. FLARE/Laplace approx. (si hay tiempo — ❌ no priorizar si retrasa el demo)

**Secciones IEEE a redactar:**
- `§ Método 2.1` — Formulación VP-SDE y función de puntuación
- `§ Método 2.2` — Arquitectura U-Net 1D con condicionamiento cruzado
- `§ Método 2.3` — Solucionador PC y esquema de discretización
- `§ Método 2.4` — Cuantificación de incertidumbre epistémica
- `§ Experimentos 3` — Descripción de MIMIC-III, splits, baselines
- `§ Resultados 4` — Tabla MMD/DTW, figuras de reconstrucción

**Plantilla IEEE a usar:** `IEEEtran.cls` — ✅ disponible en Overleaf y IEEE Author Tools

**Artefactos de salida:**
- `paper/sections/method.tex`
- `paper/sections/experiments.tex`
- `paper/figures/` (imágenes generadas en Día 5)
- `reports/ablation_study.md`

---

### Día 7 — Manuscrito Final y Verificación de Novedad

**Objetivo:** Draft IEEE limpio listo para sometimiento; código congelado.

**Tareas concretas:**
1. Integrar todas las secciones en `paper/main.tex` con formato IEEEtran
2. Verificar checklist IEEE:
   - [ ] Sin conflicto de interés declarado
   - [ ] Uso de MIMIC-III citado correctamente (PhysioNet, versión, acceso)
   - [ ] Todas las figuras con permisos / generadas por nosotros
   - [ ] Citas con formato BibTeX correcto
3. Verificar novedad algorítmica:
   - Buscar en IEEE Xplore: "ECG reconstruction diffusion model uncertainty"
   - ❌ **NO declarar novedad sin búsqueda real** — el jurado puede preguntar
4. Congelar pesos del modelo:
   ```bash
   # Guardar checkpoint final con metadatos
   torch.save({'model': model.state_dict(), 'config': config, 'day': 7}, 'checkpoints/final_model.pt')
   ```

**Artefactos de salida:**
- `paper/main.tex`
- `paper/references.bib`
- `checkpoints/final_model.pt`

---

## ESTRUCTURA DE ARCHIVOS ESPERADA

```
AMD_PROJECT/
├── Memory_Agent1.md
├── Memory_Agent2.md          ← este archivo
├── model/
│   ├── unet1d.py             ← Arquitectura U-Net 1D
│   └── vpsde.py              ← Coeficientes VP-SDE
├── training/
│   ├── loss.py               ← Score-matching loss
│   └── train.py              ← Loop de entrenamiento BF16
├── inference/
│   ├── sampler.py            ← PC sampler (Euler-Maruyama + Langevin)
│   ├── inference.py          ← reconstruct(signal, mask)
│   └── uncertainty.py        ← reconstruct_with_uncertainty(...)
├── evaluation/
│   ├── metrics.py            ← MMD, DTW
│   └── evaluate.py           ← Evaluación en test-set
├── checkpoints/
│   ├── day2_checkpoint.pt
│   └── final_model.pt
├── tests/
│   ├── test_forward_pass.py
│   └── test_reconstruction_visual.py
├── reports/
│   ├── metrics_table.md
│   ├── ablation_study.md
│   └── vram_ensemble_report.md
└── paper/
    ├── main.tex
    ├── references.bib
    ├── sections/
    │   ├── method.tex
    │   └── experiments.tex
    └── figures/
```

---

## RESTRICCIONES Y PRINCIPIOS OPERATIVOS

1. **Shape canónico:** Siempre `(batch, 12, 5000)` — coordinar con Agente 1 antes de cualquier cambio.
2. **BF16 en todo el ciclo de cómputo:** Training y inference usan `torch.autocast('cuda', dtype=torch.bfloat16)`. Si hay NaN, bajar lr antes de cambiar a fp32.
3. **autocast en inferencia es obligatorio:** `PCsampler.sample()` y `reconstruct()` deben envolver su lógica en `torch.autocast`. Omitirlo hace que las activaciones del U-Net corran en f32 — 2× VRAM y throughput reducido. ✅ Corregido 2026-04-25.
4. **Salida a Agente 3 siempre en f32:** `return x.float()` al final de `sample()` y `reconstruct()`. El browser no puede parsear BF16 nativo sin conversión adicional.
5. **Reporte de VRAM a Agente 1 debe ser preciso:** Los pesos del modelo son f32 (master weights del AdamW). El ahorro de BF16 vía autocast aplica a **activaciones intermedias**, no a pesos. No reportar ahorro de pesos a menos que se haga `model.to(torch.bfloat16)` explícitamente — decisión que requiere coordinación con Agente 1.
6. **Contrato de incertidumbre:** La salida `(mean, variance)` es el contrato con Agente 3. El umbral τ es territorio de Agente 3 — no fijar valores de τ en este código.
7. **Test-set limpio:** Nunca reportar métricas sobre datos que el modelo vio en entrenamiento.
8. **VRAM del ensemble:** Coordinar con Agente 1 el N_ENSEMBLE antes de reportar a Agente 3.
9. **Sin alucinaciones de papers:** Solo citar papers que se han leído. Citar siempre:
   - Song et al. (2021) "Score-Based Generative Modeling through SDEs" — fuente de VP-SDE y PC sampler
   - Ho et al. (2020) "Denoising Diffusion Probabilistic Models" — denoising score matching
   - Verificar DOI antes de incluir en BibTeX.

---

## REFERENCIAS TÉCNICAS VERIFICADAS

| Papel | Fuente en este Proyecto |
|-------|------------------------|
| Song et al. 2021 "Score-Based Generative Modeling through SDEs" (arXiv:2011.13456) | VP-SDE, PC sampler |
| Ho et al. 2020 "DDPM" (NeurIPS 2020) | Denoising score matching loss |
| Ronneberger et al. 2015 "U-Net" (MICCAI) | Arquitectura base U-Net con skip connections |
| PhysioNet MIMIC-III Clinical Database | Dataset — citar con DOI correcto |

> ❌ **NO citar papers sin verificar DOI y contenido real.** Si un paper no está en esta lista, buscarlo antes de citarlo.

---

## ESTADO ACTUAL

| Día | Estado     | Bloqueado por                           |
|-----|------------|-----------------------------------------|
| 1   | PENDIENTE  | Agente 1 entregue DataLoader (Día 1)    |
| 2   | PENDIENTE  | Completar Día 1                         |
| 3   | PENDIENTE  | Completar Día 2 (modelo entrenado)      |
| 4   | PENDIENTE  | Completar Día 3 + coord. VRAM Agente 1  |
| 5   | PENDIENTE  | Completar Día 2 (pesos entrenados)      |
| 6   | PENDIENTE  | Completar Día 5 (métricas)              |
| 7   | PENDIENTE  | Completar Día 6 (secciones paper)       |

---

## LOG DE ITERACIONES (actualizar en cada sesión)

| Fecha      | Cambio / Descubrimiento |
|------------|-------------------------|
| 2026-04-24 | Archivo creado. Contrato de datos confirmado con Memory_Agent1.md: shape `(batch, 12, 5000)`. |
| 2026-04-25 | Bug corregido: `PCsampler.sample()` y `reconstruct()` carecían de `torch.autocast('cuda', dtype=torch.bfloat16)`. Sin autocast, las activaciones del U-Net corrían en f32 (2× VRAM, menor throughput). Fix: bucle inverso envuelto en autocast; salida convertida con `.float()` para Agente 3. Matiz añadido a Restricciones: autocast ahorra VRAM en activaciones, NO en pesos (f32 master weights del AdamW). |
| 2026-04-25 | Bug corregido: `reconstruct()` no implementaba padding para W < 5000 pese a que el contrato lo requería. Además pasaba `signal.shape[-1:]` (= `(W,)`) como shape al sampler en lugar de `(1, 1, 5000)` — doble error. Fix: wrapper de padding/unpadding en `inference.py` para `reconstruct()` y `reconstruct_with_uncertainty()`. El recorte `output[..., :W_original]` garantiza que la salida devuelve exactamente la W de entrada. Documentado en CONTRATO DE DATOS: padding es responsabilidad interna, transparente para Agente 3. |

---

*Última actualización: 2026-04-24 | Agente 2 — Ingeniero de Matemática y Modelado*
