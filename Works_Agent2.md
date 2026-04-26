# Works_Agent2 — Bitácora de Entregas
# Ingeniero B: Matemática y Modelado
# Última actualización: 2026-04-25

---

## PARA AGENTES 1 Y 3 — Leer antes de integrar

Este archivo documenta todo lo que Agente 2 ha entregado, con rutas exactas,
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
| Workspace | `~/workspace` |

---

## DÍA 1 — ✅ COMPLETO (2026-04-26)

### Archivos entregados

| Archivo | Descripción | Estado |
|---------|-------------|--------|
| `model/__init__.py` | Paquete Python — exporta UNet1D, ResBlock1D, get_timestep_embedding, VPSDECoefficients | ✅ Creado |
| `model/unet1d.py` | UNet1D (4 niveles, bottleneck dil=16, skip connections) + ResBlock1D + get_timestep_embedding | ✅ Creado |
| `model/vpsde.py` | VPSDECoefficients: beta, alpha_bar, marginal_prob, perturb, drift_coeff, diffusion_coeff | ✅ Creado |
| `tests/test_forward_pass.py` | 18 tests + smoke-test con reporte de VRAM real | ✅ Creado |

### Arquitectura implementada (resumen para Agente 1)

```
UNet1D(in_channels=12, base_channels=64, t_emb_dim=256)

Encoder (stride-2 downsampling):
  enc0 : ResBlock1D(12→64,  dil=1)  → (B,  64, 5000)  ← skip0
  enc1 : ResBlock1D(64→128, dil=2)  → (B, 128, 2500)  ← skip1
  enc2 : ResBlock1D(128→256,dil=4)  → (B, 256, 1250)  ← skip2
  enc3 : ResBlock1D(256→512,dil=8)  → (B, 512,  625)

Bottleneck:
  ResBlock1D(512→512, dil=16)       → (B, 512,  625)

Decoder (Upsample nearest + Conv + skip cat):
  dec3 : ResBlock1D(768→256, dil=4) → (B, 256, 1250)
  dec2 : ResBlock1D(384→128, dil=2) → (B, 128, 2500)
  dec1 : ResBlock1D(192→64,  dil=1) → (B,  64, 5000)
  out  : Conv1d(64→1, k=1)          → (B,   1, 5000)

Parámetros estimados: ~20M  (~80 MB en f32 master weights)
```

### Contratos de salida hacia Agente 1 y Agente 3

```python
# Para Agente 1 — profiling ROCm (Día 3):
from model.unet1d import UNet1D
model = UNet1D().to('cuda')
# x: (B, 12, 5000) BF16,  t: (B,) → out: (B, 1, 5000)

# Para Agente 3 — inferencia (Día 3):
# La firma de reconstruct(signal, mask) se implementa en inference/inference.py
# signal: (1, 12, W), mask: (1, 12, W) → output: (1, 1, W) f32
# (W puede ser < 5000; padding interno en inference.py)

# VPSDECoefficients — contrato matemático para loss Día 2:
from model.vpsde import VPSDECoefficients
sde = VPSDECoefficients(beta_min=0.1, beta_max=20.0)
x_t, noise = sde.perturb(x0, t)   # x0:(B,12,5000), t:(B,) → x_t,noise:(B,12,5000)
mean, std  = sde.marginal_prob(x0, t)  # mean:(B,12,5000), std:(B,1,1)
```

### Resultados de tests en hardware real — MI300X (2026-04-26) — COMPLETO

```
────────────────────────────────────────────────────────────
  GPU  : AMD Radeon Graphics
  VRAM : 205.8 GB
  BF16 : ✓
────────────────────────────────────────────────────────────

[OK] UNet1D forward pass shape: (4, 1, 5000)
[OK] UNet1D batch=1 shape: (1, 1, 5000)
[OK] UNet1D sin NaN en output
[OK] UNet1D gradientes fluyen correctamente
[INFO] UNet1D parámetros: 12.97M  (51.9 MB en f32 master weights)
[INFO] VRAM pico forward pass (batch=4, BF16): 0.432 GB  (de 205.8 GB disponibles)
[OK] ResBlock1D same channels (64→64)
[OK] ResBlock1D expansión de canales (12→64)
[OK] ResBlock1D todas las dilations preservan L=5000
[OK] VPSDECoefficients ᾱ(t) rango: [0.0066, 1.0000]
[OK] VPSDECoefficients ᾱ(t) monótonamente decreciente
[OK] VPSDECoefficients marginal_prob shapes: mean(4, 12, 5000) std(4, 1, 1)
[OK] VPSDECoefficients std > 0 para todo t > 0
[OK] VPSDECoefficients perturb shapes correctos
[OK] VPSDECoefficients β(t) rango: [0.10, 20.00]
[OK] get_timestep_embedding shape (4, 256)
[OK] get_timestep_embedding es determinista
[OK] get_timestep_embedding distingue timesteps distintos

════════════════════════════════════════════════════════════
  Día 1 — Agente 2   18 passed, 0 failed
  ✓ Forward pass validado. Listo para Día 2 (score-matching).
════════════════════════════════════════════════════════════
```

### Datos confirmados en hardware real

| Métrica | Valor real | Nota |
|---------|-----------|------|
| GPU     | AMD Radeon Graphics (MI300X en ROCm) | Nombre de driver — hardware correcto |
| VRAM total | 205.8 GB | ✓ |
| BF16 | ✓ | `torch.cuda.is_bf16_supported()` = True |
| Parámetros UNet1D | **12.97 M** | Estimación previa era ~20M — corrección aplicada |
| Pesos f32 master | **51.9 MB** | Muy ligero para 205.8 GB VRAM |
| VRAM pico batch=4 BF16 | **0.432 GB** | Solo el 0.21% de la VRAM disponible |
| ᾱ(t=0) | 1.0000 | señal intacta en t=0 ✓ |
| ᾱ(t=1) | 0.0066 | ≈ruido puro en t=1 ✓ |
| β range | [0.10, 20.00] | schedule lineal correcto ✓ |

### → REPORTE A AGENTE 1 — VRAM para ensemble (Día 4)

```
MODELO:
  Parámetros   : 12.97 M
  Pesos f32    : 51.9 MB
  VRAM pico fwd: 0.432 GB  (batch=4, BF16)
  Por muestra  : ~0.108 GB (batch=1 estimado)

HEADROOM PARA ENSEMBLE (secuencial, N muestras × batch=1):
  VRAM por run : ~0.108 GB  (activaciones + modelo)
  N=20         : ~0.108 GB  (secuencial — reutiliza memoria)
  N=50         : ~0.108 GB  (secuencial — mismo footprint)

  → No hay restricción de VRAM para ningún N razonable.
  → Agente 1 puede autorizar N_ENSEMBLE hasta N=50 sin coordinación adicional.
  → Si se ejecuta en paralelo (batch=N): ~N × 0.108 GB
    N=50 paralelo ≈ 5.4 GB — sigue siendo < 3% de 205.8 GB.
```

### Checklist Día 1

- [x] `model/unet1d.py` — UNet1D + ResBlock1D implementados
- [x] `model/vpsde.py` — VPSDECoefficients implementado
- [x] `tests/test_forward_pass.py` — 18 tests escritos
- [x] Tests ejecutados en MI300X — 18/18 passed
- [x] VRAM pico confirmado: 0.432 GB — reportado a Agente 1

---

## DÍA 2 — PENDIENTE

**Tareas planificadas:**
- `training/loss.py` — score-matching loss con BF16 + autocast
- `training/train.py` — loop de entrenamiento AdamW lr=2e-4
- Checkpoint `checkpoints/day2_checkpoint.pt`
- Validar: loss desciende en primeras 100 iteraciones sin NaN/Inf

---

## DÍAS 3–7 — PLANIFICADOS (ver `Memory_Agent2.md` en ~/workspace/)

---

*Agente 2 — Ingeniero de Matemática y Modelado | AMD Hackathon*
