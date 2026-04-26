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

### Resultados de tests en hardware real — MI300X (2026-04-26)

```
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

> ⚠️ El output capturado es la segunda mitad del run (primeros tests UNet1D
> no están en el fragmento, pero el contador `18 passed, 0 failed` confirma
> que todos pasaron). En el próximo run capturar desde el inicio.

### Dato matemático confirmado en hardware

| Variable | Valor real | Significado |
|----------|-----------|-------------|
| `ᾱ(t=0)` | 1.0000 | t=0: señal intacta ✓ |
| `ᾱ(t=1)` | 0.0066 | t=1: casi ruido puro ✓ |
| `β(t=0)` | 0.10   | β_min verificado ✓ |
| `β(t=1)` | 20.00  | β_max verificado ✓ |
| `std > 0`| ✓      | Sin división por cero en loss ✓ |

> **Nota para Día 2:** `std(t=1) = √(1−0.0066) ≈ 0.9967` — la señal en t=1
> no es ruido puro exacto, sino ≈99.7% ruido. Esto es correcto y estable.

### Dato crítico para Agente 1 (coordinar VRAM antes de Día 4)

```
VRAM pico forward pass batch=4 BF16: ⚠️ no capturado en este run
VRAM disponible MI300X:               205.8 GB
```
> Ejecutar en próxima sesión:
> `python -c "import torch; from model.unet1d import UNet1D; m=UNet1D().cuda().eval(); x=torch.randn(4,12,5000,dtype=torch.bfloat16,device='cuda'); torch.autocast('cuda',dtype=torch.bfloat16).__enter__(); m(x,torch.rand(4).cuda()); print(torch.cuda.max_memory_allocated()/1e9,'GB')"`

### Checklist Día 1

- [x] `model/unet1d.py` — UNet1D + ResBlock1D implementados
- [x] `model/vpsde.py` — VPSDECoefficients implementado
- [x] `tests/test_forward_pass.py` — 18 tests escritos
- [x] Tests ejecutados en MI300X — 18/18 passed
- [ ] VRAM pico exacto medido y reportado a Agente 1 (pendiente próximo run)

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
