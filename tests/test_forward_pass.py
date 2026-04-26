"""
test_forward_pass.py — Validación de Día 1, Agente 2.

Verifica:
  1. UNet1D forward pass → shape (4, 1, 5000), sin NaN, en BF16/cuda
  2. ResBlock1D preserva longitud temporal para todas las dilations
  3. VPSDECoefficients: ᾱ(t) monotone, rangos, shapes de marginal_prob/perturb
  4. get_timestep_embedding: shape y determinismo

Uso:
    python tests/test_forward_pass.py              # smoke-test directo
    python -m pytest tests/test_forward_pass.py    # con pytest (opcional)

Sin MIMIC-III ni DataLoader — trabaja con tensores sintéticos.
"""
import sys
import math
from pathlib import Path

import torch

# ── Imports del paquete (raíz del proyecto en path) ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.unet1d import UNet1D, ResBlock1D, get_timestep_embedding
from model.vpsde import VPSDECoefficients

# ── Constantes del contrato de datos (mirror de data_loader.py) ─────────────
N_CHANNELS    = 12
TENSOR_LENGTH = 5000
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_TRAIN   = torch.bfloat16


# ═══════════════════════════════════════════════════════════════════════════
#  1. UNet1D
# ═══════════════════════════════════════════════════════════════════════════

def test_unet_output_shape() -> None:
    """Contrato principal de Día 1: out.shape == (4, 1, 5000)."""
    model = UNet1D().to(DEVICE).eval()
    x = torch.randn(4, N_CHANNELS, TENSOR_LENGTH, dtype=DTYPE_TRAIN, device=DEVICE)
    t = torch.rand(4, device=DEVICE)
    with torch.autocast(DEVICE, dtype=DTYPE_TRAIN):
        out = model(x, t)
    assert out.shape == (4, 1, TENSOR_LENGTH), \
        f"FAIL shape: esperado (4,1,5000), obtenido {tuple(out.shape)}"
    print(f"[OK] UNet1D forward pass shape: {tuple(out.shape)}")


def test_unet_batch_1() -> None:
    """batch=1 — caso de inferencia en tiempo real desde Agente 3."""
    model = UNet1D().to(DEVICE).eval()
    x = torch.randn(1, N_CHANNELS, TENSOR_LENGTH, dtype=DTYPE_TRAIN, device=DEVICE)
    t = torch.rand(1, device=DEVICE)
    with torch.autocast(DEVICE, dtype=DTYPE_TRAIN):
        out = model(x, t)
    assert out.shape == (1, 1, TENSOR_LENGTH), \
        f"FAIL batch=1: {tuple(out.shape)}"
    print(f"[OK] UNet1D batch=1 shape: {tuple(out.shape)}")


def test_unet_no_nan() -> None:
    """Ningún NaN en la salida para entradas estándar."""
    model = UNet1D().to(DEVICE).eval()
    x = torch.randn(4, N_CHANNELS, TENSOR_LENGTH, dtype=DTYPE_TRAIN, device=DEVICE)
    t = torch.rand(4, device=DEVICE)
    with torch.autocast(DEVICE, dtype=DTYPE_TRAIN):
        out = model(x, t)
    assert not torch.isnan(out.float()).any(), "FAIL: NaN detectado en la salida"
    print("[OK] UNet1D sin NaN en output")


def test_unet_gradients_flow() -> None:
    """Backprop funciona — necesario para el entrenamiento del Día 2."""
    model = UNet1D().to(DEVICE).train()
    x   = torch.randn(2, N_CHANNELS, TENSOR_LENGTH, device=DEVICE)
    t   = torch.rand(2, device=DEVICE)
    out = model(x, t)
    out.mean().backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "FAIL: ningún parámetro recibió gradiente"
    print("[OK] UNet1D gradientes fluyen correctamente")


def test_unet_param_count() -> None:
    """Reporta el número de parámetros — referencia para Agente 1 (VRAM)."""
    model = UNet1D()
    n = sum(p.numel() for p in model.parameters())
    print(f"[INFO] UNet1D parámetros: {n / 1e6:.2f}M  "
          f"({n * 4 / 1e6:.1f} MB en f32 master weights)")
    # No hay assertion de valor exacto — puede variar con base_channels
    assert n > 0


def test_unet_vram_usage() -> None:
    """Mide VRAM real del forward pass en GPU (batch=4, BF16)."""
    if DEVICE != "cuda":
        print("[SKIP] test_unet_vram_usage — no hay GPU disponible")
        return
    torch.cuda.reset_peak_memory_stats()
    model = UNet1D().to(DEVICE).eval()
    x = torch.randn(4, N_CHANNELS, TENSOR_LENGTH, dtype=DTYPE_TRAIN, device=DEVICE)
    t = torch.rand(4, device=DEVICE)
    with torch.autocast(DEVICE, dtype=DTYPE_TRAIN):
        _ = model(x, t)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[INFO] VRAM pico forward pass (batch=4, BF16): {peak_gb:.3f} GB  "
          f"(de {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB disponibles)")
    # Reportar a Agente 1 — sin assertion de valor exacto
    assert peak_gb > 0


# ═══════════════════════════════════════════════════════════════════════════
#  2. ResBlock1D
# ═══════════════════════════════════════════════════════════════════════════

def test_resblock_same_channels() -> None:
    block = ResBlock1D(64, 64, t_emb_dim=256, dilation=1).to(DEVICE)
    x     = torch.randn(2, 64, TENSOR_LENGTH, device=DEVICE)
    t_emb = torch.randn(2, 256, device=DEVICE)
    out   = block(x, t_emb)
    assert out.shape == (2, 64, TENSOR_LENGTH), f"FAIL: {out.shape}"
    print("[OK] ResBlock1D same channels (64→64)")


def test_resblock_channel_expansion() -> None:
    """Primer bloque encoder: 12 canales de entrada."""
    block = ResBlock1D(12, 64, t_emb_dim=256, dilation=1).to(DEVICE)
    x     = torch.randn(2, 12, TENSOR_LENGTH, device=DEVICE)
    t_emb = torch.randn(2, 256, device=DEVICE)
    out   = block(x, t_emb)
    assert out.shape == (2, 64, TENSOR_LENGTH), f"FAIL: {out.shape}"
    print("[OK] ResBlock1D expansión de canales (12→64)")


def test_resblock_dilations_preserve_length() -> None:
    """Todas las dilations usadas en UNet1D deben conservar L=5000."""
    for dil in [1, 2, 4, 8, 16]:
        block = ResBlock1D(64, 64, t_emb_dim=256, dilation=dil).to(DEVICE)
        x     = torch.randn(1, 64, TENSOR_LENGTH, device=DEVICE)
        t_emb = torch.randn(1, 256, device=DEVICE)
        out   = block(x, t_emb)
        assert out.shape[-1] == TENSOR_LENGTH, \
            f"FAIL dil={dil}: longitud {out.shape[-1]} ≠ {TENSOR_LENGTH}"
    print("[OK] ResBlock1D todas las dilations preservan L=5000")


# ═══════════════════════════════════════════════════════════════════════════
#  3. VPSDECoefficients
# ═══════════════════════════════════════════════════════════════════════════

def test_vpsde_alpha_bar_range() -> None:
    """ᾱ(t) ∈ (0, 1] para todo t ∈ [0, 1]."""
    sde = VPSDECoefficients()
    t   = torch.linspace(0.0, 1.0, 200)
    ab  = sde.alpha_bar(t)
    assert (ab > 0.0).all(),          "FAIL: ᾱ(t) ≤ 0 en algún punto"
    assert (ab <= 1.0 + 1e-6).all(),  "FAIL: ᾱ(t) > 1 en algún punto"
    print(f"[OK] VPSDECoefficients ᾱ(t) rango: [{ab.min():.4f}, {ab.max():.4f}]")


def test_vpsde_alpha_bar_monotone() -> None:
    """ᾱ(t) debe ser monótonamente decreciente (más ruido a mayor t)."""
    sde  = VPSDECoefficients()
    t    = torch.linspace(0.0, 1.0, 500)
    ab   = sde.alpha_bar(t)
    diff = ab[1:] - ab[:-1]
    assert (diff <= 1e-6).all(), "FAIL: ᾱ(t) no es monótonamente decreciente"
    print("[OK] VPSDECoefficients ᾱ(t) monótonamente decreciente")


def test_vpsde_marginal_prob_shapes() -> None:
    sde          = VPSDECoefficients()
    x0           = torch.randn(4, N_CHANNELS, TENSOR_LENGTH)
    t            = torch.rand(4)
    mean, std    = sde.marginal_prob(x0, t)
    assert mean.shape == (4, N_CHANNELS, TENSOR_LENGTH), f"FAIL mean.shape: {mean.shape}"
    assert std.shape  == (4, 1, 1),                       f"FAIL std.shape:  {std.shape}"
    print(f"[OK] VPSDECoefficients marginal_prob shapes: mean{tuple(mean.shape)} std{tuple(std.shape)}")


def test_vpsde_std_positive() -> None:
    """std > 0 para t ∈ (0, 1] — necesario para no dividir por cero en la loss."""
    sde      = VPSDECoefficients()
    x0       = torch.randn(4, N_CHANNELS, TENSOR_LENGTH)
    t        = torch.rand(4).clamp(1e-4, 1.0)
    _, std   = sde.marginal_prob(x0, t)
    assert (std > 0).all(), "FAIL: std ≤ 0"
    print("[OK] VPSDECoefficients std > 0 para todo t > 0")


def test_vpsde_perturb_shapes() -> None:
    sde        = VPSDECoefficients()
    x0         = torch.randn(4, N_CHANNELS, TENSOR_LENGTH)
    t          = torch.rand(4)
    x_t, noise = sde.perturb(x0, t)
    assert x_t.shape   == (4, N_CHANNELS, TENSOR_LENGTH)
    assert noise.shape == (4, N_CHANNELS, TENSOR_LENGTH)
    print("[OK] VPSDECoefficients perturb shapes correctos")


def test_vpsde_beta_range() -> None:
    sde = VPSDECoefficients(beta_min=0.1, beta_max=20.0)
    t   = torch.linspace(0.0, 1.0, 100)
    b   = sde.beta(t)
    assert b.min() >= sde.beta_min - 1e-6, f"FAIL: β_min={b.min()}"
    assert b.max() <= sde.beta_max + 1e-6, f"FAIL: β_max={b.max()}"
    print(f"[OK] VPSDECoefficients β(t) rango: [{b.min():.2f}, {b.max():.2f}]")


# ═══════════════════════════════════════════════════════════════════════════
#  4. Timestep embedding
# ═══════════════════════════════════════════════════════════════════════════

def test_timestep_embedding_shape() -> None:
    t   = torch.rand(4)
    emb = get_timestep_embedding(t, dim=256)
    assert emb.shape == (4, 256), f"FAIL: {emb.shape}"
    print("[OK] get_timestep_embedding shape (4, 256)")


def test_timestep_embedding_deterministic() -> None:
    t  = torch.tensor([0.0, 0.25, 0.5, 1.0])
    e1 = get_timestep_embedding(t, 256)
    e2 = get_timestep_embedding(t, 256)
    assert torch.allclose(e1, e2), "FAIL: embedding no es determinista"
    print("[OK] get_timestep_embedding es determinista")


def test_timestep_embedding_different_times() -> None:
    """Embeddings de distintos timesteps deben ser distintos."""
    t   = torch.tensor([0.1, 0.9])
    emb = get_timestep_embedding(t, 256)
    assert not torch.allclose(emb[0], emb[1]), \
        "FAIL: embeddings de t=0.1 y t=0.9 son idénticos"
    print("[OK] get_timestep_embedding distingue timesteps distintos")


# ═══════════════════════════════════════════════════════════════════════════
#  Smoke-test principal (entrypoint de Día 1)
# ═══════════════════════════════════════════════════════════════════════════

def run_all() -> None:
    """
    Ejecuta todos los tests en secuencia.
    Imprime PASS / FAIL con detalles de GPU si está disponible.
    """
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"\n{'─'*60}")
        print(f"  GPU  : {props.name}")
        print(f"  VRAM : {props.total_memory / 1e9:.1f} GB")
        print(f"  BF16 : {'✓' if torch.cuda.is_bf16_supported() else '✗'}")
        print(f"{'─'*60}\n")
    else:
        print("[WARN] GPU no disponible — ejecutando en CPU (BF16 limitado)\n")

    tests = [
        # UNet1D
        test_unet_output_shape,
        test_unet_batch_1,
        test_unet_no_nan,
        test_unet_gradients_flow,
        test_unet_param_count,
        test_unet_vram_usage,
        # ResBlock1D
        test_resblock_same_channels,
        test_resblock_channel_expansion,
        test_resblock_dilations_preserve_length,
        # VPSDECoefficients
        test_vpsde_alpha_bar_range,
        test_vpsde_alpha_bar_monotone,
        test_vpsde_marginal_prob_shapes,
        test_vpsde_std_positive,
        test_vpsde_perturb_shapes,
        test_vpsde_beta_range,
        # Timestep embedding
        test_timestep_embedding_shape,
        test_timestep_embedding_deterministic,
        test_timestep_embedding_different_times,
    ]

    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
            failed += 1

    print(f"\n{'═'*60}")
    print(f"  Día 1 — Agente 2   {passed} passed, {failed} failed")
    if failed == 0:
        print("  ✓ Forward pass validado. Listo para Día 2 (score-matching).")
    print(f"{'═'*60}\n")

    if failed > 0:
        raise SystemExit(f"{failed} test(s) fallaron.")


if __name__ == "__main__":
    run_all()
vram_pico_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\n[INFO] VRAM pico del forward pass: {vram_pico_gb:.3f} GB")