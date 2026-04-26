"""
UNet1D — Arquitectura de difusión condicional para reconstrucción de ECG.

Entrada : x (B, 12, 5000)  señal de 12 derivaciones + t (B,) timestep ∈ [0,1]
Salida  : (B, 1, 5000)      score ŝ_θ(x_t, t) del canal a reconstruir

Resoluciones del encoder con stride-2 downsampling:
    Nivel 0 : (B,  64, 5000)  ← skip0
    Nivel 1 : (B, 128, 2500)  ← skip1
    Nivel 2 : (B, 256, 1250)  ← skip2
    Nivel 3 : (B, 512,  625)  ← skip3  (usado sólo en bottleneck)
    Bottleneck: (B, 512, 625)

El condicionamiento cruzado de 11 derivaciones sanas (Día 2) se inyectará
en el bottleneck. Por ahora el modelo recibe los 12 canales directamente;
el canal dañado se enmascara antes de entrar (responsabilidad de inference.py).

Referencia de diseño: Ronneberger et al. 2015 (U-Net, MICCAI) adaptado a 1D.
Condicionamiento temporal: Ho et al. 2020 (DDPM) — FiLM sobre timestep embedding.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def get_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Embedding sinusoidal del timestep (estilo Transformer).

    t   : (B,) float en [0, 1]
    dim : dimensión de salida (debe ser par)
    →     (B, dim) float32

    Siempre en float32 — la precisión del embedding no debe degradarse con BF16.
    """
    assert dim % 2 == 0, f"dim debe ser par, recibido: {dim}"
    half = dim // 2
    # Frecuencias logarítmicamente espaciadas
    scale = math.log(10000.0) / (half - 1)
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32) * -scale
    )                                                   # (half,)
    args  = t.float()[:, None] * freqs[None, :]        # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


def _make_norm(channels: int, target_groups: int = 32) -> nn.GroupNorm:
    """
    GroupNorm robusto: reduce el número de grupos hasta que divida `channels`.
    Garantiza que channels=12 (primera capa encoder) funcione correctamente.
    """
    g = target_groups
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


# ─────────────────────────────────────────────────────────────────────────────
#  Bloque residual 1D
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    Bloque residual con convoluciones dilatadas y condicionamiento de tiempo.

    Estructura:
        Norm → SiLU → Conv1d(dil) → FiLM(t_emb) → Norm → SiLU → Conv1d(dil)
        + skip (proyección 1×1 si in_ch ≠ out_ch)

    Padding "same": pad = (kernel_size − 1) × dilation // 2
        Con kernel_size=7 y dilations {1,2,4,8,16}, (k-1)×dil siempre es par
        → la longitud temporal L se conserva exactamente.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        t_emb_dim:    int,
        kernel_size:  int = 7,
        dilation:     int = 1,
    ) -> None:
        super().__init__()
        # Same-padding: L_out = L_in para cualquier dilation
        pad = (kernel_size - 1) * dilation // 2

        self.norm1 = _make_norm(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=pad, dilation=dilation,
        )

        self.norm2 = _make_norm(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=pad, dilation=dilation,
        )

        # FiLM: proyecta t_emb → (scale, shift) para modular activaciones
        self.t_proj = nn.Linear(t_emb_dim, out_channels * 2)

        # Skip connection — proyección 1×1 si los canales difieren
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, in_channels,  L)
        t_emb : (B, t_emb_dim)
        →       (B, out_channels, L)
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # FiLM conditioning — modula DESPUÉS de la primera convolución
        t_out        = self.t_proj(t_emb)                    # (B, out_ch*2)
        scale, shift = t_out.chunk(2, dim=-1)                # cada (B, out_ch)
        h = h * (1.0 + scale[:, :, None]) + shift[:, :, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
#  U-Net 1D
# ─────────────────────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    """
    U-Net 1D con 4 niveles de resolución, bottleneck dilated y skip connections.

    Parámetros
    ----------
    in_channels   : 12  (derivaciones ECG; canal dañado enmascarado externamente)
    base_channels : 64  (se dobla en cada nivel del encoder)
    t_emb_dim     : 256 (dimensión del timestep embedding sinusoidal)

    VRAM estimado (batch=4, BF16, activaciones):
        Tensor más grande: (4, 512, 625) × 2 bytes ≈ 2.5 MB — trivial en MI300X.
        Pesos del modelo  ≈ 20 M parámetros × 4 bytes (f32 master) ≈ 80 MB.
    """

    def __init__(
        self,
        in_channels:   int = 12,
        base_channels: int = 64,
        t_emb_dim:     int = 256,
    ) -> None:
        super().__init__()
        C = base_channels        # 64
        self.t_emb_dim = t_emb_dim

        # ── Time embedding MLP ──────────────────────────────────────────
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        # ── Encoder ─────────────────────────────────────────────────────
        # enc0 : no downsampling — guarda skip0
        self.enc0  = ResBlock1D(in_channels, C,   t_emb_dim, dilation=1)
        # down1 : (B, C, 5000) → (B, C, 2500)
        self.down1 = nn.Conv1d(C,   C,   kernel_size=3, stride=2, padding=1)
        self.enc1  = ResBlock1D(C,   C*2, t_emb_dim, dilation=2)
        # down2 : (B, C*2, 2500) → (B, C*2, 1250)
        self.down2 = nn.Conv1d(C*2, C*2, kernel_size=3, stride=2, padding=1)
        self.enc2  = ResBlock1D(C*2, C*4, t_emb_dim, dilation=4)
        # down3 : (B, C*4, 1250) → (B, C*4, 625)
        self.down3 = nn.Conv1d(C*4, C*4, kernel_size=3, stride=2, padding=1)
        self.enc3  = ResBlock1D(C*4, C*8, t_emb_dim, dilation=8)

        # ── Bottleneck ───────────────────────────────────────────────────
        # dil=16 para campo receptivo máximo a resolución más baja
        self.bottleneck = ResBlock1D(C*8, C*8, t_emb_dim, dilation=16)

        # ── Decoder ──────────────────────────────────────────────────────
        # up3 : (B, C*8, 625) → Upsample → (B, C*8, 1250)
        #       cat skip2 (B, C*4, 1250) → dec3 in = C*8+C*4 = C*12
        self.up3  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(C*8, C*8, kernel_size=3, padding=1),
        )
        self.dec3 = ResBlock1D(C*8 + C*4, C*4, t_emb_dim, dilation=4)

        # up2 : (B, C*4, 1250) → (B, C*4, 2500)
        #       cat skip1 (B, C*2, 2500) → dec2 in = C*4+C*2 = C*6
        self.up2  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(C*4, C*4, kernel_size=3, padding=1),
        )
        self.dec2 = ResBlock1D(C*4 + C*2, C*2, t_emb_dim, dilation=2)

        # up1 : (B, C*2, 2500) → (B, C*2, 5000)
        #       cat skip0 (B, C, 5000) → dec1 in = C*2+C = C*3
        self.up1  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(C*2, C*2, kernel_size=3, padding=1),
        )
        self.dec1 = ResBlock1D(C*2 + C, C, t_emb_dim, dilation=1)

        # ── Proyección de salida ─────────────────────────────────────────
        self.out_norm = _make_norm(C)
        self.out_conv = nn.Conv1d(C, 1, kernel_size=1)      # (B, 1, 5000)

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 12, 5000)   señal ECG (canal dañado enmascarado por inference.py)
        t : (B,)             timestep en [0, 1]
        →   (B,  1, 5000)   score estimado del canal reconstruido
        """
        # ── Time embedding ──────────────────────────────────────────────
        # get_timestep_embedding corre siempre en f32;
        # self.t_mlp dentro del autocast context se casta a BF16 automáticamente.
        t_emb = get_timestep_embedding(t, self.t_emb_dim)   # (B, 256) f32
        t_emb = self.t_mlp(t_emb)                            # (B, 256)

        # ── Encoder ─────────────────────────────────────────────────────
        h0 = self.enc0(x, t_emb)                             # (B,  64, 5000)
        h1 = self.enc1(self.down1(h0), t_emb)                # (B, 128, 2500)
        h2 = self.enc2(self.down2(h1), t_emb)                # (B, 256, 1250)
        h3 = self.enc3(self.down3(h2), t_emb)                # (B, 512,  625)

        # ── Bottleneck ───────────────────────────────────────────────────
        h  = self.bottleneck(h3, t_emb)                      # (B, 512,  625)

        # ── Decoder ──────────────────────────────────────────────────────
        h  = self.up3(h)                                      # (B, 512, 1250)
        h  = self.dec3(torch.cat([h, h2], dim=1), t_emb)     # (B, 256, 1250)

        h  = self.up2(h)                                      # (B, 256, 2500)
        h  = self.dec2(torch.cat([h, h1], dim=1), t_emb)     # (B, 128, 2500)

        h  = self.up1(h)                                      # (B, 128, 5000)
        h  = self.dec1(torch.cat([h, h0], dim=1), t_emb)     # (B,  64, 5000)

        # ── Salida ───────────────────────────────────────────────────────
        return self.out_conv(F.silu(self.out_norm(h)))        # (B,   1, 5000)
