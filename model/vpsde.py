"""
VPSDECoefficients — Variance Preserving SDE noise schedule.

Referencia verificada:
    Song et al. 2021 "Score-Based Generative Modeling through SDEs"
    arXiv:2011.13456 — §2 (VP-SDE) y Appendix B (linear schedule)

Proceso de perturbación directa (forward SDE):
    dx = -½ β(t) x dt + √β(t) dW,   t ∈ [0, 1]

Distribución marginal exacta:
    p(x_t | x_0) = N( √ᾱ(t) · x_0,  (1 − ᾱ(t)) · I )

    donde  ᾱ(t) = exp( −½ ∫₀ᵗ β(s) ds )
                = exp( −½ [ β_min·t + (β_max − β_min)·t²/2 ] )   ← schedule lineal
"""
import torch


class VPSDECoefficients:
    """
    Schedule de ruido lineal para VP-SDE.

        β(t) = β_min + (β_max − β_min) · t,   t ∈ [0, 1]

    Valores por defecto tomados de Song et al. 2021 (Tabla 1):
        beta_min = 0.1,  beta_max = 20.0
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        assert 0 < beta_min < beta_max, "Se requiere 0 < beta_min < beta_max"
        self.beta_min = beta_min
        self.beta_max = beta_max

    # ------------------------------------------------------------------ #
    #  Coeficientes del schedule                                           #
    # ------------------------------------------------------------------ #

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        β(t): intensidad instantánea del ruido en el tiempo t.
        t: cualquier shape float → mismo shape, valores en [β_min, β_max].
        """
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        ᾱ(t) = exp(−½ ∫₀ᵗ β(s) ds)
              = exp(−½ [ β_min·t + (β_max−β_min)·t²/2 ])

        t: cualquier shape float → mismo shape, valores en (0, 1].
        Monótonamente decreciente: ᾱ(0) ≈ 1  (sin ruido),
                                   ᾱ(1) ≈ 0  (ruido puro).
        """
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    # ------------------------------------------------------------------ #
    #  Distribución marginal                                               #
    # ------------------------------------------------------------------ #

    def marginal_prob(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parámetros de p(x_t | x_0) = N(mean, std² · I).

        x0 : (B, C, L)
        t  : (B,)         — en [0, 1]
        →    mean : (B, C, L)
             std  : (B, 1, 1)   broadcastable con (B, C, L)
        """
        ab = self.alpha_bar(t.float())[:, None, None]          # (B, 1, 1)
        mean = torch.sqrt(ab) * x0
        std  = torch.sqrt(torch.clamp(1.0 - ab, min=1e-5))
        return mean, std

    # ------------------------------------------------------------------ #
    #  Perturbación (usada en la loss de score-matching)                   #
    # ------------------------------------------------------------------ #

    def perturb(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samplea x_t ~ p(x_t | x_0) = √ᾱ(t)·x_0 + √(1−ᾱ(t))·ε,   ε ~ N(0,I).

        x0 : (B, C, L)
        t  : (B,)
        →    x_t   : (B, C, L)  — señal perturbada
             noise  : (B, C, L)  — ruido ε añadido (target del score-matching)
        """
        mean, std = self.marginal_prob(x0, t)
        noise = torch.randn_like(x0)
        x_t   = mean + std * noise
        return x_t, noise

    # ------------------------------------------------------------------ #
    #  Utilidades para el solucionador inverso (Día 3)                     #
    # ------------------------------------------------------------------ #

    def drift_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Coeficiente de deriva f(x,t) = −½β(t).  Escalar por elemento."""
        return -0.5 * self.beta(t)

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Coeficiente de difusión g(t) = √β(t)."""
        return torch.sqrt(self.beta(t))
