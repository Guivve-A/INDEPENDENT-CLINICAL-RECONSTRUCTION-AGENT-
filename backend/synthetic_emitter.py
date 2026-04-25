"""
synthetic_emitter.py — Generador de ECG sintético de 12 canales a 500 Hz.

Genera morfología QRS realista usando componentes gaussianas parametrizadas
(ondas P, Q, R, S, T).  El estado de fase es continuo entre frames — sin
glitches en transiciones de chunk.

Salida: arrays (n_channels, chunk_size) dtype float32 listos para .tobytes().
"""

import numpy as np


class SyntheticEmitter:
    """
    Genera frames ECG continuos a 500 Hz con morfología QRS paramétrica.

    Protocolo acordado con Agente 1 (Día 1):
        - shape por frame : (12, 25)  float32
        - bytes por frame : 1200      (row-major)
        - fps             : 20        (50 ms entre frames)
    """

    # Escala de amplitud aproximada para cada una de las 12 derivaciones estándar.
    # Refleja polaridad y magnitud relativa (referencia clínica, no clínicamente exacto).
    _CHANNEL_SCALES = np.array(
        [
            1.00,  # I
            0.80,  # II
            0.30,  # III
           -0.40,  # aVR  (QRS invertido es normal en aVR)
            0.50,  # aVL
            0.70,  # aVF
            0.20,  # V1   (r pequeña, S profunda)
            0.40,  # V2
            0.70,  # V3
            1.00,  # V4
            1.00,  # V5
            0.90,  # V6
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        n_channels: int = 12,
        sample_rate: int = 500,
        chunk_size: int = 25,
        heart_rate_bpm: float = 72.0,
        seed: int = 42,
    ) -> None:
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Muestras por ciclo cardíaco: 72 BPM → ~417 muestras a 500 Hz
        self.cycle_samples: int = round(sample_rate * 60.0 / heart_rate_bpm)

        # Contador de fase continuo (se resetea al llegar a cycle_samples)
        self._phase: int = 0

        # RNG estable — datos reproducibles para debugging
        self._rng = np.random.default_rng(seed=seed)

        # Precomputa la plantilla de un ciclo completo para máxima eficiencia
        self._cycle_template: np.ndarray = self._build_cycle_template()

    # ------------------------------------------------------------------
    # Privado
    # ------------------------------------------------------------------

    def _build_cycle_template(self) -> np.ndarray:
        """
        Construye una plantilla de un ciclo cardíaco como array float32 de
        longitud `cycle_samples`.  Usa componentes gaussianas para P, QRS, T.
        """
        t = np.arange(self.cycle_samples, dtype=np.float64)
        t_norm = t / self.cycle_samples  # dominio [0, 1)

        # Onda P  (deflexión positiva pre-QRS)
        p_wave = 0.15 * np.exp(-((t_norm - 0.15) ** 2) / (2 * 0.018 ** 2))

        # Onda Q  (pequeña deflexión negativa)
        q_wave = -0.08 * np.exp(-((t_norm - 0.285) ** 2) / (2 * 0.008 ** 2))

        # Onda R  (pico dominante del QRS)
        r_wave = 1.00 * np.exp(-((t_norm - 0.300) ** 2) / (2 * 0.010 ** 2))

        # Onda S  (deflexión negativa post-R)
        s_wave = -0.18 * np.exp(-((t_norm - 0.320) ** 2) / (2 * 0.009 ** 2))

        # Onda T  (repolarización ventricular)
        t_wave = 0.35 * np.exp(-((t_norm - 0.600) ** 2) / (2 * 0.040 ** 2))

        # Deriva de línea base (sinusoide lenta, <1% del ciclo)
        baseline = 0.02 * np.sin(2 * np.pi * t_norm)

        template = p_wave + q_wave + r_wave + s_wave + t_wave + baseline
        return template.astype(np.float32)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def next_frame(self) -> np.ndarray:
        """
        Devuelve el próximo chunk de señal ECG.

        Returns
        -------
        np.ndarray
            Shape (n_channels, chunk_size), dtype float32.
            Serializable directamente como `frame.tobytes()` → 1200 bytes.
        """
        # Índices de fase para este chunk (manejo correcto del wrap-around)
        phases = (self._phase + np.arange(self.chunk_size)) % self.cycle_samples

        # Lookup en plantilla precomputada — (chunk_size,)
        template = self._cycle_template[phases]

        # Broadcast: (n_channels, 1) × (1, chunk_size) → (n_channels, chunk_size)
        frame = self._CHANNEL_SCALES[:, np.newaxis] * template[np.newaxis, :]

        # Ruido gaussiano aditivo (std ≈ 0.01 mV equivalente)
        noise = self._rng.standard_normal(frame.shape).astype(np.float32) * 0.01
        frame = frame + noise

        # Avanzar fase
        self._phase = (self._phase + self.chunk_size) % self.cycle_samples

        return frame  # (12, 25), float32
