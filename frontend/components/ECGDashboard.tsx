/**
 * ECGDashboard.tsx
 *
 * Renderiza 12 canales de ECG en tiempo real usando WebGL (webgl-plot).
 * Recibe frames binarios float32 vía WebSocket desde el backend FastAPI.
 *
 * Protocolo acordado (Día 1):
 *   - ws://localhost:8000/stream
 *   - 1200 bytes / frame  →  Float32Array[300]
 *   - Layout: row-major  →  floats[ch * 25 + i] = canal ch, muestra i
 *   - 20 fps desde servidor; requestAnimationFrame para render (target 144 fps)
 *
 * IMPORTANTE: este componente usa WebGL → solo puede montarse en el cliente.
 * Importarlo con: dynamic(() => import('./ECGDashboard'), { ssr: false })
 */

import { useEffect, useRef } from 'react';
import { WebglPlot, WebglLine, ColorRGBA } from 'webgl-plot';

// ─── Constantes de protocolo (deben coincidir con backend/main.py) ────────────
const N_CHANNELS  = 12;
const CHUNK_SIZE  = 25;
const FRAME_BYTES = N_CHANNELS * CHUNK_SIZE * 4;  // 1200
const WS_URL      = 'ws://localhost:8000/stream';

// ─── Constantes de renderizado ────────────────────────────────────────────────
/** Muestras en el buffer circular: 5 segundos a 500 Hz */
const BUFFER_SAMPLES = 2500;

/**
 * VISUAL_GAIN — factor de amplificación puramente visual.
 *
 * Problema: los datos fisiológicos (onda R ≈ 1.0 mV) con scaleY base
 * de 1/N_CHANNELS producen ~36 px de excursión en 1080p → invisible.
 *
 * Solución: multiplicar scaleY por VISUAL_GAIN para amplificar la traza
 * SIN modificar los datos. El contrato con Agente 2 (amplitudes en mV)
 * permanece inalterado — esta ganancia es solo rendering.
 *
 * Con VISUAL_GAIN = 4.0:
 *   - Onda R (1.0 mV × 4/12)  ≈ 180 px — prominente, sobresale entre canales ✓
 *   - Onda T (0.35 mV × 4/12) ≈  63 px — contiene en la banda               ✓
 *   - Onda P (0.15 mV × 4/12) ≈  27 px — visible y clara                     ✓
 *
 * Ajustar este valor durante Demo-Mode según preferencia visual del jurado.
 */
const VISUAL_GAIN = 6.0;

/** Nombres clínicos de las 12 derivaciones estándar */
const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] as const;

/**
 * Paleta de colores por canal.
 * Definida como tuplas [r, g, b] en rango [0, 1] para usarlas tanto en
 * WebglPlot (ColorRGBA) como en CSS (via multiplicación × 255).
 */
const PALETTE: [number, number, number][] = [
  [0.00, 1.00, 0.53],  // I    — verde brillante
  [0.00, 0.90, 0.60],  // II
  [0.00, 0.80, 0.70],  // III
  [0.00, 0.70, 0.80],  // aVR
  [0.00, 0.60, 0.90],  // aVL
  [0.00, 0.50, 1.00],  // aVF  — azul cian
  [0.20, 0.95, 0.45],  // V1
  [0.20, 0.85, 0.55],  // V2
  [0.20, 0.75, 0.65],  // V3
  [0.20, 0.65, 0.75],  // V4
  [0.20, 0.55, 0.85],  // V5
  [0.20, 0.45, 0.95],  // V6
];

// ─── FPS counter ──────────────────────────────────────────────────────────────
/** Referencia mutable compartida para exponer el FPS al overlay sin re-renderizar React. */
type FpsRef = { current: number };

// ─── Componente ──────────────────────────────────────────────────────────────

export default function ECGDashboard() {
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const fpsDisplay = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // ── 1. Dimensionar canvas al tamaño real en píxeles (HiDPI) ──────────────
    function sizeCanvas() {
      const dpr = window.devicePixelRatio ?? 1;
      canvas!.width  = Math.floor(canvas!.offsetWidth  * dpr);
      canvas!.height = Math.floor(canvas!.offsetHeight * dpr);
    }
    sizeCanvas();

    // ── 2. Inicializar WebglPlot ──────────────────────────────────────────────
    const wglp = new WebglPlot(canvas);

    // ── 3. Crear una línea por canal ─────────────────────────────────────────
    const lines: WebglLine[] = PALETTE.map(([r, g, b], ch) => {
      const line = new WebglLine(new ColorRGBA(r, g, b, 1.0), BUFFER_SAMPLES);

      // Espaciado uniforme en X: de -1 a +1
      line.lineSpaceX(-1, 2 / BUFFER_SAMPLES);

      // Posición vertical: canal 0 arriba (+), canal 11 abajo (-)
      // offsetY del canal ch: parte el canvas en N bandas iguales
      line.offsetY = 1 - (2 * ch + 1) / N_CHANNELS;

      // scaleY: amplitud visual = ganancia × tamaño de banda por canal
      // VISUAL_GAIN amplifica sin tocar datos — ver comentario de constante arriba
      line.scaleY = (1.0 / N_CHANNELS) * VISUAL_GAIN;

      wglp.addLine(line);
      return line;
    });

    // ── 4. Conexión WebSocket ─────────────────────────────────────────────────
    const ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen  = () => console.log(`[WS] conectado → ${WS_URL}`);
    ws.onerror = (e) => console.error('[WS] error:', e);
    ws.onclose = () => console.log('[WS] desconectado');

   ws.onmessage = (event: MessageEvent<ArrayBuffer>) => {
      if (event.data.byteLength !== FRAME_BYTES) {
        console.warn(`[WS] frame inesperado: ${event.data.byteLength}B (esperado ${FRAME_BYTES}B)`);
        return;
      }

      const floats = new Float32Array(event.data);

      // --- DIAGNÓSTICO CAPA 3: INYECTADO AQUÍ ---
      // Imprimimos solo el 5% de los frames (aprox 1 vez por segundo) para no colapsar la consola
      if (Math.random() < 0.05) {
        console.log(`[WS-RECV] ArrayBuffer len: ${event.data.byteLength}, float32 count: ${floats.length}`);
        console.log(`[WS-RECV] primeros 2 floats: ${floats[0].toFixed(6)}, ${floats[1].toFixed(6)}`);
        console.log(`[WS-RECV] min/max en buffer: ${Math.min(...floats).toFixed(6)}, ${Math.max(...floats).toFixed(6)}`);
      }
      // ------------------------------------------

      // Cada canal ocupa CHUNK_SIZE = 25 floats consecutivos (row-major)
      for (let ch = 0; ch < N_CHANNELS; ch++) {
        const chunk = floats.subarray(ch * CHUNK_SIZE, (ch + 1) * CHUNK_SIZE);
        lines[ch].shiftAdd(chunk);
      }
    };

    // ── 5. Render loop con contador de FPS ───────────────────────────────────
    let animId: number;
    let lastTime = performance.now();
    let frameCount = 0;

    function renderLoop(now: number) {
      wglp.update();

      // Actualizar FPS cada 60 frames
      frameCount++;
      if (frameCount >= 60) {
        const fps = Math.round((frameCount * 1000) / (now - lastTime));
        if (fpsDisplay.current) fpsDisplay.current.textContent = `${fps} FPS`;
        lastTime = now;
        frameCount = 0;
      }

      animId = requestAnimationFrame(renderLoop);
    }
    animId = requestAnimationFrame(renderLoop);

    // ── 6. Redimensionamiento ─────────────────────────────────────────────────
    const ro = new ResizeObserver(() => sizeCanvas());
    ro.observe(canvas);

    // ── 7. Cleanup ────────────────────────────────────────────────────────────
    return () => {
      cancelAnimationFrame(animId);
      ws.close();
      ro.disconnect();
    };
  }, []);

  // ─── JSX ──────────────────────────────────────────────────────────────────
  return (
    <div style={styles.root}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.title}>ECG MONITOR — 12 LEAD</span>
        <span style={styles.subtitle}>AMD MI300X · ROCm 7.2.0 · 500 Hz · 12 derivaciones</span>
        <div style={styles.headerRight}>
          <span style={styles.stateChip}>● MONITORING</span>
          <span ref={fpsDisplay} style={styles.fps}>— FPS</span>
        </div>
      </div>

      {/* Área principal: canvas + etiquetas */}
      <div style={styles.main}>
        {/* Canvas WebGL: ocupa todo el área */}
        <canvas ref={canvasRef} style={styles.canvas} />

        {/* Etiquetas de derivación (overlay HTML sobre el canvas) */}
        <div style={styles.labels} aria-hidden>
          {LEAD_NAMES.map((name, ch) => {
            const [r, g, b] = PALETTE[ch];
            const cssColor = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},0.85)`;
            return (
              <div key={ch} style={styles.labelRow}>
                <span style={{ ...styles.label, color: cssColor }}>{name}</span>
              </div>
            );
          })}
        </div>

        {/* Línea central de referencia por canal (decorativa) */}
        <div style={styles.gridLines} aria-hidden>
          {LEAD_NAMES.map((_, ch) => (
            <div key={ch} style={styles.gridRow}>
              <div style={styles.gridLine} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Estilos en objeto (sin dependencia de CSS externo) ───────────────────────
const styles: Record<string, React.CSSProperties> = {
  root: {
    background: '#080c08',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    fontFamily: '"JetBrains Mono", "Fira Code", "Courier New", monospace',
    userSelect: 'none',
  },
  header: {
    padding: '6px 14px',
    borderBottom: '1px solid #1a2a1a',
    display: 'flex',
    alignItems: 'center',
    gap: 20,
    flexShrink: 0,
    background: '#080c08',
  },
  title: {
    color: '#00FF88',
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: 2,
  },
  subtitle: {
    color: '#3a5a3a',
    fontSize: 10,
  },
  headerRight: {
    marginLeft: 'auto',
    display: 'flex',
    alignItems: 'center',
    gap: 16,
  },
  stateChip: {
    color: '#00FF88',
    fontSize: 10,
    letterSpacing: 1,
  },
  fps: {
    color: '#3a5a3a',
    fontSize: 10,
    minWidth: 60,
    textAlign: 'right' as const,
  },
  main: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
  },
  canvas: {
    position: 'absolute',
    inset: 0,
    width: '100%',
    height: '100%',
    display: 'block',
  },
  labels: {
    position: 'absolute',
    top: 0,
    left: 0,
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    pointerEvents: 'none',
    zIndex: 10,
  },
  labelRow: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    paddingLeft: 8,
  },
  label: {
    fontSize: 9,
    fontWeight: 700,
    letterSpacing: 1,
    background: 'rgba(8,12,8,0.6)',
    padding: '1px 5px',
    borderRadius: 2,
  },
  gridLines: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    flexDirection: 'column',
    pointerEvents: 'none',
    zIndex: 1,
  },
  gridRow: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    borderBottom: '1px solid rgba(0,80,0,0.12)',
  },
  gridLine: {
    width: '100%',
    height: 1,
    background: 'rgba(0,120,0,0.08)',
  },
};
