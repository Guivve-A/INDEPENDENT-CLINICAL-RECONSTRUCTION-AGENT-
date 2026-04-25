/**
 * streamConfig.ts — Mirror TypeScript de config/stream_config.py
 *
 * Mantener sincronizado con la fuente Python.
 * TypeScript no puede importar Python, así que se duplica manualmente.
 * Si cambia un valor en stream_config.py → actualizar aquí también.
 */

export const N_CHANNELS    = 12;
export const CHUNK_SIZE    = 25;
export const SAMPLE_RATE   = 500;   // Hz
export const FPS           = 20;    // frames/s desde el servidor
export const FRAME_BYTES   = N_CHANNELS * CHUNK_SIZE * 4;  // 1200 bytes

export const LEADS_ORDER = [
  'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
  'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
] as const;

export type LeadName = typeof LEADS_ORDER[number];
