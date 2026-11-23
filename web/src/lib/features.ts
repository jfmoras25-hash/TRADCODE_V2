/**
 * Extracción de características. ESPEJO EXACTO de ml/lsc/features.py:
 * si cambia uno, hay que cambiar el otro y volver a correr la prueba de
 * paridad (npm run test:paridad).
 */

export const SEQ_LEN = 6
export const N_FEATURES = 147
export const FEATURE_VERSION = 1
const POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
const EPS = 1e-6
const CLIP = 10

export type Punto3 = [number, number, number]
export type Punto4 = [number, number, number, number]

/** Un cuadro crudo: manos en ranuras [Right, Left] y pose (33 puntos) o null. */
export interface CuadroCrudo {
  manos: [Punto3[] | null, Punto3[] | null]
  pose: Punto4[] | null
}

export interface ManoDetectada {
  score: number
  lado: string // "Right" | "Left" según MediaPipe
  puntos: Punto3[]
}

/** Misma regla que asignar_manos() en Python. */
export function asignarManos(manos: ManoDetectada[]): [Punto3[] | null, Punto3[] | null] {
  const ranuras: [Punto3[] | null, Punto3[] | null] = [null, null]
  for (const m of [...manos].sort((a, b) => b.score - a.score)) {
    let slot = m.lado === 'Right' ? 0 : 1
    if (ranuras[slot]) {
      slot = 1 - slot
      if (ranuras[slot]) continue
    }
    ranuras[slot] = m.puntos.slice(0, 21)
  }
  return ranuras
}

const norma = (x: number, y: number, z = 0) => Math.sqrt(x * x + y * y + z * z)
const recortar = (v: number) => (v > CLIP ? CLIP : v < -CLIP ? -CLIP : v)

export function caracteristicasCuadro(c: CuadroCrudo, aspect: number, out?: Float32Array, offset = 0): Float32Array {
  const f = out ?? new Float32Array(N_FEATURES)
  for (let i = 0; i < N_FEATURES; i++) f[offset + i] = 0

  for (let s = 0; s < 2; s++) {
    const mano = c.manos[s]
    if (!mano) continue
    const p = mano.map(([x, y, z]) => [x * aspect, y, z * aspect])
    const rel = p.map(([x, y, z]) => [x - p[0][0], y - p[0][1], z - p[0][2]])
    let esc = norma(rel[9][0], rel[9][1], rel[9][2])
    if (esc < EPS) esc = Math.max(...rel.map(([x, y, z]) => norma(x, y, z)))
    esc = Math.max(esc, EPS)
    for (let j = 0; j < 21; j++)
      for (let k = 0; k < 3; k++) f[offset + s * 63 + j * 3 + k] = rel[j][k] / esc
  }
  f[offset + 126] = c.manos[0] ? 1 : 0
  f[offset + 127] = c.manos[1] ? 1 : 0

  if (c.pose) {
    const P = c.pose.map(([x, y]) => [x * aspect, y])
    const cx = (P[11][0] + P[12][0]) / 2
    const cy = (P[11][1] + P[12][1]) / 2
    const sw = Math.max(norma(P[11][0] - P[12][0], P[11][1] - P[12][1]), EPS)
    POSE_IDX.forEach((idx, j) => {
      f[offset + 128 + 2 * j] = (P[idx][0] - cx) / sw
      f[offset + 129 + 2 * j] = (P[idx][1] - cy) / sw
    })
    for (let s = 0; s < 2; s++) {
      const mano = c.manos[s]
      if (!mano) continue
      f[offset + 142 + 2 * s] = (mano[0][0] * aspect - cx) / sw
      f[offset + 143 + 2 * s] = (mano[0][1] - cy) / sw
    }
    f[offset + 146] = 1
  }
  for (let i = 0; i < N_FEATURES; i++) f[offset + i] = recortar(f[offset + i])
  return f
}

export function caracteristicasSecuencia(cuadros: CuadroCrudo[], aspect: number): Float32Array {
  const out = new Float32Array(cuadros.length * N_FEATURES)
  cuadros.forEach((c, t) => caracteristicasCuadro(c, aspect, out, t * N_FEATURES))
  return out
}
