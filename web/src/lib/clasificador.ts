import * as ort from 'onnxruntime-web/wasm'
import { caracteristicasSecuencia, FEATURE_VERSION, N_FEATURES, SEQ_LEN, type CuadroCrudo } from './features'

import { rutaPublica } from './rutas'

export interface FilaComparacion {
  modelo: string
  tipo: string
  acc_media: number
  f1_media: number
  f1_desv: number
  latencia_ms: number
}

export interface ConfigModelo {
  version_caracteristicas: number
  seq_len: number
  n_caracteristicas: number
  entrada: string
  salida: string
  etiquetas: string[]
  nombres?: Record<string, string>
  modelo?: string
  f1_validacion?: number
  comparacion?: FilaComparacion[]
  personas?: number
  secuencias?: number
  fecha?: string
}

export interface Clasificador {
  config: ConfigModelo
  predecir: (cuadros: CuadroCrudo[], aspect: number) => Promise<Float32Array>
}

export async function cargarClasificador(): Promise<Clasificador | null> {
  const r = await fetch(rutaPublica('model/config.json'), { cache: 'no-cache' })
  if (!r.ok || (r.headers.get('content-type') ?? '').includes('text/html')) return null
  const config = (await r.json()) as ConfigModelo
  if (config.modelo !== 'GRU') throw new Error('TRADCODE necesita el modelo GRU.')
  if (config.version_caracteristicas !== FEATURE_VERSION || config.n_caracteristicas !== N_FEATURES || config.seq_len !== SEQ_LEN)
    throw new Error('El modelo fue entrenado con otra versión de características. Vuelva a exportarlo.')
  ort.env.wasm.wasmPaths = rutaPublica('ort/')
  ort.env.wasm.numThreads = 1
  const sesion = await ort.InferenceSession.create(rutaPublica('model/modelo.onnx'), {
    executionProviders: ['wasm'],
  })
  const T = config.seq_len
  return {
    config,
    async predecir(cuadros, aspect) {
      const x = new ort.Tensor('float32', caracteristicasSecuencia(cuadros, aspect), [1, T, N_FEATURES])
      const out = await sesion.run({ [config.entrada]: x })
      return out[config.salida].data as Float32Array
    },
  }
}

export type TipoSena = 'letra' | 'numero' | 'palabra'

/** Clase de "no hay seña" (se recomienda grabarla con el modo Capturar). */
export const esReposo = (etiqueta: string) => /^_?(nada|ninguna|reposo|fondo)$/i.test(etiqueta)

export function tipoSena(etiqueta: string): TipoSena {
  if (/^\d+$/.test(etiqueta)) return 'numero'
  if (/^(?:[A-ZÑ]|NN)$/i.test(etiqueta)) return 'letra'
  return 'palabra'
}

const NOMBRES_TRADCODE: Record<string, string> = { NN: 'Ñ', ANNOS: 'AÑOS', DIAS: 'DÍAS', MILLON: 'MILLÓN' }
const ETIQUETAS_TRADCODE = new Map(Object.entries(NOMBRES_TRADCODE).map(([clave, nombre]) => [nombre, clave]))

export function nombreSena(config: ConfigModelo | undefined, etiqueta: string): string {
  return config?.nombres?.[etiqueta] ?? NOMBRES_TRADCODE[etiqueta] ?? etiqueta.replace(/_/g, ' ')
}

export function etiquetaSena(nombre: string): string {
  const etiqueta = nombre.trim().toUpperCase().replace(/\s+/g, '_')
  return ETIQUETAS_TRADCODE.get(etiqueta) ?? etiqueta
}
