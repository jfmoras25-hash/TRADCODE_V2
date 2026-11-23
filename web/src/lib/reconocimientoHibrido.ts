import { esReposo, tipoSena } from './clasificador'
import type { CuadroCrudo } from './features'

export type ModoReconocimiento = 'general' | 'deletreo' | 'palabras'
export const LETRAS_RF: ReadonlySet<string> = new Set('ABCDEFIKLM')

export interface Candidato {
  etiqueta: string
  p: number
  motor?: 'GRU' | 'RF'
}

/** El ganador se elige por cobertura y umbral. Nunca se mezclan probabilidades de modelos. */
export function seleccionarCandidatos(
  gru: Candidato[],
  bosque: Candidato[],
  modo: ModoReconocimiento,
  umbral: number,
): { mejor?: Candidato; alternativas: Candidato[] } {
  if (modo === 'general') return { mejor: gru[0], alternativas: gru.slice(0, 3) }

  const permitido = (c: Candidato) =>
    !esReposo(c.etiqueta) && (tipoSena(c.etiqueta) === 'letra') === (modo === 'deletreo')
  const mejorGru = gru[0]
  const mejorBosque = bosque[0]
  if (modo === 'deletreo' && mejorBosque && LETRAS_RF.has(mejorBosque.etiqueta) && mejorBosque.p >= umbral
    && !(mejorGru && permitido(mejorGru) && !LETRAS_RF.has(mejorGru.etiqueta))) {
    return {
      mejor: mejorBosque,
      alternativas: bosque.filter((c) => LETRAS_RF.has(c.etiqueta)).slice(0, 3),
    }
  }

  const mejor = mejorGru && permitido(mejorGru) ? mejorGru : undefined
  return { mejor, alternativas: mejor ? gru.filter(permitido).slice(0, 3) : [] }
}

export async function predecirBosque(cuadro: CuadroCrudo | undefined, signal: AbortSignal): Promise<Candidato[]> {
  if (!cuadro?.manos.some(Boolean)) return []
  try {
    const respuesta = await fetch('/api/letras', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ manos: cuadro.manos }),
      signal: AbortSignal.any([signal, AbortSignal.timeout(600)]),
    })
    if (!respuesta.ok) return []
    const puntuaciones: { etiqueta: string; p: number }[] = await respuesta.json()
    return puntuaciones.map((c): Candidato => ({ ...c, motor: 'RF' })).sort((a, b) => b.p - a.p)
  } catch {
    // Si el bosque local no está disponible, la GRU todavía reconoce todas las letras.
    return []
  }
}
