import { useEffect, useRef, useState } from 'react'
import type { Clasificador } from '../lib/clasificador'
import { predecirBosque, seleccionarCandidatos, type Candidato, type ModoReconocimiento } from '../lib/reconocimientoHibrido'
import type { Ajustes } from './useAjustes'
import type { Vision } from './useVision'

export interface EstadoReconocedor {
  estado: 'inactivo' | 'sin-manos' | 'reconociendo' | 'error'
  top: Candidato[]
  probs: Float32Array | null
  progreso: number // 0..1, cuánto falta para confirmar la seña actual
}

const INICIAL: EstadoReconocedor = { estado: 'inactivo', top: [], probs: null, progreso: 0 }

/**
 * Clasifica la ventana de cuadros ~10 veces por segundo, suaviza las
 * probabilidades y "confirma" una seña cuando se mantiene por encima del
 * umbral durante `sostenerMs`. Para repetir la misma seña hay que bajar la
 * mano o cambiar de seña.
 */
export function useReconocedor(
  vision: Vision,
  clf: Clasificador | null,
  ajustes: Ajustes,
  activo: boolean,
  alConfirmar: (etiqueta: string) => void,
  modo: ModoReconocimiento = 'general',
  usarBosque = true,
): EstadoReconocedor {
  const [estado, setEstado] = useState<EstadoReconocedor>(INICIAL)
  const confirmarRef = useRef(alConfirmar)
  const ajustesRef = useRef(ajustes)
  useEffect(() => {
    confirmarRef.current = alConfirmar
    ajustesRef.current = ajustes
  })

  useEffect(() => {
    if (!activo || !clf || vision.estado !== 'listo') {
      setEstado(INICIAL)
      return
    }
    const etiquetas = clf.config.etiquetas
    const T = clf.config.seq_len
    const inicioModo = performance.now()
    const abortarBosque = new AbortController()
    setEstado(INICIAL)
    let ocupado = false
    let cancelado = false
    let ema: Float32Array | null = null
    let cand = { etiqueta: '', desde: 0 }
    let confirmada = false

    const id = window.setInterval(async () => {
      if (ocupado) return
      const a = ajustesRef.current
      const ahora = performance.now()
      if (modo !== 'general' && ahora - inicioModo < a.ventanaMs) return
      if (!vision.buffer.hayManos(ahora, 400)) {
        ema = null
        cand = { etiqueta: '', desde: ahora }
        confirmada = false
        setEstado({ estado: 'sin-manos', top: [], probs: null, progreso: 0 })
        return
      }
      const cuadros = vision.buffer.muestrear(T, a.ventanaMs, ahora)
      if (!cuadros) return
      ocupado = true
      try {
        const actual = modo === 'deletreo' && usarBosque ? vision.buffer.rango(ahora - 400, ahora).at(-1) : undefined
        const [p, bosque] = await Promise.all([
          clf.predecir(cuadros, vision.aspectRef.current),
          predecirBosque(actual, abortarBosque.signal),
        ])
        if (cancelado) return
        if (!ema) ema = Float32Array.from(p)
        else for (let i = 0; i < p.length; i++) ema[i] = 0.5 * ema[i] + 0.5 * p[i]
        const orden = Array.from(ema.keys()).sort((i, j) => ema![j] - ema![i])
        const gru = orden.map((i): Candidato => ({ etiqueta: etiquetas[i], p: ema![i], motor: 'GRU' }))
        const { mejor, alternativas } = seleccionarCandidatos(gru, bosque, modo, a.umbral)
        const etiqueta = mejor?.etiqueta ?? ''
        const pm = mejor?.p ?? 0
        if (etiqueta !== cand.etiqueta) {
          cand = { etiqueta, desde: ahora }
          confirmada = false
        }
        if (pm < a.umbral) cand.desde = ahora
        const progreso = confirmada ? 1 : Math.min(1, (ahora - cand.desde) / a.sostenerMs)
        if (mejor && !confirmada && pm >= a.umbral && ahora - cand.desde >= a.sostenerMs) {
          confirmada = true
          confirmarRef.current(etiqueta)
        }
        setEstado({
          estado: 'reconociendo',
          top: alternativas,
          probs: Float32Array.from(ema),
          progreso,
        })
      } catch {
        window.clearInterval(id)
        if (!cancelado) setEstado({ ...INICIAL, estado: 'error' })
      } finally {
        ocupado = false
      }
    }, 100)
    return () => {
      cancelado = true
      abortarBosque.abort()
      window.clearInterval(id)
    }
  }, [activo, clf, vision.estado, vision.buffer, vision.aspectRef, modo, usarBosque])

  return estado
}
