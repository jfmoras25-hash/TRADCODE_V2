import { useEffect, useMemo, useRef, useState } from 'react'
import type { Vision } from '../hooks/useVision'
import { etiquetaSena, nombreSena } from '../lib/clasificador'
import { FEATURE_VERSION, SEQ_LEN, type CuadroCrudo } from '../lib/features'
import { indicesRemuestreo } from '../lib/secuencia'

interface MuestraCapturada {
  etiqueta: string
  persona: string
  aspect: number
  fecha: string
  cuadros: CuadroCrudo[]
}

export type FaseCaptura =
  | { tipo: 'quieto' }
  | { tipo: 'cuenta'; n: number }
  | { tipo: 'grabando'; k: number; total: number }
  | { tipo: 'pausa'; k: number; total: number }

const espera = (ms: number) => new Promise((r) => setTimeout(r, ms))
const r5 = (v: number) => Math.round(v * 1e5) / 1e5
const redondear = (c: CuadroCrudo): CuadroCrudo => ({
  manos: c.manos.map((m) => (m ? m.map((p) => p.map(r5)) : null)) as CuadroCrudo['manos'],
  pose: c.pose ? (c.pose.map((p) => p.map(r5)) as CuadroCrudo['pose']) : null,
})

function leerPersona() {
  try {
    return localStorage.getItem('lsc-persona') ?? ''
  } catch {
    return ''
  }
}

export function useCaptura(vision: Vision, ventanaMs: number) {
  const [persona, setPersona] = useState(leerPersona)
  const [sena, setSena] = useState('')
  const [cantidad, setCantidad] = useState(10)
  const [muestras, setMuestras] = useState<MuestraCapturada[]>([])
  const [fase, setFase] = useState<FaseCaptura>({ tipo: 'quieto' })
  const [descartadas, setDescartadas] = useState(0)
  const [guardando, setGuardando] = useState(false)
  const [mensaje, setMensaje] = useState('')
  const cancelar = useRef(false)
  const persistidas = useRef(new WeakSet<MuestraCapturada>())

  useEffect(() => {
    try {
      localStorage.setItem('lsc-persona', persona)
    } catch {
      /* almacenamiento no disponible */
    }
  }, [persona])

  useEffect(() => {
    const aviso = (e: BeforeUnloadEvent) => {
      if (muestras.some((muestra) => !persistidas.current.has(muestra))) e.preventDefault()
    }
    window.addEventListener('beforeunload', aviso)
    return () => window.removeEventListener('beforeunload', aviso)
  }, [muestras])

  async function grabar() {
    const etiqueta = etiquetaSena(sena)
    if (!etiqueta || vision.estado !== 'listo') return
    cancelar.current = false
    for (let n = 3; n > 0; n--) {
      setFase({ tipo: 'cuenta', n })
      await espera(1000)
      if (cancelar.current) return setFase({ tipo: 'quieto' })
    }
    for (let k = 1; k <= cantidad; k++) {
      setFase({ tipo: 'grabando', k, total: cantidad })
      const ini = performance.now()
      await espera(ventanaMs)
      if (cancelar.current) break
      const cuadros = vision.buffer.rango(ini, performance.now())
      const conMano = cuadros.filter((c) => c.manos[0] || c.manos[1]).length
      if (cuadros.length < 3 || conMano < cuadros.length / 2) {
        setDescartadas((d) => d + 1)
      } else {
        const sel = indicesRemuestreo(cuadros.length, SEQ_LEN).map((i) => redondear(cuadros[i]))
        setMuestras((m) => [
          ...m,
          { etiqueta, persona: persona.trim() || 'anonimo', aspect: vision.aspectRef.current, fecha: new Date().toISOString(), cuadros: sel },
        ])
      }
      if (cancelar.current) break
      setFase({ tipo: 'pausa', k, total: cantidad })
      await espera(700)
      if (cancelar.current) break
    }
    setFase({ tipo: 'quieto' })
  }

  function documento() {
    return {
      formato: 'lsc-capturas',
      version: 1,
      seq_len: SEQ_LEN,
      version_caracteristicas: FEATURE_VERSION,
      muestras,
    }
  }

  function descargar() {
    const blob = new Blob([JSON.stringify(documento())], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    const quien = (persona.trim() || 'anonimo').toLowerCase().replace(/\s+/g, '_')
    a.download = `capturas_${quien}_${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  async function guardar() {
    setGuardando(true)
    setMensaje('')
    try {
      const doc = documento()
      let guardadas = 0
      let duplicadas = 0
      for (let i = 0; i < doc.muestras.length; i += 50) {
        const lote = doc.muestras.slice(i, i + 50)
        const response = await fetch('/api/capturas', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...doc, muestras: lote }),
        })
        if (!response.ok) throw new Error('La captura no se pudo guardar en el servidor.')
        const resultado: { guardadas: number; duplicadas: number } = await response.json()
        lote.forEach((muestra) => persistidas.current.add(muestra))
        guardadas += resultado.guardadas
        duplicadas += resultado.duplicadas
      }
      setMensaje(`${guardadas} guardadas · ${duplicadas} repetidas`)
    } catch {
      setMensaje('No se pudo guardar. Puedes descargar el JSON y conservar las muestras.')
    } finally {
      setGuardando(false)
    }
  }

  return {
    persona, setPersona, sena, setSena, cantidad, setCantidad, muestras, setMuestras,
    fase, descartadas, grabar, descargar, guardar, guardando, mensaje, camaraLista: vision.estado === 'listo',
    detener: () => (cancelar.current = true),
  }
}

type Captura = ReturnType<typeof useCaptura>

export function SuperposicionCaptura({ fase }: { fase: FaseCaptura }) {
  if (fase.tipo === 'quieto') return null
  if (fase.tipo === 'cuenta') return <div className="captura-cuenta">{fase.n}</div>
  return (
    <div className={`captura-estado ${fase.tipo === 'grabando' ? 'rec' : ''}`}>
      {fase.tipo === 'grabando' ? '● Grabando' : 'Prepárese'} {fase.k}/{fase.total}
    </div>
  )
}

export function PanelCapturar({ cap, etiquetasModelo }: { cap: Captura; etiquetasModelo: string[] }) {
  const ocupado = cap.fase.tipo !== 'quieto'
  const conteo = useMemo(() => {
    const c = new Map<string, number>()
    for (const m of cap.muestras) c.set(m.etiqueta, (c.get(m.etiqueta) ?? 0) + 1)
    return [...c.entries()].sort()
  }, [cap.muestras])

  return (
    <div className="panel-cuerpo">
      <section>
        <h2>Grabar ejemplos propios</h2>
        <p className="ayuda">
          Sirve para agregar señas que no están en LSC70 o para sumar ejemplos con su propia cámara. Cada ejemplo dura lo
          mismo que la ventana de reconocimiento y se guarda como 6 cuadros, igual que el dataset.
        </p>
        <div className="formulario">
          <label>
            Su nombre
            <input maxLength={80} value={cap.persona} onChange={(e) => cap.setPersona(e.target.value)} placeholder="p. ej. jorge" />
          </label>
          <label>
            Seña
            <input
              value={cap.sena}
              onChange={(e) => cap.setSena(e.target.value)}
              placeholder="p. ej. A, HOLA, NADA"
              list="etiquetas-modelo"
              maxLength={30}
            />
            <datalist id="etiquetas-modelo">
              {etiquetasModelo.map((e) => (
                <option key={e} value={nombreSena(undefined, e)} />
              ))}
            </datalist>
          </label>
          <label>
            Ejemplos seguidos
            <input
              type="number"
              min={1}
              max={50}
              value={cap.cantidad}
              onChange={(e) => cap.setCantidad(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
            />
          </label>
        </div>
        <div className="botonera">
          {ocupado ? (
            <button onClick={cap.detener}>Detener</button>
          ) : (
            <button className="primario" onClick={cap.grabar} disabled={!cap.sena.trim() || !cap.camaraLista || cap.guardando}>
              Grabar
            </button>
          )}
        </div>
        {!cap.camaraLista && <p className="ayuda">Activa la cámara para grabar ejemplos.</p>}
      </section>

      <section>
        <h2>
          Capturados: {cap.muestras.length}
          {cap.descartadas > 0 && <span className="ayuda"> · {cap.descartadas} descartados (sin mano)</span>}
        </h2>
        {conteo.length ? (
          <ul className="conteo">
            {conteo.map(([e, n]) => (
              <li key={e}>
                <span>{nombreSena(undefined, e)}</span>
                <strong>{n}</strong>
              </li>
            ))}
          </ul>
        ) : (
          <p className="ayuda">Aún no hay ejemplos.</p>
        )}
        <div className="botonera">
          <button onClick={cap.guardar} disabled={!cap.muestras.length || ocupado || cap.guardando}>
            {cap.guardando ? 'Guardando…' : 'Guardar en TRADCODE'}
          </button>
          <button className="primario" onClick={cap.descargar} disabled={!cap.muestras.length || ocupado}>
            Descargar JSON
          </button>
          <button onClick={() => cap.setMuestras((m) => m.slice(0, -1))} disabled={!cap.muestras.length || ocupado}>
            Quitar último
          </button>
          <button
            onClick={() => window.confirm('¿Borrar todos los ejemplos capturados?') && cap.setMuestras(() => [])}
            disabled={!cap.muestras.length || ocupado}
          >
            Borrar todo
          </button>
        </div>
        {cap.mensaje && <p role="status">{cap.mensaje}</p>}
        <p className="ayuda">
          Luego: <code>python scripts/importar_capturas.py capturas_*.json</code> y vuelva a entrenar. Grabe también una
          clase <code>NADA</code> (manos quietas o en reposo) para que el modelo aprenda cuándo no hay seña.
        </p>
      </section>
    </div>
  )
}
