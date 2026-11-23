import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { esReposo, nombreSena, tipoSena, type ConfigModelo, type TipoSena } from '../lib/clasificador'
import { rutaPublica } from '../lib/rutas'
import type { EstadoReconocedor } from '../hooks/useReconocedor'

type Filtro = 'todas' | TipoSena

export function usePractica(config?: ConfigModelo) {
  const [filtro, setFiltro] = useState<Filtro>('letra')
  const [objetivo, setObjetivo] = useState<string | null>(null)
  const [aciertos, setAciertos] = useState(0)
  const [intentos, setIntentos] = useState(0)
  const [racha, setRacha] = useState(0)
  const [acierto, setAcierto] = useState(false)
  const bloqueado = useRef(false)

  const opciones = useMemo(
    () => (config?.etiquetas ?? []).filter((e) => !esReposo(e) && (filtro === 'todas' || tipoSena(e) === filtro)),
    [config, filtro],
  )

  const siguiente = useCallback(() => {
    setAcierto(false)
    bloqueado.current = false
    setObjetivo((prev) => {
      if (!opciones.length) return null
      const resto = opciones.length > 1 ? opciones.filter((o) => o !== prev) : opciones
      return resto[Math.floor(Math.random() * resto.length)]
    })
  }, [opciones])

  useEffect(siguiente, [siguiente])

  const alConfirmar = useCallback(
    (etiqueta: string) => {
      if (bloqueado.current || etiqueta !== objetivo) return
      bloqueado.current = true
      setAcierto(true)
      setAciertos((n) => n + 1)
      setIntentos((n) => n + 1)
      setRacha((n) => n + 1)
      window.setTimeout(siguiente, 1100)
    },
    [objetivo, siguiente],
  )

  const saltar = () => {
    setIntentos((n) => n + 1)
    setRacha(0)
    siguiente()
  }

  return { filtro, setFiltro, objetivo, aciertos, intentos, racha, acierto, saltar, alConfirmar, opciones }
}

type Practica = ReturnType<typeof usePractica>

export function PanelPracticar({ pr, rec, config }: { pr: Practica; rec: EstadoReconocedor; config?: ConfigModelo }) {
  const [imgFalla, setImgFalla] = useState<string | null>(null)
  const idx = pr.objetivo && config ? config.etiquetas.indexOf(pr.objetivo) : -1
  const pObjetivo = idx >= 0 && rec.probs ? rec.probs[idx] : 0

  return (
    <div className="panel-cuerpo">
      <section>
        <div className="filtros" role="radiogroup" aria-label="Qué practicar">
          {(['letra', 'numero', 'palabra', 'todas'] as Filtro[]).map((f) => (
            <button
              key={f}
              role="radio"
              aria-checked={pr.filtro === f}
              className={pr.filtro === f ? 'activo' : ''}
              onClick={() => pr.setFiltro(f)}
            >
              {{ letra: 'Letras', numero: 'Números', palabra: 'Palabras', todas: 'Todas' }[f]}
            </button>
          ))}
        </div>
      </section>

      {pr.objetivo ? (
        <section className={`reto ${pr.acierto ? 'reto-ok' : ''}`}>
          <p className="reto-pide">Haga la seña de</p>
          <p className="reto-sena">{nombreSena(config, pr.objetivo)}</p>
          {imgFalla !== pr.objetivo && (
            <img
              className="reto-img"
              src={rutaPublica(`referencias/${encodeURIComponent(pr.objetivo)}.jpg`)}
              alt={`Referencia de la seña ${pr.objetivo}`}
              onError={() => setImgFalla(pr.objetivo)}
            />
          )}
          <div className="reto-medidor" aria-label="Qué tan cerca está">
            <span style={{ width: `${Math.round(pObjetivo * 100)}%` }} />
          </div>
          <p className="ayuda">{pr.acierto ? '¡Correcto!' : `Coincidencia: ${Math.round(pObjetivo * 100)}%`}</p>
          <button onClick={pr.saltar}>Saltar</button>
        </section>
      ) : (
        <p className="ayuda">No hay señas de este tipo en el modelo.</p>
      )}

      <section className="marcador">
        <div>
          <strong>{pr.aciertos}</strong>
          <span>aciertos</span>
        </div>
        <div>
          <strong>{pr.intentos}</strong>
          <span>intentos</span>
        </div>
        <div>
          <strong>{pr.racha}</strong>
          <span>racha</span>
        </div>
      </section>
    </div>
  )
}
