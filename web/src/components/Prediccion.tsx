import type { ConfigModelo } from '../lib/clasificador'
import { nombreSena } from '../lib/clasificador'
import type { EstadoReconocedor } from '../hooks/useReconocedor'

/** Lista de las 3 señas más probables con su confianza. */
export function TopTres({ rec, config, umbral }: { rec: EstadoReconocedor; config?: ConfigModelo; umbral: number }) {
  const filas = rec.top.length ? rec.top : []
  return (
    <ol className="top3" aria-label="Señas más probables">
      {filas.map((c) => (
        <li key={c.etiqueta}>
          <span className="top3-etq">{nombreSena(config, c.etiqueta)}</span>
          <span className="top3-barra">
            <span className={c.p >= umbral ? 'alta' : ''} style={{ width: `${Math.round(c.p * 100)}%` }} />
          </span>
          <span className="top3-p">{Math.round(c.p * 100)}%</span>
        </li>
      ))}
      {!filas.length && <li className="vacio">Esperando una seña…</li>}
    </ol>
  )
}
