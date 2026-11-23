import type { Ajustes } from '../hooks/useAjustes'

interface Props {
  ajustes: Ajustes
  cambiar: <K extends keyof Ajustes>(k: K, v: Ajustes[K]) => void
  restablecer: () => void
  cerrar: () => void
}

export function PanelAjustes({ ajustes, cambiar, restablecer, cerrar }: Props) {
  return (
    <div className="modal-fondo" onClick={cerrar}>
      <div className="modal" role="dialog" aria-modal aria-labelledby="t-ajustes" onClick={(e) => e.stopPropagation()}>
        <h2 id="t-ajustes">Ajustes</h2>
        <label className="deslizador">
          <span>
            Duración de la seña <strong>{(ajustes.ventanaMs / 1000).toFixed(1)} s</strong>
          </span>
          <input type="range" min={500} max={2500} step={100} value={ajustes.ventanaMs}
            onChange={(e) => cambiar('ventanaMs', Number(e.target.value))} />
          <small>Ventana de la que se toman los 6 cuadros. Súbala si hace las señas despacio.</small>
        </label>
        <label className="deslizador">
          <span>
            Confianza mínima <strong>{Math.round(ajustes.umbral * 100)}%</strong>
          </span>
          <input type="range" min={0.3} max={0.98} step={0.01} value={ajustes.umbral}
            onChange={(e) => cambiar('umbral', Number(e.target.value))} />
        </label>
        <label className="deslizador">
          <span>
            Mantener la seña <strong>{ajustes.sostenerMs} ms</strong>
          </span>
          <input type="range" min={100} max={1500} step={50} value={ajustes.sostenerMs}
            onChange={(e) => cambiar('sostenerMs', Number(e.target.value))} />
        </label>
        <label className="casilla">
          <input type="checkbox" checked={ajustes.mostrarPuntos} onChange={(e) => cambiar('mostrarPuntos', e.target.checked)} />
          Dibujar puntos de manos y cuerpo
        </label>
        <label className="casilla">
          <input type="checkbox" checked={ajustes.espejo} onChange={(e) => cambiar('espejo', e.target.checked)} />
          Ver la cámara como espejo
        </label>
        <label className="casilla">
          <input type="checkbox" checked={ajustes.autoHablar} onChange={(e) => cambiar('autoHablar', e.target.checked)} />
          Leer en voz alta palabras confirmadas y números
        </label>
        <div className="botonera">
          <button onClick={restablecer}>Restablecer</button>
          <button className="primario" onClick={cerrar}>Listo</button>
        </div>
      </div>
    </div>
  )
}
