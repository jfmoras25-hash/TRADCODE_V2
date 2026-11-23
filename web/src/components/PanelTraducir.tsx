import { useCallback, useRef, useState } from 'react'
import type { Ajustes } from '../hooks/useAjustes'
import type { EstadoReconocedor } from '../hooks/useReconocedor'
import type { ConfigModelo } from '../lib/clasificador'
import { editarTraduccion, TRADUCCION_INICIAL, type AccionTraduccion } from '../lib/traduccion'
import { hablar, hablarSena } from '../lib/voz'
import { TopTres } from './Prediccion'

export function useTexto(config: ConfigModelo | undefined, autoHablar: boolean) {
  const [estado, setEstado] = useState(TRADUCCION_INICIAL)
  const actual = useRef(estado)
  const aplicar = useCallback(
    (accion: AccionTraduccion, omitirVoz = false) => {
      const resultado = editarTraduccion(actual.current, accion, config)
      actual.current = resultado.estado
      setEstado(resultado.estado)
      if (autoHablar && resultado.voz && !omitirVoz) hablarSena(resultado.voz)
    },
    [config, autoHablar],
  )
  const agregar = useCallback((etiqueta: string, omitirVoz = false) =>
    aplicar({ tipo: 'reconocer', etiqueta }, omitirVoz), [aplicar])
  return { ...estado, agregar, aplicar }
}

interface Props {
  rec: EstadoReconocedor
  config?: ConfigModelo
  ajustes: Ajustes
  traduccion: ReturnType<typeof useTexto>
}

export function PanelTraducir({ rec, config, ajustes, traduccion }: Props) {
  const { texto, deletreo, aplicar } = traduccion
  return (
    <div className="panel-cuerpo">
      <section>
        <h2>Texto reconocido</h2>
        <div className="texto-salida" aria-live="polite">
          {texto || <span className="placeholder">Aquí aparecerá el texto definitivo.</span>}
        </div>
        <div className="deletreo">
          <h3 id="titulo-deletreo">Palabra en formación</h3>
          <output aria-labelledby="titulo-deletreo">
            {deletreo || <span className="placeholder">Deletrea una palabra y pulsa ESPACIO para añadirla.</span>}
          </output>
        </div>
        <div className="botonera">
          <button onClick={() => aplicar({ tipo: 'espacio' })}>ESPACIO</button>
          <button onClick={() => aplicar({ tipo: 'borrar' })}>Borrar</button>
          <button onClick={() => aplicar({ tipo: 'limpiar' })}>Limpiar</button>
          <button className="primario" onClick={() => hablar(texto)} disabled={!texto.trim()}>
            Escuchar texto
          </button>
        </div>
      </section>

      <section>
        <h2>Señas similares</h2>
        <TopTres rec={rec} config={config} umbral={ajustes.umbral} />
        <p className="ayuda">
          Una seña se acepta cuando supera {Math.round(ajustes.umbral * 100)}% de confianza durante{' '}
          {ajustes.sostenerMs} ms. Para repetir la misma letra, baje la mano y vuelva a hacerla.
        </p>
      </section>
    </div>
  )
}
