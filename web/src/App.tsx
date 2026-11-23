import { useState } from 'react'
import { PanelAjustes } from './components/PanelAjustes'
import { PanelCapturar, SuperposicionCaptura, useCaptura } from './components/PanelCapturar'
import { PanelPracticar, usePractica } from './components/PanelPracticar'
import { PanelTraducir, useTexto } from './components/PanelTraducir'
import { Visor } from './components/Visor'
import { useAjustes } from './hooks/useAjustes'
import { useClasificador } from './hooks/useClasificador'
import { useReconocedor } from './hooks/useReconocedor'
import { useVision } from './hooks/useVision'
import { nombreSena } from './lib/clasificador'
import type { ModoReconocimiento } from './lib/reconocimientoHibrido'
import { activarAudioRecocha, reproducirRecocha } from './lib/voz'

const practicaLibre = new URLSearchParams(location.search).has('practica')
const hibrido = import.meta.env.MODE === 'hibrido' && !practicaLibre

export default function App() {
  const [camara, setCamara] = useState(false)
  const [reconocer, setReconocer] = useState(false)
  const [verAjustes, setVerAjustes] = useState(false)
  const [verCaptura, setVerCaptura] = useState(false)
  const [modo, setModo] = useState<ModoReconocimiento>(hibrido ? 'deletreo' : 'general')
  const [usarBosque, setUsarBosque] = useState(true)
  const [recocha, setRecocha] = useState(false)
  const { ajustes, cambiar, restablecer } = useAjustes()
  const { clf, estado: estadoModelo } = useClasificador()
  const config = clf?.config
  const vision = useVision(ajustes.mostrarPuntos, camara)
  const captura = useCaptura(vision, ajustes.ventanaMs)
  const traduccion = useTexto(config, ajustes.autoHablar)
  const practica = usePractica(config)
  const rec = useReconocedor(
    vision, clf, ajustes, reconocer && captura.fase.tipo === 'quieto',
    practicaLibre ? practica.alConfirmar : (etiqueta) => {
      traduccion.agregar(etiqueta, hibrido && recocha && reproducirRecocha(etiqueta))
    },
    modo, usarBosque,
  )
  const candidato = rec.top[0]

  function alternarCamara() {
    if (camara) {
      setReconocer(false)
      captura.detener()
    }
    setCamara(!camara)
  }

  return (
    <section className="traductor" aria-labelledby="titulo-traductor">
      <div className="traductor-camara">
        <h2 id="titulo-traductor">{practicaLibre ? 'Práctica libre' : 'Traductor'}</h2>
        <p>Coloca tus manos frente a la cámara y mantén la seña un momento.</p>
        {hibrido && (
          <div className="controles-modo">
            <label>
              Modo de reconocimiento
              <select value={modo} onChange={(e) => setModo(e.target.value === 'palabras' ? 'palabras' : 'deletreo')}>
                <option value="deletreo">Deletreo</option>
                <option value="palabras">Palabras y señas</option>
              </select>
            </label>
            {modo === 'deletreo' && (
              <label>
                <input type="checkbox" checked={usarBosque} onChange={(e) => setUsarBosque(e.target.checked)} />
                Preferir Random Forest
              </label>
            )}
            <label>
              <input type="checkbox" role="switch" checked={recocha}
                aria-describedby={recocha && modo === 'deletreo' ? 'ayuda-recocha' : undefined}
                onChange={(e) => {
                  const activo = e.currentTarget.checked
                  setRecocha(activo)
                  if (activo) void activarAudioRecocha().catch((error: unknown) => {
                    setRecocha(false)
                    console.warn('No se pudo habilitar Modo recocha.', error)
                  })
                }} />
              Modo recocha
            </label>
            {recocha && modo === 'deletreo' && (
              <small id="ayuda-recocha">Selecciona Palabras y señas para usar Modo recocha.</small>
            )}
          </div>
        )}
        <Visor vision={vision} espejo={ajustes.espejo}>
          <SuperposicionCaptura fase={captura.fase} />
        </Visor>
        <div className="panel-landmarks" aria-label="Estado de detección">
          {([
            ['Mano derecha', vision.presencia.derecha],
            ['Mano izquierda', vision.presencia.izquierda],
            ['Cuerpo', vision.presencia.cuerpo],
          ] as const).map(([nombre, detectado]) => (
            <span className="estado-landmark" key={nombre}>{nombre}: <strong className={detectado ? 'detectado' : 'no-detectado'}>
              {detectado ? 'Sí' : 'No'}
            </strong></span>
          ))}
        </div>
        <div className="controles-traductor">
          <button type="button" onClick={alternarCamara}>
            {camara ? 'Apagar cámara' : 'Activar cámara'}
          </button>
          <button type="button" onClick={() => setReconocer(!reconocer)} disabled={!clf || vision.estado !== 'listo'}>
            {reconocer ? 'Detener reconocimiento' : 'Iniciar reconocimiento'}
          </button>
          <button type="button" className="secundario" onClick={() => setVerAjustes(true)}>Ajustes</button>
        </div>
        {estadoModelo && <p role="status">{estadoModelo}</p>}
        {rec.estado === 'error' && <p role="alert">La predicción falló. Detén e inicia el reconocimiento para reintentar.</p>}
        {rec.estado === 'sin-manos' && <p>Coloca tus manos dentro del cuadro.</p>}
      </div>
      <div className="traductor-resultados">
        <div className="resumen-reconocimiento">
          <p id="resultado">
            Seña detectada: {candidato ? `${nombreSena(config, candidato.etiqueta)} · ${Math.round(candidato.p * 100)}%` : '—'}
            {hibrido && candidato?.motor && ` · ${candidato.motor === 'RF' ? 'Random Forest' : 'GRU'}`}
          </p>
          {candidato && <progress className="confirmacion" value={rec.progreso} max={1} aria-label="Confirmación de la seña" />}
          <label className="voz-automatica">
            <input type="checkbox" checked={ajustes.autoHablar} onChange={(e) => cambiar('autoHablar', e.target.checked)} />
            Voz automática
          </label>
        </div>
        {practicaLibre
          ? <PanelPracticar pr={practica} rec={rec} config={config} />
          : <PanelTraducir rec={rec} config={config} ajustes={ajustes} traduccion={traduccion} />}
        <div className="captura-datos">
          <button type="button" id="capturarSena" aria-expanded={verCaptura}
            onClick={() => {
              if (verCaptura) captura.detener()
              setVerCaptura(!verCaptura)
            }}>
            {verCaptura ? 'Ocultar captura' : 'Capturar seña'}
          </button>
          {verCaptura && <PanelCapturar cap={captura} etiquetasModelo={config?.etiquetas ?? []} />}
        </div>
      </div>
      {verAjustes && (
        <PanelAjustes ajustes={ajustes} cambiar={cambiar} restablecer={restablecer} cerrar={() => setVerAjustes(false)} />
      )}
    </section>
  )
}
