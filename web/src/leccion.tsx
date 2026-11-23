import { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Visor } from './components/Visor'
import { useAjustes } from './hooks/useAjustes'
import { useClasificador } from './hooks/useClasificador'
import { useReconocedor } from './hooks/useReconocedor'
import { useVision } from './hooks/useVision'
import { nombreSena } from './lib/clasificador'
import { ALFABETO, registrarPractica } from './lib/progreso'
import { rutaPublica } from './lib/rutas'
import { hablar } from './lib/voz'
import './tradcode.css'

const parametro = (new URLSearchParams(location.search).get('letra') ?? 'A').toUpperCase()
const letra = ALFABETO.includes(parametro) ? parametro : 'A'
const posicion = ALFABETO.indexOf(letra)
const anterior = ALFABETO[posicion - 1]
const siguiente = ALFABETO[posicion + 1]
const enlaceLeccion = (destino: string) => `leccion.html?letra=${encodeURIComponent(destino)}`

function Mano3D() {
  const host = useRef<HTMLDivElement>(null)
  const movimiento = useRef<(() => Promise<boolean>) | null>(null)
  const [mensaje, setMensaje] = useState('Cargando el modelo 3D…')
  const [poseLista, setPoseLista] = useState<{ asl: boolean; animada: boolean } | null>(null)
  const [reproduciendo, setReproduciendo] = useState(false)
  const [falloMovimiento, setFalloMovimiento] = useState(false)

  useEffect(() => {
    const cancel = new AbortController()
    let dispose: (() => void) | undefined
    async function mostrar() {
      const catalogo = await fetch(rutaPublica('modelos/poses/index.json'), { signal: cancel.signal, cache: 'no-cache' })
      if (!catalogo.ok) throw new Error('No se pudo cargar el catálogo de poses')
      const disponibles = await catalogo.json() as Record<string, unknown> | null
      if (!disponibles || typeof disponibles !== 'object' || Array.isArray(disponibles)) {
        throw new Error('El catálogo de poses es inválido')
      }
      if (cancel.signal.aborted) return
      const archivo = disponibles[letra]
      if (typeof archivo !== 'string') {
        setMensaje('Esta letra todavía no tiene una pose 3D disponible.')
        return
      }
      const response = await fetch(rutaPublica(`modelos/poses/${archivo}`), { signal: cancel.signal })
      if (!response.ok) throw new Error('No se pudo cargar la pose')
      const pose = await response.json() as { motion?: unknown } | null
      const { createHand } = await import('./modelo3d/src/hand.js')
      if (cancel.signal.aborted || !host.current) return
      const mano = createHand(host.current, { mode: 'light', background: false, numbers: false, controls: true })
      dispose = () => mano.dispose()
      if (!await mano.ready) throw new Error('No se pudo cargar la mano 3D')
      if (cancel.signal.aborted) return
      mano.setPose(pose, { instant: true })
      const animada = Array.isArray(pose?.motion) && pose.motion.length > 0
      if (animada) movimiento.current = async () => mano.playMotion(pose)
      setPoseLista({ asl: archivo.startsWith('eidos-asl/'), animada })
      setMensaje('')
    }
    mostrar().catch(() => { if (!cancel.signal.aborted) setMensaje('El modelo 3D no está disponible en este navegador.') })
    return () => {
      cancel.abort()
      movimiento.current = null
      dispose?.()
    }
  }, [])

  async function reproducirMovimiento() {
    const reproducir = movimiento.current
    if (!reproducir || reproduciendo) return
    setReproduciendo(true)
    setFalloMovimiento(false)
    try {
      await reproducir()
    } catch {
      if (movimiento.current === reproducir) setFalloMovimiento(true)
    } finally {
      if (movimiento.current === reproducir) setReproduciendo(false)
    }
  }

  return (
    <>
      <div className="visor-modelo" id="modelo3d">
        <div className="manos-3d" ref={host} aria-label={`Mano 3D para la letra ${letra}`} aria-hidden={!!mensaje} />
        {mensaje && <p role="status">{mensaje}</p>}
      </div>
      {poseLista && (
        <>
          <p className="ayuda-panel">Gira el modelo para observar los dedos desde distintos ángulos.</p>
          {poseLista.asl && <p className="ayuda-panel">Referencia 3D de lengua de señas estadounidense (ASL) · Eidos</p>}
          {poseLista.animada && (
            <button type="button" disabled={reproduciendo} onClick={() => void reproducirMovimiento()}>
              {reproduciendo ? 'Reproduciendo movimiento…' : 'Reproducir movimiento'}
            </button>
          )}
          {falloMovimiento && <p role="status">No se pudo reproducir el movimiento.</p>}
        </>
      )}
    </>
  )
}

function Leccion() {
  const { ajustes } = useAjustes()
  const { clf, estado: estadoModelo } = useClasificador()
  const [camara, setCamara] = useState(false)
  const [completa, setCompleta] = useState(false)
  const [guardado, setGuardado] = useState(true)
  const [aviso, setAviso] = useState('')
  const terminada = useRef(false)
  const sonido = useRef<HTMLAudioElement | null>(null)
  const dialogo = useRef<HTMLDialogElement>(null)
  const vision = useVision(ajustes.mostrarPuntos, camara)

  useEffect(() => () => { sonido.current?.pause() }, [])
  useEffect(() => {
    if (completa) dialogo.current?.showModal()
  }, [completa])

  function confirmar(etiqueta: string) {
    if (terminada.current) return
    if (nombreSena(clf?.config, etiqueta) !== letra) {
      const mensaje = `Esa no es la letra ${letra}. Inténtalo otra vez.`
      setAviso(mensaje)
      if (ajustes.autoHablar) hablar(mensaje)
      return
    }

    terminada.current = true
    setGuardado(registrarPractica(letra))
    setCompleta(true)
    setCamara(false)
    const felicitar = () => {
      if (ajustes.autoHablar) hablar(`¡Felicitaciones! Has completado la lección de la letra ${letra}.`)
    }
    const audio = new Audio(rutaPublica('sonidos/mision-completada.mp3'))
    sonido.current = audio
    audio.volume = 0.8
    audio.addEventListener('ended', felicitar, { once: true })
    audio.play().catch(felicitar)
  }

  const rec = useReconocedor(vision, clf, ajustes, camara && !completa, confirmar)
  const candidato = rec.top[0]
  const mensaje = rec.estado === 'error'
    ? 'La predicción falló. Apaga y vuelve a activar la cámara.'
    : completa
      ? `Has realizado correctamente la letra ${letra}.`
      : aviso || (rec.estado === 'sin-manos' ? 'Coloca tu mano en el cuadro.' : 'Esperando una seña…')

  return (
    <>
      <section className="encabezado-leccion">
        <a href="abecedario.html" className="volver-abecedario">← Volver al abecedario</a>
        <p className="etiqueta">ABECEDARIO</p>
        <h1>Aprende la letra {letra}</h1>
        <p>Observa la posición de la mano y practica la seña frente a tu cámara.</p>
      </section>
      <nav className="navegacion-lecciones" aria-label="Navegación entre letras">
        {anterior && <a rel="prev" href={enlaceLeccion(anterior)}>← Anterior: {anterior}</a>}
        {siguiente && <a rel="next" href={enlaceLeccion(siguiente)}>Siguiente: {siguiente} →</a>}
      </nav>
      <section className="informacion-letra">
        <div className="letra-destacada"><span>{letra}</span></div>
        <div className="descripcion-letra">
          <h2>¿Cómo realizar esta seña?</h2>
          <p>Observa la posición de los dedos, la orientación de la mano y el movimiento.</p>
          <p>Cuando estés listo, intenta reproducir la seña frente a la cámara.</p>
        </div>
      </section>
      <section className="zona-practica">
        <div className="panel-practica">
          <div className="titulo-panel"><h2>Observa la seña</h2></div>
          <Mano3D />
        </div>
        <div className="panel-practica">
          <div className="titulo-panel"><h2>Practica la seña</h2></div>
          <Visor vision={vision} espejo={ajustes.espejo} clase="visor-camara" />
          {estadoModelo && <p role="status">{estadoModelo}</p>}
          <button type="button" disabled={!clf || completa} onClick={() => {
            setAviso('')
            setCamara(!camara)
          }}>{camara ? 'Apagar cámara' : 'Activar cámara'}</button>
          <p className="ayuda-panel">Mantén la seña de la letra {letra} frente a la cámara.</p>
        </div>
      </section>
      <section className="panel-resultados">
        <div className="titulo-panel"><h2>Resultado de tu práctica</h2></div>
        <div className="resultado-reconocimiento">
          <div className="informacion-resultado">
            <span>SEÑA DETECTADA</span>
            <h3>{completa ? letra : candidato ? nombreSena(clf?.config, candidato.etiqueta) : '—'}</h3>
            <p role="status">{mensaje}</p>
          </div>
          <div className="informacion-confianza">
            <span>CONFIANZA DEL RECONOCIMIENTO</span>
            <h3>{candidato ? Math.round(candidato.p * 100) : 0}%</h3>
          </div>
        </div>
      </section>
      <dialog className="felicitacion-dialogo" ref={dialogo} aria-labelledby="titulo-felicitacion">
        <div className="contenido-felicitacion">
          <h2 id="titulo-felicitacion">Excelente trabajo</h2>
          <p>Has completado correctamente la lección de la letra {letra}.</p>
          {!guardado && <p>El navegador no pudo guardar tu progreso.</p>}
          <a href="abecedario.html" className="boton-felicitacion">Volver al abecedario</a>
        </div>
      </dialog>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<Leccion />)
