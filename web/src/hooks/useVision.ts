import { DrawingUtils, HandLandmarker, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { useEffect, useRef, useState } from 'react'
import type { CuadroCrudo } from '../lib/features'
import { BufferCuadros } from '../lib/secuencia'
import { aCuadroCrudo, crearDetectores, type Detectores } from '../lib/vision'

// Una sola instancia de los detectores para toda la app (crearlos tarda).
let detectoresPromesa: Promise<Detectores> | null = null
const obtenerDetectores = () => (detectoresPromesa ??= crearDetectores())

const CONEXIONES_POSE = PoseLandmarker.POSE_CONNECTIONS.filter((c) => c.start <= 22 && c.end <= 22)

export type EstadoVision = 'apagada' | 'cargando' | 'listo' | 'error'

export interface Vision {
  videoRef: React.RefObject<HTMLVideoElement | null>
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  estado: EstadoVision
  error: string | null
  fps: number
  manos: number
  presencia: { derecha: boolean; izquierda: boolean; cuerpo: boolean }
  delegado: string
  buffer: BufferCuadros
  aspectRef: React.RefObject<number>
}

const SIN_PUNTOS = { derecha: false, izquierda: false, cuerpo: false }

export function useVision(mostrarPuntos: boolean, activa = true): Vision {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [buffer] = useState(() => new BufferCuadros())
  const aspectRef = useRef(4 / 3)
  const puntosRef = useRef(mostrarPuntos)
  const [estado, setEstado] = useState<EstadoVision>(activa ? 'cargando' : 'apagada')
  const [error, setError] = useState<string | null>(null)
  const [fps, setFps] = useState(0)
  const [manos, setManos] = useState(0)
  const [presencia, setPresencia] = useState(SIN_PUNTOS)
  const [delegado, setDelegado] = useState('')

  useEffect(() => {
    puntosRef.current = mostrarPuntos
  }, [mostrarPuntos])

  useEffect(() => {
    if (!activa) {
      setEstado('apagada')
      setError(null)
      setManos(0)
      setPresencia(SIN_PUNTOS)
      setFps(0)
      return
    }
    setEstado('cargando')
    setError(null)
    let cancelado = false
    let raf = 0
    let stream: MediaStream | null = null

    async function iniciar() {
      try {
        if (!navigator.mediaDevices?.getUserMedia)
          throw new Error('Este navegador no permite usar la cámara (se necesita https o localhost).')
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false,
        })
        if (cancelado) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        const det = await obtenerDetectores().catch((e) => {
          detectoresPromesa = null // permitir reintentar al recargar
          throw new Error(
            'No se pudieron cargar los modelos de MediaPipe. Revise la conexión a internet, o ejecute ' +
              '"npm install" con internet una vez para guardarlos en public/models y usar la app sin conexión. ' +
              `(${(e as Error)?.message ?? e})`,
          )
        })
        if (cancelado) return
        setDelegado(det.delegado)
        const video = videoRef.current!
        video.srcObject = stream
        await video.play()
        if (cancelado) return
        setEstado('listo')
        bucle(det)
      } catch (e) {
        if (cancelado) return
        stream?.getTracks().forEach((t) => t.stop())
        const msg =
          e instanceof DOMException && e.name === 'NotAllowedError'
            ? 'Permiso de cámara denegado. Habilítelo en el navegador y recargue la página.'
            : e instanceof DOMException && e.name === 'NotFoundError'
              ? 'No se encontró ninguna cámara.'
              : String((e as Error)?.message ?? e)
        setError(msg)
        setEstado('error')
      }
    }

    function bucle(det: Detectores) {
      const video = videoRef.current!
      const canvas = canvasRef.current!
      const ctx = canvas.getContext('2d')!
      const dibujo = new DrawingUtils(ctx)
      let ultimoTiempoVideo = -1
      let cuadros = 0
      let inicioFps = performance.now()
      let ultimasManos = -1
      let ultimaPresencia = SIN_PUNTOS

      const paso = () => {
        if (cancelado) return
        raf = requestAnimationFrame(paso)
        if (video.readyState < 2 || video.currentTime === ultimoTiempoVideo) return
        ultimoTiempoVideo = video.currentTime
        const ahora = performance.now()
        const hr = det.manos.detectForVideo(video, ahora)
        const pr = det.pose.detectForVideo(video, ahora)
        const cuadro: CuadroCrudo = aCuadroCrudo(hr, pr)
        aspectRef.current = video.videoWidth / video.videoHeight
        buffer.agregar(ahora, cuadro)
        const [derecha, izquierda] = cuadro.manos.map(Boolean)
        const cuerpo = Boolean(cuadro.pose)
        if (derecha !== ultimaPresencia.derecha || izquierda !== ultimaPresencia.izquierda || cuerpo !== ultimaPresencia.cuerpo) {
          ultimaPresencia = { derecha, izquierda, cuerpo }
          setPresencia(ultimaPresencia)
        }

        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        if (puntosRef.current) {
          for (const p of pr.landmarks as NormalizedLandmark[][])
            dibujo.drawConnectors(p, CONEXIONES_POSE, { color: 'rgba(255,255,255,0.55)', lineWidth: 2 })
          for (const m of hr.landmarks) {
            dibujo.drawConnectors(m, HandLandmarker.HAND_CONNECTIONS, { color: '#2dd4bf', lineWidth: 3 })
            dibujo.drawLandmarks(m, { color: '#0f766e', fillColor: '#ffffff', lineWidth: 1, radius: 3 })
          }
        }

        const n = hr.landmarks.length
        if (n !== ultimasManos) {
          ultimasManos = n
          setManos(n)
        }
        cuadros++
        if (ahora - inicioFps > 1000) {
          setFps(Math.round((cuadros * 1000) / (ahora - inicioFps)))
          cuadros = 0
          inicioFps = ahora
        }
      }
      paso()
    }

    iniciar()
    return () => {
      cancelado = true
      cancelAnimationFrame(raf)
      buffer.limpiar()
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [activa, buffer])

  return { videoRef, canvasRef, estado, error, fps, manos, presencia, delegado, buffer, aspectRef }
}
