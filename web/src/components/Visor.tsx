import type { ReactNode } from 'react'
import type { Vision } from '../hooks/useVision'

export function Visor({ vision, espejo, clase = 'visor', children }: {
  vision: Vision
  espejo: boolean
  clase?: string
  children?: ReactNode
}) {
  return (
    <div className={`${clase} visor-lsc${espejo ? ' espejo' : ''}`}>
      <video ref={vision.videoRef} playsInline muted />
      <canvas ref={vision.canvasRef} aria-hidden="true" />
      {vision.estado !== 'listo' && (
        <div className="mensaje-camara" role="status">
          {vision.estado === 'error' ? vision.error : vision.estado === 'cargando' ? 'Iniciando cámara…' : 'Cámara apagada'}
        </div>
      )}
      {children}
    </div>
  )
}
