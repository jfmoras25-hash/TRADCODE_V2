import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type HandLandmarkerResult,
  type PoseLandmarkerResult,
} from '@mediapipe/tasks-vision'
import { rutaPublica } from './rutas'
import { asignarManos, type CuadroCrudo, type Punto3, type Punto4 } from './features'

const REMOTO = {
  manos: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}

/** Usa el modelo local (public/models) si existe; si no, el de Google. */
async function rutaModelo(nombre: string, remoto: string): Promise<string> {
  const local = rutaPublica(`models/${nombre}`)
  try {
    const r = await fetch(local, { method: 'HEAD' })
    const tipo = r.headers.get('content-type') ?? ''
    if (r.ok && !tipo.includes('text/html')) return local
  } catch {
    /* sin modelo local */
  }
  return remoto
}

export interface Detectores {
  manos: HandLandmarker
  pose: PoseLandmarker
  delegado: 'GPU' | 'CPU'
}

export async function crearDetectores(): Promise<Detectores> {
  const fileset = await FilesetResolver.forVisionTasks(rutaPublica('mediapipe-wasm'))
  const [rm, rp] = await Promise.all([
    rutaModelo('hand_landmarker.task', REMOTO.manos),
    rutaModelo('pose_landmarker_lite.task', REMOTO.pose),
  ])
  const crear = async (delegado: 'GPU' | 'CPU') => {
    const manos = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: rm, delegate: delegado },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })
    const pose = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: rp, delegate: delegado },
      runningMode: 'VIDEO',
      numPoses: 1,
    })
    return { manos, pose, delegado }
  }
  try {
    return await crear('GPU')
  } catch (e) {
    console.warn('GPU no disponible para MediaPipe, usando CPU', e)
    return await crear('CPU')
  }
}

export function aCuadroCrudo(hr: HandLandmarkerResult, pr: PoseLandmarkerResult): CuadroCrudo {
  const manos = hr.landmarks.map((lms, i) => ({
    score: hr.handedness[i]?.[0]?.score ?? 0,
    lado: hr.handedness[i]?.[0]?.categoryName ?? 'Right',
    puntos: lms.map((l) => [l.x, l.y, l.z] as Punto3),
  }))
  const pl = pr.landmarks[0]
  return {
    manos: asignarManos(manos),
    pose: pl ? pl.map((l) => [l.x, l.y, l.z, l.visibility ?? 1] as Punto4) : null,
  }
}
