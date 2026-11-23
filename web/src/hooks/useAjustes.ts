import { useEffect, useState } from 'react'

export interface Ajustes {
  ventanaMs: number // duración de la ventana de la que se toman los 6 cuadros
  umbral: number // confianza mínima para aceptar una seña
  sostenerMs: number // tiempo que debe mantenerse la predicción
  mostrarPuntos: boolean
  espejo: boolean
  autoHablar: boolean
}

export const AJUSTES_INICIALES: Ajustes = {
  ventanaMs: 1200,
  umbral: 0.7,
  sostenerMs: 500,
  mostrarPuntos: true,
  espejo: true,
  autoHablar: true,
}

const CLAVE = 'lsc-ajustes-v1'

function leer(): Ajustes {
  try {
    const raw = localStorage.getItem(CLAVE)
    if (raw) return { ...AJUSTES_INICIALES, ...JSON.parse(raw) }
  } catch {
    /* almacenamiento no disponible */
  }
  return AJUSTES_INICIALES
}

export function useAjustes() {
  const [ajustes, setAjustes] = useState<Ajustes>(leer)
  useEffect(() => {
    try {
      localStorage.setItem(CLAVE, JSON.stringify(ajustes))
    } catch {
      /* almacenamiento no disponible */
    }
  }, [ajustes])
  const cambiar = <K extends keyof Ajustes>(k: K, v: Ajustes[K]) => setAjustes((a) => ({ ...a, [k]: v }))
  return { ajustes, cambiar, restablecer: () => setAjustes(AJUSTES_INICIALES) }
}
