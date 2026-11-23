import { useEffect, useState } from 'react'
import { cargarClasificador, type Clasificador } from '../lib/clasificador'

let carga: Promise<Clasificador | null> | undefined

export function useClasificador() {
  const [clf, setClf] = useState<Clasificador | null>(null)
  const [estado, setEstado] = useState('Cargando el reconocedor…')

  useEffect(() => {
    let cancelado = false
    ;(carga ??= cargarClasificador()).then((modelo) => {
      if (cancelado) return
      setClf(modelo)
      setEstado(modelo ? '' : 'Faltan los archivos del modelo GRU.')
    }).catch((error: unknown) => {
      if (!cancelado) setEstado(`No se pudo cargar el modelo GRU. ${String(error)}`)
    })
    return () => { cancelado = true }
  }, [])

  return { clf, estado }
}
