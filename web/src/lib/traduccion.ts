import { esReposo, nombreSena, tipoSena, type ConfigModelo } from './clasificador'

export interface Traduccion {
  texto: string
  deletreo: string
}

export type AccionTraduccion =
  | { tipo: 'reconocer'; etiqueta: string }
  | { tipo: 'espacio' | 'borrar' | 'limpiar' }

export const TRADUCCION_INICIAL: Traduccion = { texto: '', deletreo: '' }

function unir(texto: string, palabra: string) {
  return texto && !texto.endsWith(' ') ? `${texto} ${palabra}` : texto + palabra
}

export function editarTraduccion(
  estado: Traduccion,
  accion: AccionTraduccion,
  config?: ConfigModelo,
): { estado: Traduccion; voz?: string } {
  switch (accion.tipo) {
    case 'limpiar':
      return { estado: TRADUCCION_INICIAL }
    case 'borrar':
      return { estado: estado.deletreo
        ? { ...estado, deletreo: estado.deletreo.slice(0, -1) }
        : { ...estado, texto: estado.texto.slice(0, -1) } }
    case 'espacio':
      if (estado.deletreo) {
        return { estado: { texto: unir(estado.texto, estado.deletreo), deletreo: '' }, voz: estado.deletreo }
      }
      return { estado: estado.texto && !estado.texto.endsWith(' ')
        ? { ...estado, texto: estado.texto + ' ' }
        : estado }
    case 'reconocer': {
      if (esReposo(accion.etiqueta)) return { estado }
      const nombre = nombreSena(config, accion.etiqueta)
      const tipo = tipoSena(accion.etiqueta)
      if (tipo === 'letra') return { estado: { ...estado, deletreo: estado.deletreo + nombre } }
      // Los dígitos consecutivos conservan la posibilidad de formar un número.
      const texto = tipo === 'numero' && /\d$/.test(estado.texto)
        ? estado.texto + nombre
        : unir(estado.texto, nombre)
      return { estado: { ...estado, texto }, voz: nombre }
    }
  }
}
