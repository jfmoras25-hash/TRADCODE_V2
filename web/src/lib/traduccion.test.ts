import { describe, expect, it } from 'vitest'
import config from '../../public/model/config.json'
import { tipoSena } from './clasificador'
import { editarTraduccion, TRADUCCION_INICIAL } from './traduccion'

describe('deletreo del traductor', () => {
  it('distingue las letras, palabras y números de las 47 clases reales', () => {
    expect(config.etiquetas).toHaveLength(47)
    expect(config.etiquetas.filter((etiqueta) => tipoSena(etiqueta) === 'letra')).toHaveLength(27)
    expect(tipoSena('NN')).toBe('letra')
    expect(tipoSena('HOLA')).toBe('palabra')
    expect(tipoSena('NOMBRE')).toBe('palabra')
    expect(tipoSena('10')).toBe('numero')
    expect(config.etiquetas).not.toContain('MI')
    expect(config.etiquetas).not.toContain('NOMBRE_ES')
  })

  it('mantiene JUAN temporal y lo añade al texto al pulsar ESPACIO', () => {
    let estado = TRADUCCION_INICIAL
    for (const etiqueta of ['HOLA', 'NOMBRE']) {
      const resultado = editarTraduccion(estado, { tipo: 'reconocer', etiqueta }, config)
      expect(resultado.voz).toBe(etiqueta)
      estado = resultado.estado
    }
    for (const [etiqueta, deletreo] of [['J', 'J'], ['U', 'JU'], ['A', 'JUA'], ['N', 'JUAN']]) {
      const resultado = editarTraduccion(estado, { tipo: 'reconocer', etiqueta }, config)
      expect(resultado).toEqual({ estado: { texto: 'HOLA NOMBRE', deletreo } })
      estado = resultado.estado
    }
    expect(editarTraduccion(estado, { tipo: 'espacio' }, config)).toEqual({
      estado: { texto: 'HOLA NOMBRE JUAN', deletreo: '' },
      voz: 'JUAN',
    })
  })

  it('expande palabras con guiones bajos y trata MI como palabra si llega como etiqueta', () => {
    const mi = editarTraduccion({ texto: 'HOLA', deletreo: '' }, { tipo: 'reconocer', etiqueta: 'MI' })
    expect(mi.estado.texto).toBe('HOLA MI')
    expect(editarTraduccion(mi.estado, { tipo: 'reconocer', etiqueta: 'NOMBRE_ES' }).estado.texto)
      .toBe('HOLA MI NOMBRE ES')
  })

  it('solo inserta un separador y conserva la concatenación de números', () => {
    const espacio = { tipo: 'espacio' } as const
    expect(editarTraduccion(TRADUCCION_INICIAL, espacio).estado).toBe(TRADUCCION_INICIAL)
    const uno = editarTraduccion({ texto: 'HOLA', deletreo: '' }, { tipo: 'reconocer', etiqueta: '1' }).estado
    const numero = editarTraduccion(uno, { tipo: 'reconocer', etiqueta: '4' }).estado
    const separado = editarTraduccion(numero, espacio).estado
    expect(numero.texto).toBe('HOLA 14')
    expect(separado.texto).toBe('HOLA 14 ')
    expect(editarTraduccion(separado, espacio).estado).toBe(separado)
  })

  it('permite repetir letras confirmadas y muestra la Ñ del modelo', () => {
    const resultado = ['L', 'L', 'NN'].reduce((estado, etiqueta) =>
      editarTraduccion(estado, { tipo: 'reconocer', etiqueta }, config).estado, TRADUCCION_INICIAL)
    expect(resultado).toEqual({ texto: '', deletreo: 'LLÑ' })
  })

  it('corrige primero el deletreo y limpia ambas áreas', () => {
    const estado = { texto: 'HOLA', deletreo: 'JU' }
    expect(editarTraduccion(estado, { tipo: 'borrar' }).estado).toEqual({ texto: 'HOLA', deletreo: 'J' })
    expect(editarTraduccion({ ...estado, deletreo: '' }, { tipo: 'borrar' }).estado.texto).toBe('HOL')
    expect(editarTraduccion(estado, { tipo: 'limpiar' }).estado).toEqual(TRADUCCION_INICIAL)
  })
})
