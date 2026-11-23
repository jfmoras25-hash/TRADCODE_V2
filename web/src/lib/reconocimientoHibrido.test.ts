import { describe, expect, it } from 'vitest'
import config from '../../public/model/config.json'
import { tipoSena } from './clasificador'
import { LETRAS_RF, seleccionarCandidatos, type Candidato } from './reconocimientoHibrido'

const gru = (etiqueta: string, p = 0.8): Candidato => ({ etiqueta, p, motor: 'GRU' })
const rf = (etiqueta: string, p = 0.81): Candidato => ({ etiqueta, p, motor: 'RF' })

describe('selección entre GRU y Random Forest', () => {
  it('prefiere el bosque para sus diez letras cuando pasa el umbral', () => {
    expect(LETRAS_RF.size).toBe(10)
    for (const e of LETRAS_RF) {
      expect(seleccionarCandidatos([gru(e, 0.99)], [rf(e)], 'deletreo', 0.7).mejor).toEqual(rf(e))
    }
  })

  it('mantiene las 27 letras de la GRU, incluso frente a un bosque confiado', () => {
    const letras = config.etiquetas.filter((e) => tipoSena(e) === 'letra')
    expect(letras).toHaveLength(27)
    for (const e of letras) {
      const candidato = gru(e)
      expect(seleccionarCandidatos([candidato], [], 'deletreo', 0.7).mejor).toBe(candidato)
      if (!LETRAS_RF.has(e)) {
        expect(seleccionarCandidatos([candidato], [rf('A', 0.99)], 'deletreo', 0.7).mejor).toBe(candidato)
      }
    }
  })

  it('usa la GRU si el bosque está bajo el umbral o gana BIEN o MAL', () => {
    for (const bosque of [[rf('A', 0.69)], [rf('BIEN'), rf('A', 0.19)], [rf('MAL')]]) {
      expect(seleccionarCandidatos([gru('B')], bosque, 'deletreo', 0.7).mejor).toEqual(gru('B'))
    }
  })

  it('conserva todas las palabras y números de la GRU activa', () => {
    const otras = config.etiquetas.filter((e) => tipoSena(e) !== 'letra')
    expect(otras).toHaveLength(20)
    expect(otras).toEqual(expect.arrayContaining(['HOLA', 'BUENAS', 'TARDES', 'NOMBRE']))
    for (const etiqueta of otras) {
      const palabra = gru(etiqueta)
      const salida = seleccionarCandidatos([palabra, gru('A', 0.1)], [rf('A')], 'palabras', 0.7)
      expect(salida).toEqual({ mejor: palabra, alternativas: [palabra] })
    }
  })

  it('se abstiene si el ganador está fuera del modo y conserva el modo general', () => {
    expect(seleccionarCandidatos([gru('A'), gru('HOLA', 0.19)], [], 'palabras', 0.7).mejor).toBeUndefined()
    expect(seleccionarCandidatos([gru('HOLA'), gru('A', 0.19)], [rf('BIEN')], 'deletreo', 0.7).mejor).toBeUndefined()
    expect(seleccionarCandidatos([gru('A')], [rf('B')], 'general', 0.7).mejor).toEqual(gru('A'))
  })
})
