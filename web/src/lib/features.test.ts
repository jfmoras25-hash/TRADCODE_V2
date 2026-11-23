import { describe, expect, it } from 'vitest'
import casos from './paridad.casos.json'
import { asignarManos, caracteristicasCuadro, N_FEATURES, type CuadroCrudo } from './features'
import { BufferCuadros, indicesRemuestreo } from './secuencia'

describe('paridad de características con Python', () => {
  it(`coincide en ${casos.length} cuadros`, () => {
    let peor = 0
    for (const c of casos as { aspect: number; cuadro: CuadroCrudo; esperado: number[] }[]) {
      const f = caracteristicasCuadro(c.cuadro, c.aspect)
      expect(f.length).toBe(N_FEATURES)
      for (let i = 0; i < N_FEATURES; i++) peor = Math.max(peor, Math.abs(f[i] - c.esperado[i]))
    }
    expect(peor).toBeLessThan(1e-4)
  })
})

describe('asignación de manos', () => {
  const p = Array.from({ length: 21 }, () => [0.5, 0.5, 0] as [number, number, number])
  it('dos manos con la misma etiqueta van a ranuras distintas', () => {
    const r = asignarManos([
      { score: 0.6, lado: 'Right', puntos: p },
      { score: 0.9, lado: 'Right', puntos: p },
    ])
    expect(r[0]).not.toBeNull()
    expect(r[1]).not.toBeNull()
  })
  it('una mano izquierda va a la ranura 1', () => {
    const r = asignarManos([{ score: 0.9, lado: 'Left', puntos: p }])
    expect(r[0]).toBeNull()
    expect(r[1]).not.toBeNull()
  })
})

describe('secuencias', () => {
  it('remuestreo igual que Python', () => {
    expect(indicesRemuestreo(6, 6)).toEqual([0, 1, 2, 3, 4, 5])
    expect(indicesRemuestreo(30, 6)).toEqual([0, 6, 12, 17, 23, 29])
  })
  it('muestrea T cuadros de la ventana', () => {
    const b = new BufferCuadros()
    for (let t = 0; t <= 1200; t += 33) b.agregar(t, { manos: [null, null], pose: null })
    expect(b.muestrear(6, 1000, 1200)?.length).toBe(6)
  })
})
