import type { CuadroCrudo } from './features'

/**
 * Guarda los cuadros recientes con su marca de tiempo y toma T cuadros
 * equiespaciados dentro de una ventana (igual que las transiciones de 6
 * cuadros de LSC70).
 */
export class BufferCuadros {
  private items: { t: number; c: CuadroCrudo }[] = []
  private readonly maxMs: number

  constructor(maxMs = 4000) {
    this.maxMs = maxMs
  }

  agregar(t: number, c: CuadroCrudo) {
    this.items.push({ t, c })
    while (this.items.length && t - this.items[0].t > this.maxMs) this.items.shift()
  }

  limpiar() {
    this.items = []
  }

  /** ¿Hubo alguna mano en los últimos `ms` milisegundos? */
  hayManos(ahora: number, ms: number): boolean {
    for (let i = this.items.length - 1; i >= 0 && ahora - this.items[i].t <= ms; i--)
      if (this.items[i].c.manos[0] || this.items[i].c.manos[1]) return true
    return false
  }

  /** Cuadros dentro de [desde, hasta]. */
  rango(desde: number, hasta: number): CuadroCrudo[] {
    return this.items.filter((x) => x.t >= desde && x.t <= hasta).map((x) => x.c)
  }

  muestrear(T: number, ventanaMs: number, ahora: number): CuadroCrudo[] | null {
    const ini = ahora - ventanaMs
    const dentro = this.items.filter((x) => x.t >= ini - 50)
    if (dentro.length < Math.min(3, T)) return null
    const out: CuadroCrudo[] = []
    let j = 0
    for (let i = 0; i < T; i++) {
      const objetivo = ini + (i * ventanaMs) / (T - 1)
      while (j + 1 < dentro.length && Math.abs(dentro[j + 1].t - objetivo) <= Math.abs(dentro[j].t - objetivo)) j++
      out.push(dentro[j].c)
    }
    return out
  }
}

/** T índices equiespaciados de n elementos (igual que indices_remuestreo en Python). */
export function indicesRemuestreo(n: number, T: number): number[] {
  return Array.from({ length: T }, (_, i) => Math.round((i * (n - 1)) / Math.max(T - 1, 1)))
}
