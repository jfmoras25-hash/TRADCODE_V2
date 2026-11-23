export const ALFABETO = [...'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ']
const CLAVE = 'progresoTRADCODE'

export function leerProgreso(): { letras: string[]; practicas: number } {
  try {
    const datos = JSON.parse(localStorage.getItem(CLAVE) ?? '{}')
    return {
      letras: Array.isArray(datos?.letras)
        ? [...new Set<string>(datos.letras.filter((letra: string) => ALFABETO.includes(letra)))]
        : [],
      practicas: Number.isSafeInteger(datos?.practicas) && datos.practicas >= 0 ? datos.practicas : 0,
    }
  } catch {
    return { letras: [], practicas: 0 }
  }
}

export function registrarPractica(letra: string): boolean {
  const progreso = leerProgreso()
  if (!progreso.letras.includes(letra)) progreso.letras.push(letra)
  progreso.practicas++
  try {
    localStorage.setItem(CLAVE, JSON.stringify(progreso))
    return true
  } catch {
    return false
  }
}
