import { ALFABETO, leerProgreso } from './lib/progreso'

const { letras, practicas } = leerProgreso()
const total = ALFABETO.length
const porcentaje = Math.round(letras.length / total * 100)
const resumen = {
  'letras-aprendidas': `${letras.length} / ${total}`,
  'progreso-general': `${porcentaje}%`,
  'practicas-completadas': String(practicas),
  'porcentaje-progreso': `${porcentaje}%`,
  'detalle-letras': `${letras.length} de ${total} letras completadas`,
  'contador-letras': `${letras.length} de ${total} letras`,
}
for (const [id, texto] of Object.entries(resumen)) {
  document.getElementById(id)!.textContent = texto
}
document.getElementById('avance-progreso')!.style.width = `${porcentaje}%`

document.querySelectorAll('.letra').forEach((tarjeta) => {
  if (letras.includes(tarjeta.textContent.trim())) tarjeta.classList.add('completada')
})
