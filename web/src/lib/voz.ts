const RESPUESTAS_RECOCHA = new Map<string, readonly string[]>([
  ['DEDO_MEDIO', ['Vaya y métaselo por el...']],
  ['HIJUEPUTA', ['Más...', 'será su madre']],
])

let audioRecocha: AudioContext | undefined
let colaRecocha: Promise<void> | undefined
let lecturaPendiente: string | undefined

function prepararFrase(texto: string) {
  const u = new SpeechSynthesisUtterance(texto)
  const voces = speechSynthesis.getVoices()
  u.voice =
    voces.find((v) => v.lang === 'es-CO') ??
    voces.find((v) => v.lang.startsWith('es-4') || v.lang === 'es-MX' || v.lang === 'es-US') ??
    voces.find((v) => v.lang.startsWith('es')) ??
    null
  u.lang = u.voice?.lang ?? 'es-CO'
  u.rate = 0.9
  return u
}

/** Lectura en voz alta con la síntesis de voz del navegador (en español). */
export function hablar(texto: string) {
  const t = texto.trim()
  if (!t || !('speechSynthesis' in window)) return
  if (colaRecocha) {
    lecturaPendiente = t
    return
  }
  speechSynthesis.cancel()
  speechSynthesis.speak(prepararFrase(t.toLowerCase()))
}

export function hablarSena(sena: string) {
  hablar(sena === 'B' ? 'bé' : sena)
}

/** Desbloquea Web Audio desde el clic que enciende el interruptor. */
export async function activarAudioRecocha() {
  audioRecocha ??= new AudioContext()
  await audioRecocha.resume()
}

function pronunciarFragmento(texto: string) {
  return new Promise<void>((resolver, rechazar) => {
    const frase = prepararFrase(texto)
    // Continúa cuando el navegador termina de despachar el evento de voz.
    frase.onend = () => window.setTimeout(resolver, 0)
    frase.onerror = (evento) => rechazar(new Error(evento.error))
    speechSynthesis.speak(frase)
  })
}

async function beep() {
  const audio = audioRecocha
  if (!audio) throw new Error('El audio de Modo recocha no está habilitado')
  if (audio.state !== 'running') await audio.resume()
  const tono = audio.createOscillator()
  const volumen = audio.createGain()
  const inicio = audio.currentTime
  const fin = inicio + 0.22
  tono.frequency.value = 880
  volumen.gain.setValueAtTime(0, inicio)
  volumen.gain.linearRampToValueAtTime(0.18, inicio + 0.012)
  volumen.gain.setValueAtTime(0.18, fin - 0.012)
  volumen.gain.linearRampToValueAtTime(0, fin)
  tono.connect(volumen)
  volumen.connect(audio.destination)
  await new Promise<void>((resolver) => {
    tono.onended = () => {
      tono.disconnect()
      volumen.disconnect()
      window.setTimeout(resolver, 0)
    }
    tono.start(inicio)
    tono.stop(fin)
  })
}

/** Encola una respuesta por cada llamada del reconocedor al confirmar. */
export function reproducirRecocha(etiqueta: string): boolean {
  const fragmentos = RESPUESTAS_RECOCHA.get(etiqueta)
  if (!fragmentos || !audioRecocha || !('speechSynthesis' in window)) return false
  const anterior = colaRecocha
  if (!anterior) speechSynthesis.cancel()
  const turno = (anterior ?? Promise.resolve()).then(async () => {
    await pronunciarFragmento(fragmentos[0])
    await beep()
    if (fragmentos[1]) await pronunciarFragmento(fragmentos[1])
  }).catch((error: unknown) => {
    console.warn('No se pudo reproducir Modo recocha.', error)
  })
  colaRecocha = turno
  void turno.then(() => {
    if (colaRecocha !== turno) return
    colaRecocha = undefined
    const pendiente = lecturaPendiente
    lecturaPendiente = undefined
    if (pendiente) hablar(pendiente)
  })
  return true
}
