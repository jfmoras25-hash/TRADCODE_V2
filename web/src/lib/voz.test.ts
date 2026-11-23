import { afterEach, expect, it, vi } from 'vitest'

afterEach(() => vi.unstubAllGlobals())

it('espera a que termine cada fragmento y el BEEP, incluso con confirmaciones consecutivas', async () => {
  vi.resetModules()
  const eventos: string[] = []
  const finales: (() => void)[] = []
  vi.stubGlobal('window', globalThis)
  vi.stubGlobal('SpeechSynthesisUtterance', class {
    onend?: () => void
    constructor(public text: string) {}
  })
  vi.stubGlobal('speechSynthesis', {
    getVoices: () => [],
    cancel: () => eventos.push('cancelar'),
    speak: (frase: { text: string; onend?: () => void }) => {
      eventos.push(frase.text)
      if (frase.onend) finales.push(frase.onend)
    },
  })
  vi.stubGlobal('AudioContext', class {
    state = 'running'
    currentTime = 0
    resume = async () => {}
    createGain = () => ({
      gain: { setValueAtTime() {}, linearRampToValueAtTime() {} },
      connect() {},
      disconnect() {},
    })
    createOscillator() {
      const tono = {
        frequency: { value: 0 },
        onended: () => {},
        connect() {},
        disconnect() {},
        start: () => eventos.push('BEEP'),
        stop: () => finales.push(tono.onended),
      }
      return tono
    }
  })

  const { activarAudioRecocha, hablar, reproducirRecocha } = await import('./voz')
  await activarAudioRecocha()
  expect(reproducirRecocha('HOLA')).toBe(false)
  expect(reproducirRecocha('DEDO_MEDIO')).toBe(true)
  expect(reproducirRecocha('HIJUEPUTA')).toBe(true)
  await vi.waitFor(() => expect(eventos).toEqual(['cancelar', 'Vaya y métaselo por el...']))
  hablar('HOLA')
  const secuencia = ['BEEP', 'Más...', 'BEEP', 'será su madre']
  for (const paso of secuencia) {
    const longitud = eventos.length
    finales.shift()!()
    await vi.waitFor(() => expect(eventos.slice(longitud)).toEqual([paso]))
  }
  finales.shift()!()
  await vi.waitFor(() => expect(eventos.slice(-2)).toEqual(['cancelar', 'hola']))
})
