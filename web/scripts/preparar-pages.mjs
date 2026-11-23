import { createHash } from 'node:crypto'
import { copyFile, readFile, writeFile } from 'node:fs/promises'

const origen = new URL('../../ml/candidatas/gru_52/', import.meta.url)
const salida = new URL('../dist-pages/', import.meta.url)
const manifiesto = JSON.parse(await readFile(new URL('exportacion.json', origen), 'utf8'))
const config = JSON.parse(await readFile(new URL('config.json', origen), 'utf8'))
if (manifiesto.clases !== 52 || config.modelo !== 'GRU' || new Set(config.etiquetas).size !== 52) {
  throw new Error('GitHub Pages requiere la GRU validada de 52 categorías.')
}

for (const nombre of ['config.json', 'modelo.onnx']) {
  const archivo = new URL(nombre, origen)
  const huella = createHash('sha256').update(await readFile(archivo)).digest('hex')
  if (huella !== manifiesto.exportacion_sha256[nombre]) throw new Error(`Huella inesperada de ${nombre}`)
  await copyFile(archivo, new URL(`model/${nombre}`, salida))
}
await writeFile(new URL('.nojekyll', salida), '')
