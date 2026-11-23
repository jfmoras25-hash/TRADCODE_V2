// Copia los archivos WebAssembly de MediaPipe y ONNX Runtime a public/ y,
// si hay internet, descarga los modelos de MediaPipe para no depender de
// Google durante la sustentación. Se ejecuta solo con npm install/dev/build.
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const raiz = join(dirname(fileURLToPath(import.meta.url)), '..')
const pub = join(raiz, 'public')
const nm = join(raiz, 'node_modules')

function copiarDir(origen, destino, filtro = () => true) {
  mkdirSync(destino, { recursive: true })
  for (const f of readdirSync(origen)) if (filtro(f)) copyFileSync(join(origen, f), join(destino, f))
}

copiarDir(join(nm, '@mediapipe/tasks-vision/wasm'), join(pub, 'mediapipe-wasm'))
copiarDir(join(nm, 'onnxruntime-web/dist'), join(pub, 'ort'), (f) => /^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f))
console.log('✔ WebAssembly de MediaPipe y ONNX Runtime copiados a public/')

const modelos = {
  'hand_landmarker.task':
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  'pose_landmarker_lite.task':
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}
mkdirSync(join(pub, 'models'), { recursive: true })
for (const [nombre, url] of Object.entries(modelos)) {
  const destino = join(pub, 'models', nombre)
  if (existsSync(destino)) continue
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(30000) })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    writeFileSync(destino, Buffer.from(await r.arrayBuffer()))
    console.log(`✔ ${nombre} descargado`)
  } catch (e) {
    console.warn(`⚠ No se pudo descargar ${nombre} (${e.message}); la app lo cargará desde internet.`)
  }
}
