# TRADCODE · Lengua de Señas Colombiana

TRADCODE es un proyecto académico que reconoce un conjunto de señas de la Lengua de Señas Colombiana (LSC) y muestra el resultado como texto y voz. La versión de entrega se ejecuta en el puerto **8011** con una GRU de **52 clases** y un Random Forest complementario para el deletreo.

## Instalación y ejecución en Windows

Se verificó con **Python 3.12 de 64 bits**, **Node.js 24 con npm** y **Microsoft Edge**. Chrome también admite las API web que usa la aplicación. Usa PowerShell desde la raíz del repositorio. La primera instalación necesita conexión a PyPI, al registro de npm y al servidor público de modelos de Google.

```powershell
py -3.12 --version
node --version
npm.cmd --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
.\.venv\Scripts\python.exe -m pip check
Set-Location .\web
npm.cmd ci
npm.cmd test
npm.cmd run build:hibrido
Set-Location ..
.\.venv\Scripts\python.exe -B -m backend.prueba_hibrida
```

Si tu instalación de Python no trae el comando `py`, usa `python --version` para comprobar que `python` apunta a la versión 3.12 y luego ejecuta `python -m venv .venv`. `npm ci` instala el archivo de bloqueo y prepara los detectores de MediaPipe y los archivos WebAssembly. La compilación genera `web/dist-hibrido/`.

Abre [TRADCODE](http://127.0.0.1:8011/) en el mismo equipo. Entra al traductor, elige el modo, pulsa **Activar cámara** e **Iniciar reconocimiento**. Permite el acceso a la cámara si el navegador lo solicita. [Estado del servidor y lista de etiquetas](http://127.0.0.1:8011/api/estado)

La aplicación se ejecuta con los pesos incluidos. Puedes usarla sin descargar LSC70, PyTorch ni los datos de entrenamiento. `backend/requirements.txt` contiene las dependencias de ejecución, incluida la versión exacta de scikit-learn que exige el Random Forest. `ml/requirements.txt` contiene las dependencias adicionales para investigación y entrenamiento.

## Qué puedes hacer

- **Palabras y señas** muestra las predicciones de la GRU, su confianza y las alternativas disponibles. Confirma números y señas como `HOLA`, `ADIOS`, `MI` y `NOMBRE_ES`. `BUENAS` y `TARDES` se confirman por separado.
- **Deletreo** usa la GRU para las 27 letras y da preferencia al Random Forest para A, B, C, D, E, F, I, K, L y M cuando la predicción es elegible. Las letras confirmadas se acumulan en la palabra en formación. **ESPACIO** la agrega al texto. Puedes borrar, limpiar, escuchar el texto o habilitar voz automática.
- **Modo recocha** activa respuestas de voz y un tono para `DEDO_MEDIO` e `HIJUEPUTA` después de confirmarlas en **Palabras y señas**. El interruptor empieza apagado.
- **Capturar seña** graba secuencias de seis cuadros de landmarks. Permite descargar un JSON o guardarlo en `ml/datos/capturas_hibrido/` con validación y deduplicación. Las muestras permanecen locales. Guardar una captura no inicia entrenamiento.
- **Lecciones** ofrece navegación entre las 27 letras, ejercicios con cámara y progreso guardado en el navegador. Hay 21 poses 3D girables y ampliables. A–E son las poses preexistentes del proyecto y 16 son referencias de ASL identificadas como tales. J y Z tienen reproducción de movimiento. N, Ñ, Q, S, T y X conservan el ejercicio y muestran el aviso de pose pendiente. [Procedencia de las poses de Eidos](web/public/modelos/poses/eidos-asl/README.md)

## Arquitectura y archivos

| Componente | Archivos y función |
|---|---|
| Frontend | `web/` contiene seis páginas, estilos, React, TypeScript, Vite, visor Three.js y pruebas con Vitest. |
| Reconocimiento GRU | MediaPipe Tasks detecta manos y postura en el navegador. El preprocesamiento usa seis cuadros con 147 características por cuadro. ONNX Runtime Web ejecuta la GRU en WebAssembly y devuelve 52 puntuaciones. |
| Backend | `backend/prueba_hibrida.py` sirve la compilación, el estado y los archivos del modelo. `POST /api/letras` evalúa el bosque en Deletreo y `POST /api/capturas` guarda las muestras a petición del usuario. El vídeo se procesa en el navegador. |
| Artefactos de ejecución | `ml/candidatas/gru_52/` contiene ONNX, configuración y el manifiesto SHA-256. `ml/candidatas/gru_50/config.json` permite validar el orden heredado de etiquetas. `ml/legado_rf/modelo_senas.pkl` es el Random Forest histórico. |
| Referencia anterior | `web/public/model/` conserva la GRU de 47 clases para pruebas y la variante local de 8010. El endpoint de 8011 entrega los archivos de la GRU 52 bajo `/model/`. |
| Investigación | `ml/lsc/`, `ml/scripts/` y `ml/tests/` contienen preprocesamiento, extracción, entrenamiento manual y utilidades de prueba. Las capturas y el dataset quedan fuera de Git. |

## Alcance y limitaciones

El reconocimiento está limitado a las **52 etiquetas entrenadas**, que incluyen 27 letras, 8 números y 17 palabras o señas adicionales. `NN` representa Ñ. El traductor construye texto a partir de predicciones confirmadas. No interpreta gramática, contexto ni conversación continua de LSC.

La GRU de 52 clases obtuvo **84,63 % de exactitud** y **84,25 % de F1 macro** sobre 527 secuencias de validación. LSC70 se dividió por participante y las cinco clases incorporadas provienen de capturas de una sola persona. Esos resultados no establecen generalización para esas cinco clases. La iluminación, el encuadre, la forma de realizar la seña y el umbral afectan la confirmación. La voz depende del navegador y del sistema operativo. Las referencias ASL son material visual y los ejercicios siguen evaluando LSC. [Métricas por clase](ml/candidatas/gru_52/metricas_por_clase.csv) · [Matriz de confusión](ml/candidatas/gru_52/matriz_confusion.csv)

## Procedencia, licencias y entrega

La GRU parte de LSC70 v2, de la Universidad del Cauca, publicado bajo **CC BY 4.0**, y de capturas locales. El Random Forest proviene de una versión anterior de TRADCODE. Los pesos son artefactos del proyecto y los JSON privados de entrenamiento se mantienen fuera del repositorio. [LSC70 v2 y licencia](https://data.mendeley.com/datasets/9ssyn8tff5/2) · [Créditos y licencias](web/public/creditos.txt)

Consulta el [manual técnico](TRADCODE.md) para entrenamiento y el [informe y procedimiento de publicación](PUBLICACION.md) para preparar GitHub y Classroom. Los recursos heredados sin licencia individual documentada requieren confirmar su autorización de redistribución antes de dar acceso al repositorio.
