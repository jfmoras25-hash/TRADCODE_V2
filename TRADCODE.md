# TRADCODE

Manual técnico de TRADCODE. La edición de entrega usa la GRU de 52 categorías y el Random Forest híbrido en 8011. MediaPipe detecta manos y postura, el preprocesamiento genera 147 características por cuadro y ONNX Runtime Web clasifica seis cuadros. El navegador procesa el video. FastAPI sirve las páginas y guarda los puntos solo cuando se pulsa **Guardar en TRADCODE**.

## Inicio y variante de referencia en Windows

Sigue la [instalación del README](README.md#instalación-y-ejecución-en-windows) para usar 8011. Para probar la edición anterior de 47 categorías en 8010, usa los siguientes comandos desde la raíz con el entorno `.venv` ya creado.

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
Set-Location .\web
npm.cmd ci
npm.cmd run build
Set-Location ..
.\.venv\Scripts\python.exe -B -m backend.main
```

Abre [http://127.0.0.1:8010/](http://127.0.0.1:8010/). Pulsa **Entrar al traductor**, **Activar cámara** e **Iniciar reconocimiento**. El servidor usa solo la interfaz local de Windows. Si cambias el frontend, vuelve a compilarlo. El modelo y los detectores disponibles en `web/public/` se copian en la compilación. [README de estructura y entrenamiento](README.md)

## Inventario y transformación

| Fuente | Inventario | Resultado |
|---|---|---|
| `Fronted` anterior | Seis HTML, seis CSS, traductor, voz, abecedario, lección, cliente y trabajador Holistic, módulos de mano/pose 3D y respaldo de funciones. | Se recuperaron Inicio, Traductor, Aprendizaje, Abecedario, Lección y Sobre el proyecto, incluidos estilos, navegación y textos. React gestiona las dos pantallas interactivas y TypeScript conserva el progreso del abecedario. |
| Recursos anteriores | Cinco SVG, cinco JPG, un MP3, un GLB, cinco poses JSON y el detector Holistic. | Las ilustraciones, fotografías, sonido y visor de mano siguen disponibles. Holistic se sustituyó por los detectores de manos y postura del proyecto nuevo. |
| Backend anterior | FastAPI/Uvicorn, ruta de inicio, WebSocket `/ws`, clasificación estática y dinámica, captura CSV, validación y conteo. Incluye detector Holistic Python, utilidades de normalización y recuento, revisión de coordenadas, prueba OpenCV, scripts de entrenamiento, dos juegos de requisitos, dos pickle, CSV y respaldos. | Se conserva el servicio local, el inicio y el guardado validado de muestras. `POST /api/capturas` guarda JSON de seis cuadros compatibles con la herramienta de importación y cuenta muestras guardadas o repetidas. El protocolo, preprocesamiento estático/dinámico y clasificación antiguos fueron sustituidos por el flujo GRU en el navegador. |
| Investigación anterior | `EntrenamientoDinamico`, `LSC50`, el NPZ dinámico de la raíz y los CSV de muestras. | Siguen en sus rutas originales como investigación e historial. Sus dimensiones y esquema corresponden al motor anterior. |
| Base nueva | React, TypeScript y Vite, MediaPipe Tasks, normalización, ONNX Runtime Web, práctica libre, configuración, grabación de secuencias y pipeline Python de LSC70. | Se reutiliza el motor y se incorpora la práctica libre desde Aprendizaje. La grabación admite descarga JSON y guardado local. El panel técnico de la demostración se integra en Sobre el proyecto y en la documentación. |
| Artefactos de la base | LSC70, ZIP, landmarks, caché, métricas, GRU y configuración ONNX. | Se conserva el entrenamiento existente. El servidor publica exclusivamente `web/dist` y las rutas de aplicación. |

## Lecciones, voz y capturas

- El abecedario mantiene las 27 letras y la clave `progresoTRADCODE` del navegador. Una práctica completada actualiza las estadísticas y reproduce el sonido y, si está activada, la voz. La GRU reconoce la etiqueta interna `NN` como Ñ.
- El visor muestra A–E y 16 referencias de ASL de Eidos. J y Z tienen reproducción de movimiento. N, Ñ, Q, S, T y X conservan la práctica por cámara y muestran un aviso de pose pendiente. Saludos, Expresiones y Conversaciones conservan sus tarjetas sin ejercicios navegables.
- El traductor conserva activación manual de cámara, detección de ambas manos y cuerpo, confianza, alternativas y voz automática. El sonido de la solución y la voz se ejecutan en el navegador. La malla facial de Holistic quedó fuera porque el modelo utiliza manos y postura.
- Las capturas se guardan bajo `ml/datos/capturas_hibrido/` en 8011 y `ml/datos/capturas/` en 8010. El servidor valida los 21 puntos por mano y 33 de postura, filtra repeticiones exactas de la misma persona y seña, y almacena un JSON por muestra. Se conserva también **Descargar JSON**. Las capturas requieren importación y entrenamiento explícitos para cambiar el modelo.
- El umbral de distancia del deduplicador del CSV dependía del vector de 63 coordenadas de una sola mano. La captura nueva compara la identidad de la secuencia completa. La segmentación de 60 cuadros del modelo anterior se sustituye por la ventana y la confirmación temporal de la GRU.
- Los créditos distinguen el material del proyecto y las tecnologías y recursos de terceros. [Créditos y licencias](web/public/creditos.txt)

## Distribución del traductor

En escritorio, la cámara y el panel de resultados comparten una cuadrícula de hasta 1680 px. A 1180 px o menos, las dos tarjetas se apilan para conservar un visor cómodo. Las barras de confianza se expanden dentro de la tarjeta de resultados.

## Registro de limpieza

- `web/src/App.css` y `web/src/index.css` pertenecían a la demostración anterior. Las seis entradas HTML y sus módulos usan los estilos de TRADCODE y `src/tradcode.css`.
- `web/src/components/Camara.tsx` y `PanelAcerca.tsx` solo declaraban componentes sin consumidores. El visor activo es `Visor.tsx` y los créditos se publican en `sobre.html`. La función `PrediccionEnVivo` tampoco tenía consumidores. La lista activa `TopTres` se conserva.
- LSC50, EntrenamientoDinamico y los reconocedores antiguos se encuentran fuera de la base nueva. El entrenamiento LSC70 y las utilidades sintéticas se conservan por sus referencias en las pruebas y el manual. Los entornos locales, cachés de instalación y registros se conservan para evitar afectar procesos y reinstalaciones.
- Los bloques de `web/estilo.css` que solo estilizan la tarjeta `.camara` de la vista anterior quedan sustituidos por las dos tarjetas del traductor. El ZIP de instalación de Node es una copia de distribución que ningún script consume después de extraer el ejecutable. Los dos registros vacíos `vite-dev*.log` proceden de intentos de arranque anteriores y se retiran si se pueden abrir en exclusivo.

## Comprobaciones

### Probar el reconocimiento híbrido

Desde `web`, ejecuta `npm.cmd run build:hibrido`. Después inicia el servidor desde la raíz de `TRADCODE_LSC_BASE`.

```powershell
.\.venv\Scripts\python.exe -B -m backend.prueba_hibrida
```

Abre [TRADCODE de prueba](http://127.0.0.1:8011/). Esta instancia sirve `web/dist-hibrido/` y carga la GRU de 52 clases desde `ml/candidatas/gru_52/`. Si ya hay una instancia en el 8011, detenla con Ctrl+C en su propia terminal antes de iniciar la nueva. El servidor de referencia usa el [puerto 8010](http://127.0.0.1:8010/) con su modelo de 47 clases.

- En **Deletreo**, Random Forest puede confirmar A, B, C, D, E, F, I, K, L y M. Se usa el bosque si su primera opción es una de esas letras, alcanza el umbral y la primera opción de la GRU no es una letra fuera de su cobertura. La GRU conserva las 27 letras y funciona como alternativa si el bosque queda bajo el umbral o no responde. Desmarca **Preferir Random Forest** para comparar con la GRU sola.
- En **Palabras y señas**, la GRU confirma palabras, números y las cinco señas añadidas mediante capturas, incluidas **DEDO MEDIO** e **HIJUEPUTA**. Si gana una letra, el modo espera otra seña. `BUENAS` y `TARDES` forman **BUENAS TARDES** al confirmarse en orden.
- Cambiar de modo reinicia la confirmación y conserva el texto y el búfer. La confianza y las alternativas se muestran para el motor que aporta la predicción. Las lecciones y la práctica libre siguen con el flujo general de la GRU.
- Las muestras de prueba se guardan en `ml/datos/capturas_hibrido/`, en un JSON por ejemplo y con deduplicación. También se puede descargar un JSON sin guardarlo en el servidor. El [procedimiento para preparar nuevas letras](ml/legado_rf/README.md#preparar-nuevas-letras) explica cómo evaluar una futura versión del bosque.

Ejecuta `npm.cmd test` desde `web` para verificar la paridad de características y las reglas de selección entre motores. El [estado de la prueba](http://127.0.0.1:8011/api/estado) enumera las 52 etiquetas GRU y las diez letras RF.

### Entrenar la GRU de 52 clases manualmente

El repositorio de entrega permite ejecutar la aplicación con sus pesos. Para reproducir el entrenamiento de 52 clases también se necesitan el ONNX y el informe de GRU 50, los landmarks de LSC70 y las capturas originales, que son archivos locales. El informe completo de GRU 52 también queda local. Para preparar ese entorno y verificar la organización de LSC70 en PowerShell desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt -r .\ml\requirements.txt
.\.venv\Scripts\python.exe -B .\ml\scripts\01_inspeccionar_dataset.py --raiz .\ml\datos\LSC70 --secciones LSC70AN LSC70W
```

El [dataset LSC70 v2](https://data.mendeley.com/datasets/9ssyn8tff5/2) se descomprime en `ml/datos/LSC70/`, con `LSC70AN` y `LSC70W` directamente dentro. Los scripts `02_extraer_landmarks.py` y `04_entrenar_candidata_gru50.py` son etapas de investigación local. Con los insumos de GRU 52 disponibles:

```powershell
.\.venv\Scripts\python.exe -B .\ml\scripts\05_entrenar_gru52.py --auditar
.\.venv\Scripts\python.exe -B -u .\ml\scripts\05_entrenar_gru52.py --salida gru_52_r2
```

El nombre predeterminado de salida es `gru_52`. Usa un nombre nuevo por ejecución. El script se inicia únicamente con un comando de entrenamiento. Valida los JSON, descarta secuencias duplicadas o con menos de tres cuadros con manos, recupera los pesos ONNX de GRU 50 y verifica su equivalencia numérica antes de añadir las dos salidas nuevas. Conserva el reparto por participante de LSC70 y las particiones de las tres clases anteriores. Si una clase nueva tiene una sola persona, sus últimas muestras quedan en validación. Es una evaluación de ejemplos reservados de la misma persona.

Revisa `resumen.json`, `metricas_por_clase.csv`, `matriz_confusion.csv`, `errores_validacion.csv` e `historial.json` dentro de la versión nueva. La carpeta también conserva los pesos PyTorch y la exportación ONNX. Al iniciar el servidor con una versión explícita, se comprueban sus 52 etiquetas y las huellas de la exportación:

```powershell
.\.venv\Scripts\python.exe -B -m backend.prueba_hibrida --version gru_52_r2
```
