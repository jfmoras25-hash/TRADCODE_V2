# Preparar la entrega de TRADCODE

El primer commit debe salir de `TRADCODE_LSC_BASE` con un historial nuevo. La versión de entrega es la que usa GRU 52 y Random Forest en 8011. El [README](README.md) contiene los pasos de instalación y prueba para el docente.

## Inventario de publicación

| Se incluye | Motivo |
|---|---|
| Fuentes de `web/`, `package-lock.json`, frontend estático, audio, poses, malla GLB y créditos | Interfaz, lecciones y compilación reproducible. Las condiciones de redistribución pendientes se indican abajo. |
| `backend/` y `backend/requirements.txt` | Servidor local y dependencias de ejecución. |
| `ml/candidatas/gru_52/` con `modelo.onnx`, `config.json` y `exportacion.json` | Pesos de entrega, etiquetas y comprobación de integridad sin el informe privado de entrenamiento. También se incluyen historial, métricas por clase y matriz agregada. |
| `ml/candidatas/gru_50/config.json` | El backend coteja el orden de las 50 etiquetas heredadas. El ONNX de esa versión y su informe completo son experimentos locales. |
| `ml/legado_rf/modelo_senas.pkl` y su evaluador | Bosque complementario. El evaluador verifica su SHA-256 y requiere scikit-learn 1.9.1. |
| `web/public/model/` con GRU 47 y configuración | Variante anterior de 8010 y catálogo usado por las pruebas. En 8011, la ruta del backend sirve explícitamente la GRU 52. |
| `ml/lsc/`, `ml/scripts/`, `ml/tests/` y `ml/requirements.txt` | Metodología y herramientas de investigación. La prueba web de paridad lee un JSON de coordenadas sintéticas versionado. |

### Exclusiones

- `ml/datos/` completo. Incluye capturas privadas, LSC70, imágenes, landmarks y caché. `ml/datos_prueba/` se regenera.
- `ml/resultados/`, informes completos `resumen.json` de las candidatas y variantes experimentales. Los informes contienen el reparto de participantes y nombres de archivos de captura. El manifiesto de 8011 solo contiene el conteo de clases y huellas de los dos archivos publicados.
- `ml/modelos_mediapipe/` y `web/public/models/`. En la máquina de desarrollo hay copias de los mismos detectores. El frontend los descarga durante la instalación. La extracción de Python usa su propio directorio cuando se ejecuta.
- `web/public/mediapipe-wasm/`, `web/public/ort/`, `web/dist/`, `web/dist-hibrido/` y `web/node_modules/`. Son runtimes copiados desde npm, compilaciones y dependencias que se regeneran.
- Imágenes de práctica opcionales en `web/public/referencias/`, copias de capturas, `.venv/`, `.tools/`, registros y archivos de credenciales. `web/public/referencias/LEEME.md` conserva las instrucciones para añadir referencias locales.

La auditoría conservó `web/public/favicon.svg`, que no tiene referencias explícitas en las seis páginas, y `backend/prueba_gru50.py`, que documenta una variante histórica. Para ejecutar esa variante se necesitan sus artefactos locales. El archivo `web/src/lib/paridad.casos.json` es un recurso sintético que sí usan las pruebas.

Con estas exclusiones no se necesita **Git LFS** para la entrega. `LSC70.zip` ocupa aproximadamente **944 MiB** y debe descargarse desde la fuente si se va a reproducir el entrenamiento. La caché de landmarks ronda **97 MiB** y es regenerable. GitHub advierte por encima de 50 MiB y bloquea archivos de más de 100 MiB en Git normal. [Límites oficiales de GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

### Créditos y revisión previa

- Los detectores de Google y dependencias de npm se obtienen de sus fuentes. La malla de mano WebXR, Three.js, Orbitron y Eidos tienen créditos y avisos de licencia en la carpeta pública. [Créditos](web/public/creditos.txt)
- LSC70 se cita por su versión y DOI. Se distribuye el resultado entrenado y el informe agregado. El ZIP, las capturas privadas e identificadores se quedan en el equipo. [LSC70 v2](https://data.mendeley.com/datasets/9ssyn8tff5/2)
- Los SVG y JPG heredados, el sonido `mision-completada.mp3` y las poses A–E conservan su procedencia local. El proyecto no tenía licencias individuales para esos recursos. Antes de compartirlos, confirma su autoría o autorización de redistribución, así como el permiso aplicable a los pesos entrenados con capturas locales. El repositorio no declara una licencia general para el código propio.
- Si el docente admite repositorios privados, usa esa visibilidad y concede acceso al docente antes de entregar. Si se exige visibilidad pública, resuelve primero los permisos de los recursos heredados.

## Crear el repositorio y entregar

1. Crea un [repositorio vacío en GitHub](https://github.com/new), con el nombre que indicó el docente. Deja desmarcadas las opciones de README, licencia y `.gitignore`, porque se inicializa con los archivos locales. Si la consigna viene de **GitHub Classroom**, acepta primero la tarea y usa la URL del repositorio asignado.
2. Desde PowerShell, entra exactamente en la raíz de `TRADCODE_LSC_BASE` y revisa los archivos que Git va a registrar.

```powershell
Set-Location -LiteralPath 'C:\ruta\a\TRADCODE_LSC_BASE'
if (Test-Path -LiteralPath .git) { throw 'Esta carpeta ya tiene un repositorio. Revisa su historial antes de continuar.' }
git init -b main
git rev-parse --show-toplevel
git status --short --untracked-files=all
git add --all
git diff --cached --name-only
git diff --cached --stat
git var GIT_AUTHOR_IDENT
```

3. La salida de `git rev-parse` debe terminar en `TRADCODE_LSC_BASE`. Revisa la lista preparada y confirma que solo contiene los archivos del inventario. Si hay algo privado, retíralo del índice con `git rm --cached -- RUTA` y vuelve a revisar las exclusiones. En el nombre y correo del autor, usa los datos que quieras mostrar en el historial. Para fijar el correo `noreply` solo para este repositorio, sustituye el marcador en `git config --local user.email "TU_CORREO_NOREPLY_DE_GITHUB"`. Reemplaza `TU_USUARIO` y `TU_REPOSITORIO` con la URL que mostró GitHub.

```powershell
git commit -m "Entrega inicial de TRADCODE"
git remote add origin 'https://github.com/TU_USUARIO/TU_REPOSITORIO.git'
git remote -v
git push -u origin main
git rev-parse HEAD
```

4. En GitHub, abre la vista del repositorio y verifica que aparezca el README y el commit. En **Google Classroom**, adjunta el enlace al repositorio o al commit según la consigna. Si GitHub Classroom creó un repositorio con historial, clónalo y copia allí solo los archivos del inventario antes de hacer el commit. Conserva el historial que generó la tarea.
