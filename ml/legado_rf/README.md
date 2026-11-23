# Random Forest legado de TRADCODE

Evaluador independiente. No inicia un servidor ni escribe datos. `modelo_senas.pkl` es la copia exacta de `Backend/modelo_senas.pkl` del TRADCODE anterior. El evaluador solo carga esa copia si coincide su SHA-256 y se usa scikit-learn 1.9.1, la versión registrada en el modelo.

## Entrada y normalización

1. Tomar los **landmarks normalizados de imagen** (`landmarks` del Hand Landmarker), en el orden de MediaPipe. El índice 0 es la muñeca. Los índices 1 a 4 son del pulgar, 5 a 8 del índice, 9 a 12 del medio, 13 a 16 del anular y 17 a 20 del meñique. El índice 9 es la base del dedo medio. [Enumeración oficial de MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py)
2. Conservar los triples crudos `x, y, z` y aplanarlos en ese orden. El detector del traductor ya los guarda como `cuadro.manos`, con ranuras **derecha e izquierda**. El clonado de la normalización antigua usa esos puntos antes de la corrección por aspecto y del recorte de características de la GRU. El espejo visual de la cámara es CSS y no cambia esas coordenadas.
3. Usar la mano derecha tal como se detecta. Si solo hay izquierda, reflejar cada `x` con `x' = 2 × muñeca.x − x`, conservando `y` y `z`. Si aparecen ambas manos, se prioriza la derecha, como en el traductor antiguo. Esto sigue la lateralidad que informa el detector. Si este confunde las manos, el preprocesamiento seguirá esa asignación.
4. Restar el vector de la muñeca a cada uno de los 21 puntos. Dividir por la norma euclidiana 3D del punto 9 ya centrado cuando es mayor que cero. Con escala cero se conservan los puntos centrados, tal como en `Backend/utilidades.py`.

El bosque tiene 100 árboles y devuelve la etiqueta con mayor probabilidad. La marca `aceptada_con_umbral_antiguo` reproduce el umbral original de 0,70. Las clases guardadas son **A, B, C, D, E, F, I, K, L, M, BIEN y MAL**.

## Verificación con datos existentes

- Las 1.153 filas del CSV antiguo dieron exactamente las mismas coordenadas normalizadas que la función original. De las 1.126 filas con una etiqueta presente en el modelo, el bosque acertó 1.115. El CSV puede incluir datos de entrenamiento del bosque, por lo que esta comprobación no mide generalización.
- En las 109 secuencias reservadas de LSC70 de las diez letras cubiertas tomé el cuadro 4 de cada secuencia. En 108 había mano y el adaptador seleccionó la derecha. La mejor etiqueta coincidió en 74 casos. El umbral original aceptó 49 predicciones, 46 correctas y 3 incorrectas. Se omitió un cuadro sin mano.
- Las dos entradas de consola se comprobaron con capturas reales exportadas por el frontend, incluyendo un cuadro de mano derecha y uno de izquierda. Las señas anotadas en esas capturas eran MI y ADIOS, así que sirven para verificar la entrada. En 61 cuadros reales de mano izquierda comprobé además la paridad exacta con el reflejo y la normalización antiguos. El rendimiento con la cámara física y las diez letras queda pendiente.

## Prueba con tu cámara

1. En TRADCODE, abre **Traductor → Activar cámara → Capturar seña**. Pon una etiqueta que conozcas, como `A`, y **1** en **Ejemplos seguidos**.
2. Pulsa **Grabar**, mantén la seña durante la toma y usa **Descargar JSON**. Puedes repetirlo usando solo la otra mano. El evaluador lee ese archivo y no interviene en el reconocimiento activo.
3. Desde PowerShell en la raíz de `TRADCODE_LSC_BASE`, reemplaza la ruta del ejemplo por la del JSON descargado. Los números de muestra y cuadro empiezan en 1.

```powershell
.\ml\.venv\Scripts\python.exe -B .\ml\legado_rf\evaluar.py --captura "C:\ruta\a\capturas.json" --muestra 1 --cuadro 4
```

Si no se detectó ninguna mano en ese cuadro, prueba otro de los seis. Para consultar las clases o evaluar un JSON que contenga directamente los 63 números, usa:

```powershell
.\ml\.venv\Scripts\python.exe -B .\ml\legado_rf\evaluar.py --clases
.\ml\.venv\Scripts\python.exe -B .\ml\legado_rf\evaluar.py --coordenadas "C:\ruta\a\puntos.json" --mano izquierda
```

Al usar `--coordenadas`, indica `izquierda` solo si los puntos son crudos. Para puntos de mano derecha o ya reflejados usa `derecha`. Para una integración posterior, `ReconocedorLegado.evaluar` devuelve etiqueta y confianza y `mano_de_cuadro` adapta el formato de cámara. Se pueden conectar a un modo aislado de deletreo con un filtro de las 10 letras y una evaluación de cámara antes de activarlo.

## Preparar nuevas letras

1. En la [instancia híbrida](http://127.0.0.1:8011/), activa la cámara y usa **Capturar seña**. Usa un identificador estable por participante y una etiqueta por letra. El campo acepta Ñ y la guarda como `NN`, la etiqueta que también entiende la GRU. **Guardar en TRADCODE** añade JSON a `ml/datos/capturas_hibrido/`. Las capturas previas y el CSV legado se conservan. También puedes archivar el JSON descargado.
2. Para la futura importación, lee los JSON con el esquema `lsc-capturas`, valida seis cuadros por muestra y conserva solo etiquetas del alfabeto. Revisa la seña grabada y excluye duplicados, manos ausentes y cuadros inestables. Para el RF selecciona un cuadro estable con una mano y reutiliza `mano_de_cuadro` y `normalizar` de este evaluador. Conserva el participante y el archivo de origen en la tabla derivada. El importador GRU genera secuencias y esa salida requiere adaptación para el bosque.
3. Cuando se autorice el entrenamiento, combina las letras nuevas con las letras válidas del CSV histórico. Mantén las filas antiguas sin identificador de persona fuera de una evaluación que pretenda medir personas nuevas. Separa las nuevas capturas por participante antes de elegir las filas de entrenamiento y validación. Mantén juntas las filas de una misma toma. Entrena un bosque candidato de 63 características con el preprocesamiento anterior.
4. Guarda cada candidato y sus informes en una carpeta versionada dentro de `ml/candidatas/`. Mantén `ml/legado_rf/modelo_senas.pkl` como respaldo. Antes de publicarlo, comprueba la lista de clases, las métricas por letra, la abstención y las confusiones con ambas manos, y compara el candidato con el bosque actual y la GRU sobre las mismas personas reservadas.
5. Amplía la lista de letras preferentes solo después de aprobar un candidato. La GRU sigue cubriendo las letras que el nuevo bosque aún no cubra.

Este procedimiento queda documentado. La instancia híbrida carga los pesos existentes.
