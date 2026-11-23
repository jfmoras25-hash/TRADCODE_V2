# Interfaz de TRADCODE

React y TypeScript implementan el traductor y las lecciones. Vite compila las seis páginas HTML. MediaPipe Tasks detecta manos y postura. ONNX Runtime Web ejecuta en el navegador la GRU que sirve el backend.

Desde esta carpeta, con Node.js 24:

```powershell
npm.cmd ci
npm.cmd test
npm.cmd run build:hibrido
```

La entrega híbrida queda en `dist-hibrido/`. FastAPI sirve la aplicación en [el puerto 8011](http://127.0.0.1:8011/) y entrega el modelo de 52 clases por las rutas `/model/`. `npm.cmd run build` genera la variante de referencia de 8010 en `dist/`. Durante el desarrollo, `npm.cmd run dev` reenvía las rutas `/api` a 8010. Consulta el [README](../README.md) para iniciar la versión de entrega.
