/**
 * URL absoluta de un archivo de public/. Con `base: './'` las rutas relativas
 * se resolverían respecto al módulo JS (assets/) en los import() dinámicos,
 * así que se resuelven contra la página.
 */
export const rutaPublica = (p: string) => new URL(import.meta.env.BASE_URL + p, document.baseURI).href
