// Service Worker for AksaraScan v4.0 PWA Offline Support
const CACHE_NAME = 'aksarascan-v17';
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './styles.css',
  './vendor/fonts/Newsreader-Regular.ttf',
  './vendor/fonts/PlusJakartaSans-Regular.ttf',
  './vendor/fonts/NotoSansJavanese.ttf',
  './vendor/fonts/NotoSansJavanese-Bold.ttf',
  './vendor/ort/ort.min.js',
  './vendor/ort/ort-wasm-simd-threaded.mjs',
  './vendor/ort/ort-wasm-simd-threaded.wasm',
  './image_enhancer.js',
  './enhancer_worker.js',
  './stage2_verifier.js',
  './aksara_embed.onnx',
  './reference_prototypes.json',
  './stage2_calibration.json',
  './javanese_lm_data.js',
  './javanese_lexicon.js',
  './javanese_lm_engine.js',
  './orthography.js',
  './app.js',
  './v3_ui_enhancer.js',
  './v4_features.js',
  './favicon.svg',
  './favicon.png',
  './favicon.ico',
  './apple-touch-icon.png',
  './icon-192.png',
  './icon-512.png',
  './model.json',
  './model.js',
  './yolo_meta.json',
  './aksara_yolo.onnx',
  './manifest.webmanifest',
  './sample_1.png',
  './sample_2.png',
  './sample_3.png',
  './sample_4.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request);
    }).catch(() => {
      return caches.match('./index.html');
    })
  );
});
