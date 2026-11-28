// Service Worker for AksaraScan PWA Offline Support
const CACHE_NAME = 'aksarascan-v14';
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './styles.css',
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
  // Cache-first strategy for local assets
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((networkResponse) => {
        // Cache font responses dynamically
        if (
          event.request.url.includes('fonts.googleapis.com') ||
          event.request.url.includes('fonts.gstatic.com')
        ) {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      });
    }).catch(() => {
      // Offline fallback
      return caches.match('./index.html');
    })
  );
});
