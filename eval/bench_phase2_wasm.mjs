/**
 * Benchmark kompabilitas op & latensi ORT-Web 1.20.1 WASM di Chromium nyata
 * untuk seluruh model Fase 2 AksaraScan v4.0 (Detektor YOLOX/YOLOv8, Stage-2 Embedder, CRNN Track B).
 *
 * Penggunaan:
 *   node eval/bench_phase2_wasm.mjs
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const EVAL_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(EVAL_DIR, '..');
const ORT_DIST = path.join(EVAL_DIR, 'node_modules', 'onnxruntime-web', 'dist');
const { chromium } = await import(pathToFileURL(path.join(EVAL_DIR, 'node_modules', 'playwright-core', 'index.mjs')).href);

function findChrome() {
  const candidates = [
    process.env.CHROME_EXE,
    path.join(process.env.LOCALAPPDATA || '', 'ms-playwright', 'chromium-1234', 'chrome-win64', 'chrome.exe'),
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  ].filter(Boolean);
  return candidates.find((c) => fs.existsSync(c));
}

const SCRATCH_WASM = 'C:/Users/WA/.gemini/antigravity/brain/e8f20c41-7ca5-46b2-92f2-7e5b3331f29f/scratch/wasm_test';

const ROUTE_MAP = {
  '/v31.onnx': path.join(ROOT_DIR, 'app', 'web_camscanner', 'aksara_yolo.onnx'),
  '/stage2_embed.onnx': path.join(ROOT_DIR, 'app', 'web_camscanner', 'aksara_embed.onnx'),
  '/crnn.onnx': path.join(ROOT_DIR, 'runs_crnn', 'aksara_crnn_ctc.onnx'),
  '/yolox_smoke_tiny640_fp16.onnx': path.join(ROOT_DIR, 'runs_yolox', 'v4_smoke', 'yolox-tiny_img640_s42_fp16.onnx'),
  '/yolox_nano_p2_640_fp32.onnx': path.join(SCRATCH_WASM, 'yolox-nano_p2_640_fp32.onnx'),
  '/yolox_tiny_p2_640_int8.onnx': path.join(SCRATCH_WASM, 'yolox-tiny_p2_640_int8.onnx'),
  '/yolox_tiny_p2_960_fp16w.onnx': path.join(SCRATCH_WASM, 'yolox-tiny_p2_960_fp16w.onnx'),
  '/yolox_tiny_p2_960_int8.onnx': path.join(SCRATCH_WASM, 'yolox-tiny_p2_960_int8.onnx'),
};

const MIME_MAP = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
};

const server = http.createServer((req, res) => {
  const u = (req.url || '/').split('?')[0];
  if (u === '/test.html') {
    res.setHeader('Content-Type', MIME_MAP['.html']);
    res.end(`<!doctype html><html><body><script src="/ort/ort.min.js"></script><script>
      window.benchModel = async (url, shape) => {
        ort.env.wasm.wasmPaths = '/ort/';
        ort.env.wasm.numThreads = crossOriginIsolated ? 4 : 1;
        try {
          const t0 = performance.now();
          const sess = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
          const loadMs = Math.round(performance.now() - t0);
          const inName = sess.inputNames[0];
          const n = shape.reduce((a, b) => a * b, 1);
          const inp = new ort.Tensor('float32', new Float32Array(n).fill(0.25), shape);
          await sess.run({ [inName]: inp });
          const times = [];
          let outShapes = {};
          for (let i = 0; i < 7; i++) {
            const ta = performance.now();
            const out = await sess.run({ [inName]: inp });
            times.push(performance.now() - ta);
            if (i === 0) {
              for (const k of sess.outputNames) outShapes[k] = out[k].dims;
            }
          }
          times.sort((a, b) => a - b);
          return {
            ok: true,
            loadMs,
            p50Ms: Math.round(times[Math.floor(times.length * 0.5)]),
            p95Ms: Math.round(times[Math.min(times.length - 1, Math.floor(times.length * 0.95))]),
            inName,
            outShapes,
            crossOriginIsolated,
            wasmThreads: ort.env.wasm.numThreads,
          };
        } catch (e) {
          return { ok: false, error: String(e) };
        }
      };
    </script></body></html>`);
    return;
  }
  const fp = u.startsWith('/ort/') ? path.join(ORT_DIST, u.slice(5)) : ROUTE_MAP[u];
  if (!fp || !fs.existsSync(fp)) {
    res.statusCode = 404;
    res.end('not found');
    return;
  }
  res.setHeader('Content-Type', MIME_MAP[path.extname(fp).toLowerCase()] || 'application/octet-stream');
  fs.createReadStream(fp).pipe(res);
});

await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true, executablePath: findChrome() });
const page = await browser.newPage();
await page.goto(`${base}/test.html`, { waitUntil: 'domcontentloaded' });

const cases = [
  ['v3.1_yolov8n_416_fp32', '/v31.onnx', [1, 3, 416, 416]],
  ['stage2_embed_batch1_64', '/stage2_embed.onnx', [1, 3, 64, 64]],
  ['stage2_embed_batch16_64', '/stage2_embed.onnx', [16, 3, 64, 64]],
  ['crnn_track_b_64x768', '/crnn.onnx', [1, 1, 64, 768]],
  ['yolox_nano_p2_640_fp32', '/yolox_nano_p2_640_fp32.onnx', [1, 3, 640, 640]],
  ['yolox_tiny_p2_640_fp16w', '/yolox_smoke_tiny640_fp16.onnx', [1, 3, 640, 640]],
  ['yolox_tiny_p2_640_int8', '/yolox_tiny_p2_640_int8.onnx', [1, 3, 640, 640]],
  ['yolox_tiny_p2_960_fp16w', '/yolox_tiny_p2_960_fp16w.onnx', [1, 3, 960, 960]],
  ['yolox_tiny_p2_960_int8', '/yolox_tiny_p2_960_int8.onnx', [1, 3, 960, 960]],
];

const report = [];
for (const [label, url, shape] of cases) {
  const fp = ROUTE_MAP[url];
  if (!fp || !fs.existsSync(fp)) {
    console.log(`[SKIP] ${label}: file missing`);
    continue;
  }
  const sizeMB = Number((fs.statSync(fp).size / (1024 * 1024)).toFixed(2));
  const res = await page.evaluate(({ url, shape }) => window.benchModel(url, shape), { url, shape });
  const row = { label, sizeMB, shape, ...res };
  report.push(row);
  console.log(label, JSON.stringify(row));
}

await browser.close();
server.close();

const outDir = path.join(EVAL_DIR, 'reports');
fs.mkdirSync(outDir, { recursive: true });
const outFile = path.join(outDir, 'phase2_wasm_bench.json');
fs.writeFileSync(outFile, JSON.stringify({ generatedAt: new Date().toISOString(), ortVersion: '1.20.1', models: report }, null, 2));
console.log(`[SAVED] ${outFile}`);
