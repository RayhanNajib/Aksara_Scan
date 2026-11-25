/**
 * Harness Evaluasi JavaScript AksaraScan (ONNX Runtime Web)
 * Menjalankan model ONNX yang sama persis (`app/web_camscanner/aksara_yolo.onnx`)
 * dalam dua mode:
 *   - `--mode browser` (bawaan): mengeksekusi pipeline penuh `app.js` + `orthography.js` + `javanese_lexicon.js`
 *     menggunakan `onnxruntime-web` WASM di dalam Chromium tanpa koneksi jaringan eksternal.
 *   - `--mode raw-ort`: menjalankan inferensi letterbox 416x416 murni di Node dengan `onnxruntime-web` WASM + `sharp`
 *     untuk uji paritas numerik langsung terhadap `onnxruntime` Python.
 *
 * Penggunaan:
 *   node eval/run_eval.mjs --manifest eval/parity_20.jsonl --mode browser --out eval/artifacts/js_browser.json
 */
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath, pathToFileURL } from 'node:url';
import sharp from 'sharp';
import * as ort from 'onnxruntime-web';
import { chromium } from 'playwright-core';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const WEB_DIR = path.join(ROOT_DIR, 'app', 'web_camscanner');
const ORT_DIST = path.join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');

function parseArgs() {
  const args = process.argv.slice(2);
  const cfg = {
    manifest: path.join(__dirname, 'parity_20.jsonl'),
    mode: 'browser',
    out: null,
    limit: 0,
    dumpTensorDir: null,
    confThresh: null,
  };
  for (let i = 0; i < args.length; i++) {
    const k = args[i];
    if (k === '--manifest') cfg.manifest = path.resolve(ROOT_DIR, args[++i]);
    else if (k === '--mode') cfg.mode = args[++i];
    else if (k === '--out') cfg.out = path.resolve(ROOT_DIR, args[++i]);
    else if (k === '--limit') cfg.limit = parseInt(args[++i], 10);
    else if (k === '--dump-tensor-dir') cfg.dumpTensorDir = path.resolve(ROOT_DIR, args[++i]);
    else if (k === '--conf') cfg.confThresh = parseFloat(args[++i]);
  }
  return cfg;
}

function loadManifest(manifestPath, limit) {
  const lines = fs.readFileSync(manifestPath, 'utf-8')
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(Boolean);
  const items = lines.map(l => JSON.parse(l));
  return limit > 0 ? items.slice(0, limit) : items;
}

function resolveImgPath(imgRel) {
  return path.isAbsolute(imgRel) ? imgRel : path.resolve(ROOT_DIR, imgRel);
}

const MIME_MAP = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.webmanifest': 'application/manifest+json',
};

function startLocalServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = (req.url || '/').split('?')[0];
      let filePath = null;
      if (reqPath.startsWith('/__ort__/')) {
        filePath = path.join(ORT_DIST, reqPath.replace('/__ort__/', ''));
      } else {
        const cleanRel = reqPath === '/' ? 'index.html' : reqPath.replace(/^\/+/, '');
        filePath = path.join(WEB_DIR, cleanRel);
      }
      if (!filePath || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.statusCode = 404;
        res.end('Not found');
        return;
      }
      const ext = path.extname(filePath).toLowerCase();
      res.setHeader('Content-Type', MIME_MAP[ext] || 'application/octet-stream');
      fs.createReadStream(filePath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, url: `http://127.0.0.1:${addr.port}` });
    });
  });
}

function findChromeExecutable() {
  const candidates = [
    process.env.CHROME_EXE,
    path.join(process.env.LOCALAPPDATA || '', 'ms-playwright', 'chromium-1234', 'chrome-win64', 'chrome.exe'),
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return undefined;
}

function yoloIoU(a, b) {
  const ix0 = Math.max(a[0], b[0]), iy0 = Math.max(a[1], b[1]);
  const ix1 = Math.min(a[2], b[2]), iy1 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix1 - ix0), ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

function yoloNms(candidates, sameCatIou = 0.42, crossCatIou = 0.72) {
  candidates.sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const cand of candidates) {
    let suppress = false;
    for (const k of kept) {
      const sameExclusiveGroup = (cand.category !== 'sandhangan' && k.category !== 'sandhangan');
      const thr = (cand.category === k.category || sameExclusiveGroup) ? sameCatIou : crossCatIou;
      if (yoloIoU(cand.box, k.box) > thr) { suppress = true; break; }
    }
    if (!suppress) kept.push(cand);
  }
  return kept;
}

async function runRawOrtMode(items, cfg) {
  const meta = JSON.parse(fs.readFileSync(path.join(WEB_DIR, 'yolo_meta.json'), 'utf-8'));
  const imgsz = meta.imgsz || 416;
  const confThr = cfg.confThresh ?? meta.conf_thresh ?? 0.25;
  const iouThr = meta.iou_thresh ?? 0.42;
  const nc = meta.classes.length;

  ort.env.wasm.wasmPaths = pathToFileURL(ORT_DIST).href + '/';
  ort.env.wasm.numThreads = 1;
  const onnxBytes = fs.readFileSync(path.join(WEB_DIR, 'aksara_yolo.onnx'));
  const session = await ort.InferenceSession.create(onnxBytes, { executionProviders: ['wasm'] });
  const inpName = session.inputNames[0];
  const outName = session.outputNames[0];

  if (cfg.dumpTensorDir) fs.mkdirSync(cfg.dumpTensorDir, { recursive: true });

  const results = [];
  for (const it of items) {
    const imgPath = resolveImgPath(it.image);
    const t0 = performance.now();
    const imgSharp = sharp(imgPath).removeAlpha().toColorspace('srgb');
    const metadata = await imgSharp.metadata();
    const origW = metadata.width;
    const origH = metadata.height;

    const scale = Math.min(imgsz / origW, imgsz / origH);
    const nw = Math.round(origW * scale);
    const nh = Math.round(origH * scale);
    const padX = Math.floor((imgsz - nw) / 2);
    const padY = Math.floor((imgsz - nh) / 2);

    const resizedBuf = await imgSharp
      .resize(nw, nh, { fit: 'fill', kernel: sharp.kernel.linear })
      .extend({
        top: padY,
        bottom: imgsz - nh - padY,
        left: padX,
        right: imgsz - nw - padX,
        background: { r: 114, g: 114, b: 114 },
      })
      .raw()
      .toBuffer();

    const plane = imgsz * imgsz;
    const chw = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      const p = i * 3;
      chw[i] = resizedBuf[p] / 255.0;
      chw[plane + i] = resizedBuf[p + 1] / 255.0;
      chw[2 * plane + i] = resizedBuf[p + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', chw, [1, 3, imgsz, imgsz]);
    const outMap = await session.run({ [inpName]: tensor });
    const outTensor = outMap[outName];
    const numAnchors = outTensor.dims[2];
    const raw = outTensor.data;

    if (cfg.dumpTensorDir) {
      fs.writeFileSync(path.join(cfg.dumpTensorDir, `${it.id}_input.bin`), Buffer.from(chw.buffer));
      fs.writeFileSync(path.join(cfg.dumpTensorDir, `${it.id}_output.bin`), Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength));
    }

    const cands = [];
    for (let a = 0; a < numAnchors; a++) {
      let bestCls = -1, bestScore = confThr;
      for (let c = 0; c < nc; c++) {
        const sc = raw[(4 + c) * numAnchors + a];
        if (sc > bestScore) { bestScore = sc; bestCls = c; }
      }
      if (bestCls < 0) continue;
      const cx = raw[a], cy = raw[numAnchors + a], bw = raw[2 * numAnchors + a], bh = raw[3 * numAnchors + a];
      const x0 = Math.max(0, (cx - bw / 2 - padX) / scale);
      const y0 = Math.max(0, (cy - bh / 2 - padY) / scale);
      const x1 = Math.min(origW, (cx + bw / 2 - padX) / scale);
      const y1 = Math.min(origH, (cy + bh / 2 - padY) / scale);
      if (x1 - x0 < 3 || y1 - y0 < 3) continue;
      const label = meta.classes[bestCls];
      cands.push({
        classIdx: bestCls,
        label,
        category: meta.categories[bestCls],
        unicode: meta.unicode[label] || '',
        confidence: bestScore,
        box: [x0, y0, x1, y1],
      });
    }
    const kept = yoloNms(cands, iouThr, 0.72);
    const dt = performance.now() - t0;

    results.push({
      id: it.id,
      image: it.image,
      origW,
      origH,
      elapsedMs: Math.round(dt * 100) / 100,
      detections: kept,
    });
  }
  return { mode: 'raw-ort', engine: 'onnxruntime-web@1.20.1-wasm-node', results };
}

async function runBrowserMode(items, cfg) {
  const { server, url } = await startLocalServer();
  const exe = findChromeExecutable();
  const browser = await chromium.launch({ executablePath: exe, headless: true });
  const externalRequestsInit = [];
  const externalRequestsDuringInference = [];
  let inInference = false;
  const consoleErrors = [];

  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();

    page.on('pageerror', err => consoleErrors.push(String(err)));
    page.on('console', msg => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });
    await page.route('**/*', async (route) => {
      const rUrl = route.request().url();
      if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) {
        await route.continue();
      } else if (rUrl.startsWith('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/')) {
        const fname = path.basename(rUrl.split('?')[0]);
        const localFile = path.join(ORT_DIST, fname);
        if (fs.existsSync(localFile)) {
          const ext = path.extname(localFile).toLowerCase();
          await route.fulfill({
            status: 200,
            contentType: MIME_MAP[ext] || 'application/octet-stream',
            body: fs.readFileSync(localFile),
          });
        } else {
          (inInference ? externalRequestsDuringInference : externalRequestsInit).push(rUrl);
          await route.abort();
        }
      } else {
        (inInference ? externalRequestsDuringInference : externalRequestsInit).push(rUrl);
        if (rUrl.includes('fonts.googleapis.com')) {
          await route.fulfill({ status: 200, contentType: 'text/css', body: '' });
        } else {
          await route.abort();
        }
      }
    });

    await page.goto(`${url}/index.html`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.querySelector('#engineStatus')?.textContent.includes('YOLOv8 Siap'),
      null,
      { timeout: 30000 },
    );
    inInference = true;

    const results = [];
    for (let idx = 0; idx < items.length; idx++) {
      const it = items[idx];
      const imgPath = resolveImgPath(it.image);
      const t0 = performance.now();

      await page.evaluate(() => {
        const sd = document.querySelector('#srcDim');
        if (sd) sd.textContent = '__LOADING__';
        window.__aksEvalRunDone = false;
        const btn = document.querySelector('#btnRun');
        let sawDisabled = false;
        const obs = new MutationObserver(() => {
          if (btn.disabled) sawDisabled = true;
          else if (sawDisabled && !btn.disabled) {
            window.__aksEvalRunDone = true;
            obs.disconnect();
          }
        });
        obs.observe(btn, { attributes: true, attributeFilter: ['disabled'] });
      });

      await page.locator('#fileInput').setInputFiles(imgPath);
      await page.waitForFunction(() => window.__aksEvalRunDone === true, null, { timeout: 60000 });
      await page.waitForTimeout(40);
      const dt = performance.now() - t0;

      const snap = await page.evaluate(() => {
        const meta = {
          canvasW: document.querySelector('#mainCanvas')?.width || 0,
          canvasH: document.querySelector('#mainCanvas')?.height || 0,
          unicodeText: document.querySelector('#outJavanese')?.textContent || '—',
          sentenceLatin: document.querySelector('#outSentenceLatin')?.textContent || '—',
          sentenceIndonesian: document.querySelector('#outIndonesian')?.textContent || '—',
          latinPerSyllable: document.querySelector('#outLatin')?.textContent || '—',
          avgConfText: document.querySelector('#meterVal')?.textContent || '0%',
          cardCount: document.querySelectorAll('#glyphsGrid .glyph-card').length,
        };
        const glyphs = Array.from(document.querySelectorAll('#glyphsGrid .glyph-card')).map(card => ({
          unicode: card.querySelector('.glyph-card__unicode')?.textContent || '',
          label: card.querySelector('.glyph-card__label')?.textContent || '',
          confPct: card.querySelector('.glyph-card__conf')?.textContent || '',
          uncertain: card.classList.contains('uncertain'),
        }));
        return { ...meta, glyphs };
      });

      // Extract exact currentDetections boxes via #btnDownload blob intercept or window scope fallback
      // Note: currentDetections is module-scoped in app.js, so let's capture Download JSON cleanly!
      const jsonExport = await page.evaluate(async () => {
        let captured = null;
        const origCreateObjectURL = URL.createObjectURL;
        URL.createObjectURL = (blob) => {
          captured = blob;
          return 'blob:noop';
        };
        const btn = document.querySelector('#btnDownload');
        const origClick = HTMLAnchorElement.prototype.click;
        HTMLAnchorElement.prototype.click = () => {};
        try {
          if (btn && document.querySelectorAll('#glyphsGrid .glyph-card').length > 0) {
            btn.click();
          }
        } finally {
          HTMLAnchorElement.prototype.click = origClick;
          URL.createObjectURL = origCreateObjectURL;
        }
        if (captured) {
          const text = await captured.text();
          return JSON.parse(text);
        }
        return null;
      });

      results.push({
        id: it.id,
        image: it.image,
        canvasW: snap.canvasW,
        canvasH: snap.canvasH,
        elapsedMs: Math.round(dt * 100) / 100,
        unicodeText: snap.unicodeText,
        sentenceLatin: snap.sentenceLatin,
        sentenceIndonesian: snap.sentenceIndonesian,
        latinPerSyllable: snap.latinPerSyllable,
        detections: jsonExport ? jsonExport.deteksi : [],
      });
    }

    return {
      mode: 'browser',
      engine: 'chromium-playwright+onnxruntime-web@1.20.1',
      consoleErrors,
      externalNetworkRequestsDuringInference: externalRequestsDuringInference,
      externalNetworkRequestsOnInit: externalRequestsInit,
      results,
    };
  } finally {
    await browser.close();
    server.close();
  }
}

async function main() {
  const cfg = parseArgs();
  const items = loadManifest(cfg.manifest, cfg.limit);
  const payload = cfg.mode === 'raw-ort'
    ? await runRawOrtMode(items, cfg)
    : await runBrowserMode(items, cfg);

  if (cfg.out) {
    fs.mkdirSync(path.dirname(cfg.out), { recursive: true });
    fs.writeFileSync(cfg.out, JSON.stringify(payload, null, 2), 'utf-8');
    console.log(`Saved JS eval results (${payload.results.length} items) -> ${cfg.out}`);
  } else {
    console.log(JSON.stringify(payload, null, 2));
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
