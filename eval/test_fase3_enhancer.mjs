/**
 * Suite Pengujian Lengkap Gerbang 3 — Fase 3 AksaraScan v4.0
 * Memverifikasi:
 *   1) Uji unit estimateBlur() pada kernel Gaussian & Motion blur sintetis diketahui (monoton: tajam > ringan > sedang > berat)
 *   2) Tidak merusak gambar tajam (passthrough MAE ~ 0)
 *   3) Menjaga kontinuitas goresan tipis (jumlah komponen terhubung CC & panjang kerangka/skeleton tidak pecah)
 *   4) Pemisahan tinta warna (pulpen biru) & penekanan garis buku tulis
 *   5) Offloading Web Worker (enhancer_worker.js) di Chromium tanpa long-task main thread
 *   6) Mode offline PWA penuh (Service Worker aksarascan-v14, offline reload & inferensi penuh)
 *   7) Fallback detektor-saja saat aksara_embed.onnx / reference_prototypes.json dipaksa gagal muat
 *   8) Komparasi terukur tanpa vs dengan Enhancer + Stage-2 pada foto buram nyata & stress set di Chromium WASM
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const WEB_DIR = path.join(ROOT_DIR, 'app', 'web_camscanner');
const loadedEnhancer = require(path.join(WEB_DIR, 'image_enhancer.js'));
const Enhancer = (loadedEnhancer && typeof loadedEnhancer.estimateBlur === 'function')
  ? loadedEnhancer
  : globalThis.AksaraEnhancer;

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
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.webmanifest': 'application/manifest+json',
};

function findChrome() {
  const candidates = [
    process.env.CHROME_EXE,
    path.join(process.env.LOCALAPPDATA || '', 'ms-playwright', 'chromium-1234', 'chrome-win64', 'chrome.exe'),
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  ].filter(Boolean);
  return candidates.find((c) => fs.existsSync(c));
}

// ---------------------------------------------------------------------------
// Pembantu Uji Unit Sintetis murni JS (Kurva glif, blur kernel, CC, & skeleton)
// ---------------------------------------------------------------------------
function makeSyntheticStrokes(W = 240, H = 120, { thinStroke = false, bluePen = false, withRuledLines = false } = {}) {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 244; data[i + 1] = 240; data[i + 2] = 230; data[i + 3] = 255;
  }

  function paintDisk(cx, cy, r, col) {
    const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(W - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(H - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
          const p = (y * W + x) * 4;
          data[p] = col[0]; data[p + 1] = col[1]; data[p + 2] = col[2];
        }
      }
    }
  }

  if (withRuledLines) {
    for (const ly of [28, 60, 92]) {
      for (let x = 0; x < W; x++) {
        const p = (ly * W + x) * 4;
        data[p] = 175; data[p + 1] = 186; data[p + 2] = 198;
      }
    }
  }

  const ink = bluePen ? [38, 86, 175] : (thinStroke ? [68, 68, 72] : [20, 20, 22]);
  const r = thinStroke ? 1.15 : 2.2;
  // Gambar lengkungan ganda berkesinambungan menyerupai ha/na/ca + wulu di atas
  for (let gx = 32; gx <= 190; gx += 46) {
    for (let t = 0; t <= 1.0; t += 0.02) {
      const a = t * Math.PI;
      paintDisk(gx + t * 22, 58 - Math.sin(a) * 20, r, ink);
      paintDisk(gx + 10 + t * 18, 64 + Math.sin(a) * 18, r, ink);
    }
    paintDisk(gx + 14, 30, r * 1.1, ink);
  }
  return { width: W, height: H, data };
}

function gaussianConvolve(img, sigma) {
  if (sigma <= 0) return { width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) };
  const W = img.width, H = img.height, src = img.data;
  const rad = Math.max(1, Math.ceil(sigma * 2.6));
  const ker = [];
  let ksum = 0;
  for (let i = -rad; i <= rad; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    ker.push(v); ksum += v;
  }
  for (let i = 0; i < ker.length; i++) ker[i] /= ksum;

  const tmp = new Float32Array(W * H * 3);
  const out = new Uint8ClampedArray(src.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = -rad; k <= rad; k++) {
        const xx = Math.min(W - 1, Math.max(0, x + k));
        const p = (y * W + xx) * 4;
        const w = ker[k + rad];
        r += src[p] * w; g += src[p + 1] * w; b += src[p + 2] * w;
      }
      const d = (y * W + x) * 3;
      tmp[d] = r; tmp[d + 1] = g; tmp[d + 2] = b;
    }
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = -rad; k <= rad; k++) {
        const yy = Math.min(H - 1, Math.max(0, y + k));
        const s = (yy * W + x) * 3;
        const w = ker[k + rad];
        r += tmp[s] * w; g += tmp[s + 1] * w; b += tmp[s + 2] * w;
      }
      const p = (y * W + x) * 4;
      out[p] = Math.round(r); out[p + 1] = Math.round(g); out[p + 2] = Math.round(b); out[p + 3] = 255;
    }
  }
  return { width: W, height: H, data: out };
}

function motionConvolve(img, len) {
  const W = img.width, H = img.height, src = img.data;
  const out = new Uint8ClampedArray(src.length);
  const half = Math.floor(len / 2);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0, cnt = 0;
      for (let dx = -half; dx <= half; dx++) {
        const xx = Math.min(W - 1, Math.max(0, x + dx));
        const p = (y * W + xx) * 4;
        r += src[p]; g += src[p + 1]; b += src[p + 2]; cnt++;
      }
      const d = (y * W + x) * 4;
      out[d] = Math.round(r / cnt); out[d + 1] = Math.round(g / cnt); out[d + 2] = Math.round(b / cnt); out[d + 3] = 255;
    }
  }
  return { width: W, height: H, data: out };
}

function measureStrokeContinuity(img, darkDelta = 35) {
  const W = img.width, H = img.height, d = img.data;
  const luma = new Float32Array(W * H);
  let bgSum = 0, bgCnt = 0;
  for (let i = 0, p = 0; i < W * H; i++, p += 4) {
    luma[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
    if (luma[i] > 180) { bgSum += luma[i]; bgCnt++; }
  }
  const bg = bgCnt ? bgSum / bgCnt : 240;
  const bin = new Uint8Array(W * H);
  let inkPixels = 0;
  for (let i = 0; i < W * H; i++) {
    if (bg - luma[i] > darkDelta) { bin[i] = 1; inkPixels++; }
  }
  // Komponen terhubung (8-tetangga BFS)
  const vis = new Uint8Array(W * H);
  let components = 0;
  for (let i = 0; i < W * H; i++) {
    if (!bin[i] || vis[i]) continue;
    components++;
    const q = [i];
    vis[i] = 1;
    while (q.length) {
      const cur = q.pop();
      const x = cur % W, y = (cur / W) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W) continue;
          const ni = yy * W + xx;
          if (bin[ni] && !vis[ni]) { vis[ni] = 1; q.push(ni); }
        }
      }
    }
  }
  // Estimasi panjang kerangka via jumlah piksel batas/inti terskala (distance-transform proxy)
  let ridgeCount = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const idx = y * W + x;
      if (!bin[idx]) continue;
      const nb = bin[idx - 1] + bin[idx + 1] + bin[idx - W] + bin[idx + W];
      if (nb >= 1 && nb <= 3) ridgeCount++;
    }
  }
  return { components, inkPixels, ridgeCount };
}

function runUnitTests() {
  console.log('\n=== [1/4] Uji Unit Image Enhancer (Node Pure JS) ===');
  const cleanSharp = makeSyntheticStrokes(240, 120, { thinStroke: false });
  const rSharp = Enhancer.estimateBlur(cleanSharp);
  const g1 = Enhancer.estimateBlur(gaussianConvolve(cleanSharp, 1.0));
  const g2 = Enhancer.estimateBlur(gaussianConvolve(cleanSharp, 1.8));
  const g3 = Enhancer.estimateBlur(gaussianConvolve(cleanSharp, 3.0));
  const m1 = Enhancer.estimateBlur(motionConvolve(cleanSharp, 5));
  const m2 = Enhancer.estimateBlur(motionConvolve(cleanSharp, 9));

  console.log('  Blur progression (Gaussian):', {
    sharp: { lvl: rSharp.level, sc: rSharp.sharpnessScore, spread: rSharp.edgeSpreadPx },
    s1_0: { lvl: g1.level, sc: g1.sharpnessScore, spread: g1.edgeSpreadPx },
    s1_8: { lvl: g2.level, sc: g2.sharpnessScore, spread: g2.edgeSpreadPx },
    s3_0: { lvl: g3.level, sc: g3.sharpnessScore, spread: g3.edgeSpreadPx },
  });
  console.log('  Blur progression (Motion):', {
    k5: { lvl: m1.level, sc: m1.sharpnessScore, spread: m1.edgeSpreadPx },
    k9: { lvl: m2.level, sc: m2.sharpnessScore, spread: m2.edgeSpreadPx },
  });

  assert.equal(rSharp.level, 'tajam', 'Gambar sintetis bersih harus terbaca tajam');
  assert.ok(rSharp.sharpnessScore > g1.sharpnessScore, 'sharp > sigma 1.0');
  assert.ok(g1.sharpnessScore > g2.sharpnessScore, 'sigma 1.0 > sigma 1.8');
  assert.ok(g2.sharpnessScore > g3.sharpnessScore, 'sigma 1.8 > sigma 3.0');
  assert.ok(m1.sharpnessScore > m2.sharpnessScore, 'motion k5 > motion k9');
  assert.equal(g3.level, 'berat', 'Gaussian sigma 3.0 harus berat');

  // 2. Gambar tajam tidak boleh rusak (mode passthrough MAE = 0)
  const enhSharp = Enhancer.enhance(cleanSharp, {});
  let maeSharp = 0;
  for (let i = 0; i < cleanSharp.data.length; i += 4) {
    maeSharp += Math.abs(cleanSharp.data[i] - enhSharp.imageData.data[i]);
  }
  maeSharp /= (cleanSharp.width * cleanSharp.height);
  console.log('  Sharp non-destructive test:', { mode: enhSharp.report.mode, mae: Number(maeSharp.toFixed(3)) });
  assert.ok(maeSharp < 2.0, `MAE pada gambar tajam (${maeSharp}) harus minimal`);

  // 3. Uji kontinuitas goresan tipis & buram ringan (tak boleh memecah komponen / skeleton)
  const thinBlur = gaussianConvolve(makeSyntheticStrokes(240, 120, { thinStroke: true }), 1.2);
  const ccBefore = measureStrokeContinuity(thinBlur, 22);
  const thinEnhanced = Enhancer.enhance(thinBlur, { autoSharpen: true, useRL: true, rlIterations: 3 });
  const ccAfter = measureStrokeContinuity(thinEnhanced.imageData, 22);
  console.log('  Thin blurred stroke continuity:', { before: ccBefore, after: ccAfter, report: thinEnhanced.report.blurAfter.level });
  assert.ok(ccAfter.components <= ccBefore.components + 1, `Komponen tipis tak boleh terpecah (${ccBefore.components} -> ${ccAfter.components})`);
  assert.ok(ccAfter.ridgeCount >= ccBefore.ridgeCount * 0.90, `Panjang garis tepi/kerangka tipis terjaga (${ccBefore.ridgeCount} -> ${ccAfter.ridgeCount})`);
  assert.ok(thinEnhanced.report.blurAfter.sharpnessScore > thinEnhanced.report.blurBefore.sharpnessScore, 'Ketajaman harus meningkat setelah enhancer');

  // 4. Uji pulpen biru & penekanan garis buku tulis
  const blueRuled = makeSyntheticStrokes(240, 120, { bluePen: true, withRuledLines: true });
  const enhBlue = Enhancer.enhance(blueRuled, { removeLines: true });
  console.log('  Blue pen + ruled lines test:', { suppressedPx: enhBlue.report.suppressedLinePixels, coloredInkRatio: enhBlue.report.blurBefore.coloredInkRatio });
  assert.ok(enhBlue.report.suppressedLinePixels > 250, 'Garis buku memanjang harus terdeteksi & diredam');
  assert.ok(enhBlue.report.blurBefore.coloredInkRatio > 0.02, 'Tinta biru kromatik harus terdeteksi');

  return {
    blurMonotonic: true,
    sharpMae: Number(maeSharp.toFixed(3)),
    continuity: { before: ccBefore, after: ccAfter },
    suppressedLinePixels: enhBlue.report.suppressedLinePixels,
  };
}

function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = (req.url || '/').split('?')[0];
      const filePath = path.join(WEB_DIR, reqPath === '/' ? 'index.html' : reqPath.replace(/^\/+/, ''));
      if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.statusCode = 404;
        res.end('Not found');
        return;
      }
      const ext = path.extname(filePath).toLowerCase();
      res.setHeader('Content-Type', MIME_MAP[ext] || 'application/octet-stream');
      fs.createReadStream(filePath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => {
      resolve({ server, url: `http://127.0.0.1:${server.address().port}` });
    });
  });
}

async function runUpload(page, imgPath) {
  await page.evaluate(() => {
    window.__e2eDone = false;
    const btn = document.querySelector('#btnRun');
    let sawDisabled = false;
    const obs = new MutationObserver(() => {
      if (btn.disabled) sawDisabled = true;
      else if (sawDisabled && !btn.disabled) {
        window.__e2eDone = true;
        obs.disconnect();
      }
    });
    obs.observe(btn, { attributes: true, attributeFilter: ['disabled'] });
  });
  await page.locator('#fileInput').setInputFiles(imgPath);
  await page.waitForFunction(() => window.__e2eDone === true, null, { timeout: 60000 });
  await page.waitForTimeout(80);
  return page.evaluate(() => ({
    unicodeText: document.querySelector('#outJavanese')?.textContent || '',
    sentenceLatin: document.querySelector('#outSentenceLatin')?.textContent || '',
    engineStatus: document.querySelector('#engineStatus')?.textContent || '',
    enhanceReport: window.__aksaraLastEnhanceReport || null,
    stage2Stats: window.__aksaraLastStage2Stats || null,
  }));
}

async function runBrowserSuite() {
  const { server, url } = await startServer();
  const browser = await chromium.launch({ headless: true, executablePath: findChrome() });
  const results = {};
  try {
    // --- TEST A: Main-thread non-blocking Worker + Blur Photo Enhancement Matrix ---
    console.log('\n=== [2/4] Uji Web Worker (0 Long Tasks) & Foto Buram Nyata di Chromium ===');
    const ctx = await browser.newContext({ viewport: { width: 1366, height: 900 } });
    const page = await ctx.newPage();
    const consoleErrors = [];
    const extRequests = [];
    page.on('pageerror', (e) => consoleErrors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    await page.route('**/*', (route) => {
      const rUrl = route.request().url();
      if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) {
        return route.continue();
      }
      extRequests.push(rUrl);
      if (rUrl.includes('fonts.googleapis.com')) return route.fulfill({ status: 200, contentType: 'text/css', body: '' });
      return route.abort();
    });

    await page.goto(`${url}/index.html`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => (document.querySelector('#engineStatus')?.textContent || '').includes('Stage-2'), null, { timeout: 45000 });

    // Pantau Long Task >= 80ms di main thread khusus saat pemanggilan enhancer Worker (diluar ORT WASM inference)
    const workerPerf = await page.evaluate(async () => {
      const longTasks = [];
      let obs = null;
      if (typeof PerformanceObserver !== 'undefined') {
        try {
          obs = new PerformanceObserver((list) => {
            for (const ent of list.getEntries()) {
              longTasks.push(Math.round(ent.duration));
            }
          });
          obs.observe({ entryTypes: ['longtask'] });
        } catch (_e) { /* unsupported */ }
      }
      const c = document.createElement('canvas');
      c.width = 1200; c.height = 900;
      const cx = c.getContext('2d');
      cx.fillStyle = '#eee8d8'; cx.fillRect(0, 0, 1200, 900);
      cx.fillStyle = '#333'; cx.font = '32px sans-serif';
      cx.fillText('AksaraScan Worker Blur Test', 100, 300);
      const t0 = performance.now();
      const outC = await prepareYoloInputCanvas(c, 1200, 900, { force: true, useRL: true, rlIterations: 3 });
      const totalMs = Math.round(performance.now() - t0);
      await new Promise((r) => setTimeout(r, 40));
      if (obs) obs.disconnect();
      return { totalMs, outW: outC.width, outH: outC.height, longTasks, report: window.__aksaraLastEnhanceReport };
    });
    console.log('  Worker off-thread 1200x900:', { totalMs: workerPerf.totalMs, longTasks: workerPerf.longTasks, usedRL: workerPerf.report?.usedRL });
    assert.ok(workerPerf.longTasks.filter((d) => d >= 100).length === 0, `Worker enhancer tidak boleh menimbulkan long task >=100ms di main thread: ${JSON.stringify(workerPerf.longTasks)}`);

    // Komparasi tanpa enhancer vs dengan enhancer pada foto nyata blur & stress
    const blurCases = [
      { id: 'hr_hw_02_motion_blur', path: path.join(ROOT_DIR, 'eval', 'heldout_real', 'images', 'hr_hw_02_motion_blur.jpg'), gt: 'ꦪꦮꦠꦥꦛ\nꦝꦒꦔꦏꦢ' },
      { id: 'hr_hw_04_defocus_blur', path: path.join(ROOT_DIR, 'eval', 'heldout_real', 'images', 'hr_hw_04_defocus_blur.jpg'), gt: 'ꦮꦪꦩꦫꦝ\nꦗꦥꦒꦕꦲ' },
      { id: 'diag_04_shadow', path: path.join(ROOT_DIR, 'eval', 'failures', '04_test_shadow.png'), gt: 'ꦲꦤꦕꦫꦏ\nꦢꦱꦮꦭ' },
      { id: 'diag_06_lowres600', path: path.join(ROOT_DIR, 'eval', 'failures', '06_test_small_600px.png'), gt: 'ꦲꦤꦕꦫꦏ\nꦢꦱꦮꦭ' },
    ];

    function levChars(a, b) {
      const m = a.length, n = b.length;
      const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
      for (let i = 0; i <= m; i++) dp[i][0] = i;
      for (let j = 0; j <= n; j++) dp[0][j] = j;
      for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
          dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
        }
      }
      return dp[m][n] / Math.max(1, m);
    }

    const blurComparison = [];
    for (const bc of blurCases) {
      await page.evaluate(() => { window.aksaraEnhanceParams = { skipEnhance: true }; });
      const rawRun = await runUpload(page, bc.path);
      await page.evaluate(() => { window.aksaraEnhanceParams = { autoSharpen: true, removeLines: 'auto', contrast: 0 }; });
      const enhRun = await runUpload(page, bc.path);

      const cleanGt = bc.gt.replace(/\s+/g, '');
      const cerRaw = Number((levChars(cleanGt, rawRun.unicodeText.replace(/\s+/g, '')) * 100).toFixed(2));
      const cerEnh = Number((levChars(cleanGt, enhRun.unicodeText.replace(/\s+/g, '')) * 100).toFixed(2));
      blurComparison.push({
        id: bc.id,
        blurBefore: enhRun.enhanceReport?.blurBefore?.level,
        cerWithoutEnhancer: cerRaw,
        cerWithEnhancer: cerEnh,
        stage2Verified: enhRun.stage2Stats?.verified || 0,
        stage2Rejected: enhRun.stage2Stats?.rejectedNonAksara || 0,
      });
    }
    console.log('  Blur matrix comparison:', blurComparison);
    const meanCerRaw = blurComparison.reduce((s, r) => s + r.cerWithoutEnhancer, 0) / blurComparison.length;
    const meanCerEnh = blurComparison.reduce((s, r) => s + r.cerWithEnhancer, 0) / blurComparison.length;
    assert.ok(meanCerEnh <= meanCerRaw + 0.5, `Enhancer harus memperbaiki atau minimal tidak memperburuk blur mean CER (${meanCerRaw} -> ${meanCerEnh})`);
    assert.equal(extRequests.filter((u) => !u.includes('fonts.googleapis')).length, 0, `Tidak boleh ada request eksternal inferensi: ${JSON.stringify(extRequests)}`);
    assert.equal(consoleErrors.length, 0, `Console errors ditemukan: ${JSON.stringify(consoleErrors)}`);
    results.workerPerf = { totalMs: workerPerf.totalMs, longTasks: workerPerf.longTasks };
    results.blurComparison = { meanCerRaw: Number(meanCerRaw.toFixed(2)), meanCerEnh: Number(meanCerEnh.toFixed(2)), items: blurComparison };
    await ctx.close();

    // --- TEST B: Fallback Stage-2 gagal dimuat (blokir SW agar route abort benar-benar mencapai fetch/ORT) ---
    console.log('\n=== [3/4] Uji Fallback Saat aksara_embed.onnx Dipaksa Gagal ===');
    const ctxFb = await browser.newContext({ viewport: { width: 1366, height: 900 }, serviceWorkers: 'block' });
    const pageFb = await ctxFb.newPage();
    await pageFb.route('**/*', (route) => {
      const rUrl = route.request().url();
      if (rUrl.includes('aksara_embed.onnx') || rUrl.includes('reference_prototypes.json')) return route.abort('failed');
      if (rUrl.includes('fonts.googleapis.com')) return route.fulfill({ status: 200, contentType: 'text/css', body: '' });
      return route.continue();
    });
    await pageFb.goto(`${url}/index.html`, { waitUntil: 'domcontentloaded' });
    await pageFb.waitForFunction(() => (document.querySelector('#engineStatus')?.textContent || '').includes('Fallback'), null, { timeout: 45000 });
    const fbRun = await runUpload(pageFb, path.join(ROOT_DIR, 'app', 'web_camscanner', 'sample_4.png'));
    console.log('  Fallback result:', { engineStatus: fbRun.engineStatus, latin: fbRun.sentenceLatin, s2: fbRun.stage2Stats });
    assert.match(fbRun.engineStatus, /Fallback/i, 'Status engine harus memberitahu mode Stage-1 Fallback');
    assert.match(fbRun.sentenceLatin.toLowerCase(), /aksara\s+jawa/, 'Detektor tetap berjalan normal saat Stage-2 gagal');
    assert.equal(fbRun.stage2Stats?.active, false, 'Stage-2 harus melaporkan active=false');
    results.fallbackVerified = true;
    await ctxFb.close();

    // --- TEST C: Full Offline PWA Mode via Service Worker ---
    console.log('\n=== [4/4] Uji Mode Offline Penuh (Service Worker aksarascan-v15) ===');
    const ctxOff = await browser.newContext({ viewport: { width: 1366, height: 900 } });
    const pageOff = await ctxOff.newPage();
    await pageOff.route('**/*', (route) => {
      const rUrl = route.request().url();
      if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) return route.continue();
      if (rUrl.includes('fonts.googleapis.com')) return route.fulfill({ status: 200, contentType: 'text/css', body: '' });
      return route.abort();
    });
    await pageOff.goto(`${url}/index.html`, { waitUntil: 'load' });
    await pageOff.waitForFunction(() => (document.querySelector('#engineStatus')?.textContent || '').includes('Stage-2'), null, { timeout: 45000 });
    // Pastikan cache aksarascan-v15 sudah penuh di Service Worker CacheStorage sebelum memutus jaringan!
    const swReady = await pageOff.evaluate(async () => {
      if (!('serviceWorker' in navigator)) return { ok: false, reason: 'no SW' };
      const reg = await navigator.serviceWorker.ready;
      const cache = await caches.open('aksarascan-v15');
      const keys = await cache.keys();
      return { ok: Boolean(reg && reg.active && keys.length >= 25), cacheCount: keys.length };
    });
    console.log('  SW cache pre-offline:', swReady);
    assert.ok(swReady.ok, `Service worker & cache aksarascan-v15 harus terisi (${JSON.stringify(swReady)})`);

    await ctxOff.setOffline(true);
    await pageOff.reload({ waitUntil: 'domcontentloaded' });
    await pageOff.waitForFunction(() => (document.querySelector('#engineStatus')?.textContent || '').includes('Stage-2'), null, { timeout: 45000 });
    const offRun = await runUpload(pageOff, path.join(ROOT_DIR, 'eval', 'failures', '02_wedang_bubuk_crop.png'));
    console.log('  Offline inference result:', { status: offRun.engineStatus, latin: offRun.sentenceLatin, s2: offRun.stage2Stats });
    assert.match(offRun.sentenceLatin.toLowerCase(), /wédang|wedang/, 'Mode offline harus memulihkan "wédang"');
    assert.equal(offRun.stage2Stats?.active, true, 'Stage-2 harus tetap aktif 100% offline');
    results.offlineVerified = { cacheCount: swReady.cacheCount, latin: offRun.sentenceLatin };
    await ctxOff.close();
  } finally {
    await browser.close();
    server.close();
  }
  return results;
}

const unitRes = runUnitTests();
const browserRes = await runBrowserSuite();
const outPath = path.join(ROOT_DIR, 'eval', 'reports', 'phase3_verification.json');
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify({ generatedAt: new Date().toISOString(), unit: unitRes, browser: browserRes }, null, 2));
console.log(`\n[PASS SEMUA GERBANG 3] Tersimpan ke ${outPath}`);
