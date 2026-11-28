/**
 * Uji End-to-End Berbasis Asersi Positif AksaraScan v4.0 (Fase 5 — Matriks Penuh)
 * Menguji:
 *  1) Matriks 4 kombinasi: Desktop (1440×900) & Mobile (390×844) × Tema Gelap & Terang
 *     - Asersi positif di keadaan awal (dropzone, 4 contoh uji, 0 overflow, validator U+A980..U+A9DF)
 *     - Asersi positif inferensi nyata: `02_wedang_bubuk_crop.png` & `sample_4.png`
 *     - Kontras tema, tab Tentang (atribusi tim + CC BY 4.0 + SIL OFL 1.1), footer ringkas
 *     - Asersi `0` console/page error & `0` network request eksternal (hanya SAH setelah asersi positif)
 *  2) Mode Batch multi-halaman (`sample_1.png` + `sample_4.png` -> 2/2 selesai + ekspor aktif)
 *  3) Skenario Fallback Model (`aksara_embed.onnx` gagal muat -> `Stage-1 Fallback`, inferensi tetap berhasil)
 *  4) Skenario PWA Offline penuh (Service Worker cache `aksarascan-v*`, `setOffline(true)` + reload + inferensi sukses)
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const WEB_DIR = path.join(ROOT_DIR, 'app', 'web_camscanner');

const MIME_MAP = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.ttf': 'font/ttf',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.webmanifest': 'application/manifest+json',
};

function readSwCacheName() {
  const swText = fs.readFileSync(path.join(WEB_DIR, 'sw.js'), 'utf-8');
  const m = swText.match(/CACHE_NAME\s*=\s*['"]([^'"]+)['"]/);
  assert.ok(m, 'CACHE_NAME tidak ditemukan di sw.js');
  return m[1];
}

function startLocalServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = (req.url || '/').split('?')[0];
      const filePath = path.join(WEB_DIR, reqPath === '/' ? 'index.html' : reqPath.replace(/^\/+/, ''));
      if (!filePath.startsWith(WEB_DIR) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
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

function findChrome() {
  const candidates = [
    process.env.CHROME_EXE,
    path.join(process.env.LOCALAPPDATA || '', 'ms-playwright', 'chromium-1234', 'chrome-win64', 'chrome.exe'),
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  ].filter(Boolean);
  return candidates.find((c) => fs.existsSync(c));
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
  const t0 = performance.now();
  await page.locator('#fileInput').setInputFiles(imgPath);
  await page.waitForFunction(() => window.__e2eDone === true, null, { timeout: 60000 });
  await page.waitForTimeout(50);
  const latencyMs = Math.round(performance.now() - t0);

  const data = await page.evaluate(() => ({
    unicodeText: document.querySelector('#outJavanese')?.textContent || '',
    sentenceLatin: document.querySelector('#outSentenceLatin')?.textContent || '',
    sentenceIndonesian: document.querySelector('#outIndonesian')?.textContent || '',
    hasOrthoBadge: Boolean(document.querySelector('#v3OrthoBadge')),
    hasRawLabel: document.body.innerText.includes('Mentah detektor:'),
    mtChecked: document.querySelector('#chkOptInOnlineMT')?.checked ?? false,
    hasExportBtn: Boolean(document.querySelector('#btnExportV3Corrections')),
    btnEngineText: document.querySelector('#btnEngineYolo')?.textContent || '',
    workbenchVisible: !document.querySelector('#workbenchWrap')?.hidden,
    canvasW: document.querySelector('#mainCanvas')?.width || 0,
    overflowPx: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
  }));

  return { ...data, latencyMs };
}

async function runMatrixCell(browser, url, cell) {
  const consoleErrors = [];
  const externalReqs = [];
  const context = await browser.newContext({
    viewport: { width: cell.width, height: cell.height },
    serviceWorkers: 'block',
  });
  const page = await context.newPage();
  page.on('pageerror', (e) => consoleErrors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });

  await page.route('**/*', async (route) => {
    const rUrl = route.request().url();
    if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) {
      await route.continue();
    } else {
      externalReqs.push(rUrl);
      await route.abort();
    }
  });

  try {
    await page.goto(`${url}/index.html`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.querySelector('#engineStatus')?.textContent.includes('YOLOv8 Siap'),
      null,
      { timeout: 45000 }
    );
    await page.waitForFunction(
      () => window.AksaraStage2?.getStatus().ready === true,
      null,
      { timeout: 30000 }
    );

    if (cell.theme === 'light') {
      await page.click('#themeToggle');
      await page.waitForTimeout(80);
    }

    // 1. Asersi positif landing + validator ortografi sebelum inferensi
    const landing = await page.evaluate(() => {
      const o = window.AksaraOrthography;
      return {
        theme: document.documentElement.getAttribute('data-theme'),
        dropzoneVisible: !document.querySelector('#dropzone')?.hidden,
        workbenchHidden: Boolean(document.querySelector('#workbenchWrap')?.hidden),
        demoCards: document.querySelectorAll('.demo-card').length,
        hasCamBtn: Boolean(document.querySelector('#btnCam')),
        hasFileBtn: Boolean(document.querySelector('#btnFile')),
        overflowPx: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
        orthoValid: o.validateJavaneseUnicode('\uA9CB\uA9AE\uA9BA\uA9A2\uA981').valid,
        orthoInvalid: !o.validateJavaneseUnicode('\uA9A2\uA981\uA9B6').valid && !o.validateJavaneseUnicode('\uA98F\uA9BF\uA9BC').valid,
      };
    });
    assert.equal(landing.theme, cell.theme, `${cell.name}: tema aktif harus ${cell.theme}`);
    assert.equal(landing.dropzoneVisible, true, `${cell.name}: dropzone harus terlihat`);
    assert.equal(landing.workbenchHidden, true, `${cell.name}: workbench awal harus tersembunyi`);
    assert.ok(landing.demoCards >= 4, `${cell.name}: harus ada minimal 4 contoh uji`);
    assert.ok(landing.hasCamBtn && landing.hasFileBtn, `${cell.name}: tombol kamera & berkas harus ada`);
    assert.equal(landing.overflowPx, 0, `${cell.name}: landing horizontal overflow harus 0 px (got ${landing.overflowPx})`);
    assert.equal(landing.orthoValid && landing.orthoInvalid, true, `${cell.name}: validator U+A980..U+A9DF gagal`);

    // 2. Asersi positif inferensi nyata (02_wedang_bubuk_crop.png)
    const cropPath = path.join(ROOT_DIR, 'eval', 'failures', '02_wedang_bubuk_crop.png');
    const rCrop = await runUpload(page, cropPath);
    assert.equal(rCrop.workbenchVisible, true, `${cell.name}: workbench harus terbuka sesudah unggah`);
    assert.ok(rCrop.canvasW > 0, `${cell.name}: kanvas terisi`);
    assert.equal(rCrop.overflowPx, 0, `${cell.name}: workbench horizontal overflow harus 0 px`);
    assert.equal(rCrop.mtChecked, false, `${cell.name}: toggle MT daring baku harus MATI`);
    assert.equal(rCrop.hasExportBtn, true, `${cell.name}: tombol ekspor koreksi lokal harus tersedia`);
    assert.match(rCrop.btnEngineText, /55\s*Kelas/i, `${cell.name}: label harus 55 Kelas`);
    assert.equal(rCrop.hasOrthoBadge, false, `${cell.name}: badge usang tidak boleh ada`);
    assert.equal(rCrop.hasRawLabel, false, `${cell.name}: label mentah lama tidak boleh ada`);
    assert.ok(rCrop.latencyMs < 3500, `${cell.name}: latensi ${rCrop.latencyMs}ms harus < 3500ms`);
    assert.match(rCrop.sentenceLatin.toLowerCase(), /wédang|wedang/, `${cell.name}: harus memuat wédang`);
    assert.match(rCrop.sentenceLatin.toLowerCase(), /bubuk/, `${cell.name}: harus memuat bubuk`);
    assert.match(rCrop.sentenceLatin.toLowerCase(), /tanpa\s+gula/, `${cell.name}: harus memuat tanpa gula`);
    assert.match(rCrop.sentenceLatin.toLowerCase(), /ngamuk/, `${cell.name}: harus memuat ngamuk`);
    assert.match(rCrop.sentenceLatin.toLowerCase(), /mundhak\s+tuwa/, `${cell.name}: harus memuat mundhak tuwa`);
    assert.match(rCrop.sentenceIndonesian.toLowerCase(), /tanpa\s+gula/, `${cell.name}: terjemahan harus memuat tanpa gula`);

    // 3. Asersi positif sample_4.png
    const sample4Path = path.join(WEB_DIR, 'sample_4.png');
    const rS4 = await runUpload(page, sample4Path);
    assert.match(rS4.sentenceLatin.toLowerCase(), /aksara\s+jawa/, `${cell.name}: sample_4 harus memuat aksara jawa`);
    assert.match(rS4.sentenceLatin.toLowerCase(), /mangan\s+s[eê]ga/, `${cell.name}: sample_4 harus memuat mangan sêga`);
    assert.match(rS4.sentenceLatin.toLowerCase(), /krama/, `${cell.name}: sample_4 harus memuat krama`);

    // 4. Audit kontras kontrol aktif & Tab Tentang + atribusi CC BY / SIL OFL + footer
    await page.click('[data-nav="about"]');
    await page.waitForTimeout(60);
    const uiAudit = await page.evaluate(() => {
      const parseRgb = (s) => (s.match(/\d+/g) || [0, 0, 0]).slice(0, 3).map(Number);
      const lum = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const diffEl = (sel) => {
        const el = document.querySelector(sel);
        const st = getComputedStyle(el);
        return Math.abs(lum(parseRgb(st.color)) - lum(parseRgb(st.backgroundColor)));
      };
      return {
        navContrast: diffEl('.nav-pill.active'),
        engContrast: diffEl('.engine-btn.active'),
        aboutVisible: !document.querySelector('#viewAbout')?.hidden,
        aboutText: document.querySelector('#viewAbout')?.innerText || '',
        footerText: document.querySelector('.footer-minimal')?.innerText || '',
        overflowPx: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
      };
    });
    assert.ok(uiAudit.navContrast > 80, `${cell.name}: kontras nav aktif rendah (${uiAudit.navContrast})`);
    assert.ok(uiAudit.engContrast > 80, `${cell.name}: kontras tombol mesin rendah (${uiAudit.engContrast})`);
    assert.equal(uiAudit.aboutVisible, true, `${cell.name}: view Tentang harus terbuka`);
    assert.match(uiAudit.aboutText, /Muhammad Rayhan Najib/, `${cell.name}: nama penulis harus ada di Tentang`);
    assert.match(uiAudit.aboutText, /Mahfud Jiono/, `${cell.name}: nama dosen harus ada di Tentang`);
    assert.match(uiAudit.aboutText, /CC BY 4\.0/, `${cell.name}: atribusi CC BY 4.0 harus tampil di Tentang`);
    assert.match(uiAudit.aboutText, /SIL Open Font License/, `${cell.name}: lisensi SIL OFL harus tampil di Tentang`);
    assert.ok(!uiAudit.footerText.includes('240533600012'), `${cell.name}: footer harus tetap ringkas`);
    assert.equal(uiAudit.overflowPx, 0, `${cell.name}: overflow di tab Tentang harus 0`);

    // 5. Baru setelah asersi positif lulus: nol error & nol jaringan eksternal!
    assert.equal(consoleErrors.length, 0, `${cell.name}: console/page error ditemukan: ${consoleErrors.join('; ')}`);
    assert.equal(externalReqs.length, 0, `${cell.name}: permintaan jaringan eksternal ditemukan: ${externalReqs.join(', ')}`);

    console.log(`[PASS MATRIX] ${cell.name} | crop=${rCrop.latencyMs}ms | sample4=${rS4.latencyMs}ms | navDiff=${Math.round(uiAudit.navContrast)}`);
  } finally {
    await context.close();
  }
}

async function runBatchCheck(browser, url) {
  const consoleErrors = [];
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: 'block' });
  const page = await context.newPage();
  page.on('pageerror', e => consoleErrors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });

  await page.route('**/*', (route) => {
    const rUrl = route.request().url();
    if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) return route.continue();
    return route.abort();
  });

  try {
    await page.goto(`${url}/index.html`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.AksaraStage2?.getStatus().ready === true, null, { timeout: 45000 });
    await page.click('[data-nav="batch"]');
    await page.locator('#batchInput').setInputFiles([
      path.join(WEB_DIR, 'sample_1.png'),
      path.join(WEB_DIR, 'sample_4.png'),
    ]);
    await page.waitForFunction(
      () => (document.querySelector('#batchSummary')?.textContent || '').includes('2 selesai'),
      null,
      { timeout: 60000 }
    );
    const batchState = await page.evaluate(() => ({
      summary: document.querySelector('#batchSummary')?.textContent || '',
      items: document.querySelectorAll('#batchList .batch-item').length,
      copyDisabled: document.querySelector('#btnBatchCopyAll')?.disabled,
      txtDisabled: document.querySelector('#btnBatchTxt')?.disabled,
      mdDisabled: document.querySelector('#btnBatchMd')?.disabled,
      jsonDisabled: document.querySelector('#btnBatchJson')?.disabled,
    }));
    assert.equal(batchState.items, 2, 'Batch harus memproses 2 item');
    assert.equal(batchState.copyDisabled || batchState.txtDisabled || batchState.mdDisabled || batchState.jsonDisabled, false, 'Tombol ekspor batch harus aktif');
    assert.equal(consoleErrors.length, 0, `Batch error: ${consoleErrors.join('; ')}`);
    console.log('[PASS BATCH]', batchState.summary);
  } finally {
    await context.close();
  }
}

async function runFallbackAndOfflineChecks(browser, url, swCacheName) {
  // Fallback Stage-1 (model Stage-2 diblokir -> Stage-1 tetap membaca benar tanpa crash)
  const ctxFb = await browser.newContext({ viewport: { width: 1366, height: 900 }, serviceWorkers: 'block' });
  const pageFb = await ctxFb.newPage();
  const fbPageErrors = [];
  pageFb.on('pageerror', e => fbPageErrors.push(String(e)));

  await pageFb.route('**/*', (route) => {
    const rUrl = route.request().url();
    if (
      rUrl.endsWith('/aksara_embed.onnx') ||
      rUrl.endsWith('/reference_prototypes.json') ||
      rUrl.endsWith('/stage2_calibration.json')
    ) {
      return route.abort();
    }
    if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) return route.continue();
    return route.abort();
  });

  try {
    await pageFb.goto(`${url}/index.html`, { waitUntil: 'domcontentloaded' });
    await pageFb.waitForFunction(
      () => (document.querySelector('#engineStatus')?.textContent || '').includes('Stage-1 Fallback'),
      null,
      { timeout: 30000 }
    );
    const rFb = await runUpload(pageFb, path.join(WEB_DIR, 'sample_4.png'));
    assert.match(rFb.sentenceLatin.toLowerCase(), /aksara\s+jawa/, 'Fallback Stage-1 harus membaca aksara jawa');
    assert.match(rFb.sentenceLatin.toLowerCase(), /mangan\s+s[eê]ga/, 'Fallback Stage-1 harus membaca mangan sêga');
    assert.equal(fbPageErrors.length, 0, `Fallback pageerror: ${fbPageErrors.join('; ')}`);
    console.log('[PASS FALLBACK STAGE-1]', rFb.sentenceLatin);
  } finally {
    await ctxFb.close();
  }

  // Offline PWA Penuh
  const ctxOff = await browser.newContext({ viewport: { width: 1366, height: 900 } });
  const pageOff = await ctxOff.newPage();
  const offErrors = [];
  pageOff.on('pageerror', e => offErrors.push(String(e)));
  pageOff.on('console', m => { if (m.type() === 'error') offErrors.push(m.text()); });

  await pageOff.route('**/*', (route) => {
    const rUrl = route.request().url();
    if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) return route.continue();
    return route.abort();
  });

  try {
    await pageOff.goto(`${url}/index.html`, { waitUntil: 'load' });
    await pageOff.waitForFunction(
      () => (document.querySelector('#engineStatus')?.textContent || '').includes('Stage-2'),
      null,
      { timeout: 45000 }
    );
    const swReady = await pageOff.evaluate(async (expectedCache) => {
      if (!('serviceWorker' in navigator)) return { ok: false, reason: 'no SW' };
      const reg = await navigator.serviceWorker.ready;
      const cache = await caches.open(expectedCache);
      const keys = await cache.keys();
      return { ok: Boolean(reg?.active && keys.length >= 35), cacheName: expectedCache, cacheCount: keys.length };
    }, swCacheName);
    assert.ok(swReady.ok, `Cache ${swCacheName} harus terisi penuh (${JSON.stringify(swReady)})`);

    await ctxOff.setOffline(true);
    await pageOff.reload({ waitUntil: 'domcontentloaded' });
    await pageOff.waitForFunction(
      () => (document.querySelector('#engineStatus')?.textContent || '').includes('Stage-2'),
      null,
      { timeout: 45000 }
    );
    const rOff = await runUpload(pageOff, path.join(ROOT_DIR, 'eval', 'failures', '02_wedang_bubuk_crop.png'));
    assert.match(rOff.sentenceLatin.toLowerCase(), /wédang|wedang/, 'Offline PWA harus memulihkan wédang');
    assert.match(rOff.sentenceLatin.toLowerCase(), /tanpa\s+gula/, 'Offline PWA harus memulihkan tanpa gula');
    assert.equal(offErrors.length, 0, `Offline console error: ${offErrors.join('; ')}`);
    console.log('[PASS OFFLINE PWA]', { cache: swReady.cacheName, entries: swReady.cacheCount, latin: rOff.sentenceLatin });
  } finally {
    await ctxOff.close();
  }
}

async function main() {
  const swCacheName = readSwCacheName();
  const { server, url } = await startLocalServer();
  const browser = await chromium.launch({ headless: true, executablePath: findChrome() });

  try {
    const matrix = [
      { name: 'Desktop_1440_Dark', width: 1440, height: 900, theme: 'dark' },
      { name: 'Desktop_1440_Light', width: 1440, height: 900, theme: 'light' },
      { name: 'Mobile_390_Dark', width: 390, height: 844, theme: 'dark' },
      { name: 'Mobile_390_Light', width: 390, height: 844, theme: 'light' },
    ];
    for (const cell of matrix) {
      await runMatrixCell(browser, url, cell);
    }
    await runBatchCheck(browser, url);
    await runFallbackAndOfflineChecks(browser, url, swCacheName);
    console.log('\n[PASS SEMUA E2E FASE 5] 4 matriks + batch + fallback Stage-1 + offline PWA lolos.');
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((err) => {
  console.error('[E2E GAGAL]', err);
  process.exit(1);
});
