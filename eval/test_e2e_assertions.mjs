/**
 * Uji End-to-End Berbasis Asersi Positif AksaraScan v3.0 (Playwright + Chromium Offline).
 * Fase 5a / 5c — memverifikasi:
 *   1) Pemuatan aplikasi web penuh (`index.html` + `javanese_lm_data.js` + `javanese_lm_engine.js` + `orthography.js` + `v3_ui_enhancer.js`)
 *   2) Transkripsi tepat & pemulihan Dekoder Beam pada kasus kegagalan nyata (`Wédang bubuk tanpa gula, aja ngamuk mundhak tuwa.`)
 *   3) Kestabilan deteksi + pasangan/sandhangan pada sampel bawaan (`sample_4.png` -> aksara/jawa/mangan/sega/krama/kothi)
 *   4) Uji positif & negatif validator ortografi Unicode Jawa (`validateJavaneseUnicode`)
 *   5) Keberadaan kontrol UI v3.0 (transkripsi mentah `#outRawJavanese`, tombol ekspor koreksi, toggle MT daring default MATI)
 *   6) Nol console/page error, nol permintaan jaringan eksternal selama inferensi, dan latensi per baris < 2000 ms.
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
const ORT_DIST = path.join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');

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

function startLocalServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = (req.url || '/').split('?')[0];
      const filePath = reqPath.startsWith('/__ort__/')
        ? path.join(ORT_DIST, reqPath.replace('/__ort__/', ''))
        : path.join(WEB_DIR, reqPath === '/' ? 'index.html' : reqPath.replace(/^\/+/, ''));
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
    mtChecked: document.querySelector('#chkOptInOnlineMT')?.checked,
    hasExportBtn: Boolean(document.querySelector('#btnExportV3Corrections')),
    btnEngineText: document.querySelector('#btnEngineYolo')?.textContent || '',
  }));
  return { ...data, latencyMs };
}

async function main() {
  const { server, url } = await startLocalServer();
  const browser = await chromium.launch({ headless: true, executablePath: findChrome() });
  const consoleErrors = [];
  const externalInfReqs = [];
  let inInference = false;

  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    page.on('pageerror', (e) => consoleErrors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });

    await page.route('**/*', async (route) => {
      const rUrl = route.request().url();
      if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) {
        await route.continue();
      } else if (rUrl.startsWith('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/')) {
        const localFile = path.join(ORT_DIST, path.basename(rUrl.split('?')[0]));
        if (fs.existsSync(localFile)) {
          await route.fulfill({
            status: 200,
            contentType: MIME_MAP[path.extname(localFile).toLowerCase()] || 'application/octet-stream',
            body: fs.readFileSync(localFile),
          });
        } else {
          await route.abort();
        }
      } else {
        if (inInference) externalInfReqs.push(rUrl);
        if (rUrl.includes('fonts.googleapis.com')) await route.fulfill({ status: 200, contentType: 'text/css', body: '' });
        else await route.abort();
      }
    });

    await page.goto(`${url}/index.html`, { waitUntil: 'domcontentloaded' });
    try {
      await page.waitForFunction(
        () => document.querySelector('#engineStatus')?.textContent.includes('YOLOv8 Siap'),
        null,
        { timeout: 45000 },
      );
    } catch (waitErr) {
      const statusText = await page.evaluate(() => document.querySelector('#engineStatus')?.textContent || '<none>');
      throw new Error(`${waitErr.message} | #engineStatus=${statusText} | consoleErrors=${JSON.stringify(consoleErrors)}`);
    }
    inInference = true;

    // 1. Asersi positif/negatif validator ortografi di dalam browser
    const orthoChecks = await page.evaluate(() => {
      const o = window.AksaraOrthography;
      return {
        validWedang: o.validateJavaneseUnicode('\uA9CB\uA9AE\uA9BA\uA9A2\uA981').valid,
        invalidReversed: o.validateJavaneseUnicode('\uA9A2\uA981\uA9B6').valid,
        invalidCakraPepet: o.validateJavaneseUnicode('\uA98F\uA9BF\uA9BC').valid,
        invalidKeretSwara: o.validateJavaneseUnicode('\uA98F\uA9BD\uA9B6').valid,
        invalidPangkonSwara: o.validateJavaneseUnicode('\uA98F\uA9C0\uA9B6').valid,
      };
    });
    assert.equal(orthoChecks.validWedang, true, 'Validator harus menerima kluster wédang kanonik');
    assert.equal(orthoChecks.invalidReversed, false, 'Validator harus menolak urutan terbalik panyigeg+wulu');
    assert.equal(orthoChecks.invalidCakraPepet, false, 'Validator harus menolak cakra+pepet');
    assert.equal(orthoChecks.invalidKeretSwara, false, 'Validator harus menolak keret+swara');
    assert.equal(orthoChecks.invalidPangkonSwara, false, 'Validator harus menolak swara setelah pangkon');
    console.log('[PASS] Asersi positif & negatif validator ortografi U+A980..U+A9DF di Chromium');

    // 2. Uji gambar gagal utama user (02_wedang_bubuk_crop.png) & gambar penuh (01_wedang_bubuk_full.png)
    const cropPath = path.join(ROOT_DIR, 'eval', 'failures', '02_wedang_bubuk_crop.png');
    const rCrop = await runUpload(page, cropPath);
    console.log('[diag_02 crop]', rCrop);
    assert.equal(rCrop.mtChecked, false, 'Toggle MT daring harus MATI secara baku');
    assert.equal(rCrop.hasExportBtn, true, 'Tombol ekspor koreksi lokal harus tersedia');
    assert.match(rCrop.btnEngineText, /55\s*Kelas/i, 'Label UI harus menyebut 55 Kelas');
    assert.equal(rCrop.hasOrthoBadge, false, 'Badge Ortografi Valid harus sudah dihapus');
    assert.equal(rCrop.hasRawLabel, false, 'Label Mentah detektor harus sudah dihapus');
    assert.ok(rCrop.latencyMs < 2000, `Latensi baris (${rCrop.latencyMs} ms) harus < 2000 ms`);
    assert.match(rCrop.sentenceLatin.toLowerCase(), /wédang|wedang/, 'Beam decoder harus memulihkan kata "wédang"');
    assert.match(rCrop.sentenceLatin.toLowerCase(), /bubuk/, 'Harus terbaca "bubuk"');
    assert.match(rCrop.sentenceLatin.toLowerCase(), /tanpa\s+gula/, 'Harus terbaca "tanpa gula"');
    assert.match(rCrop.sentenceLatin.toLowerCase(), /ngamuk/, 'Harus terbaca "ngamuk"');
    assert.match(rCrop.sentenceLatin.toLowerCase(), /mundhak\s+tuwa/, 'Harus terbaca "mundhak tuwa"');
    assert.match(rCrop.sentenceIndonesian.toLowerCase(), /tanpa\s+gula/, 'Terjemahan Indonesia harus memuat "tanpa gula"');
    console.log('[PASS] Asersi positif 02_wedang_bubuk_crop.png:', rCrop.sentenceLatin);

    // 3. Uji sampel 3 baris pasangan/sandhangan (sample_4.png)
    const sample4Path = path.join(ROOT_DIR, 'app', 'web_camscanner', 'sample_4.png');
    const rS4 = await runUpload(page, sample4Path);
    console.log('[sample_4]', rS4);
    assert.match(rS4.sentenceLatin.toLowerCase(), /aksara\s+jawa/, 'Sample 4 harus mengandung "aksara jawa"');
    assert.match(rS4.sentenceLatin.toLowerCase(), /mangan\s+s[eê]ga/, 'Sample 4 harus mengandung "mangan sêga"');
    assert.match(rS4.sentenceLatin.toLowerCase(), /krama/, 'Sample 4 harus mengandung "krama"');
    console.log('[PASS] Asersi positif sample_4.png:', rS4.sentenceLatin);

    // 4. Uji konsistensi kontras tema Gelap -> Terang & navigasi tab baru
    const lumDiff = async () => page.evaluate(() => {
      const parseRgb = (s) => (s.match(/\d+/g) || [0, 0, 0]).slice(0, 3).map(Number);
      const lum = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const checkEl = (sel) => {
        const el = document.querySelector(sel);
        const st = getComputedStyle(el);
        return Math.abs(lum(parseRgb(st.color)) - lum(parseRgb(st.backgroundColor)));
      };
      return {
        theme: document.documentElement.getAttribute('data-theme'),
        navContrast: checkEl('.nav-pill.active'),
        engContrast: checkEl('.engine-btn.active'),
        titleColor: getComputedStyle(document.querySelector('.page-title')).color,
        bodyBg: getComputedStyle(document.body).backgroundColor,
      };
    });
    const darkMetrics = await lumDiff();
    assert.ok(darkMetrics.navContrast > 80, `Kontras nav aktif gelap (${darkMetrics.navContrast}) harus tinggi`);
    assert.ok(darkMetrics.engContrast > 80, `Kontras mesin aktif gelap (${darkMetrics.engContrast}) harus tinggi`);

    await page.click('#themeToggle');
    await page.waitForTimeout(100);
    const lightMetrics = await lumDiff();
    assert.equal(lightMetrics.theme, 'light', 'Tema harus berpindah ke light');
    assert.ok(lightMetrics.navContrast > 80, `Kontras nav aktif terang (${lightMetrics.navContrast}) harus tinggi (tidak ketutup bg)`);
    assert.ok(lightMetrics.engContrast > 80, `Kontras mesin aktif terang (${lightMetrics.engContrast}) harus tinggi`);
    assert.notEqual(darkMetrics.titleColor, lightMetrics.titleColor, 'Warna font judul harus berubah saat berganti tema');
    console.log('[PASS] Kontras tema Gelap & Terang konsisten:', { darkNav: Math.round(darkMetrics.navContrast), lightNav: Math.round(lightMetrics.navContrast) });

    // 5. Uji navigasi 4 tab & footer ringkas
    await page.click('[data-nav="about"]');
    await page.waitForTimeout(80);
    const aboutCheck = await page.evaluate(() => ({
      aboutVisible: !document.querySelector('#viewAbout').hidden,
      aboutText: document.querySelector('#viewAbout').innerText,
      footerText: document.querySelector('.footer-minimal').innerText,
    }));
    assert.equal(aboutCheck.aboutVisible, true, 'Tab Tentang harus terbuka');
    assert.match(aboutCheck.aboutText, /Muhammad Rayhan Najib/, 'Nama anggota harus tampil di tab Tentang');
    assert.match(aboutCheck.aboutText, /Mahfud Jiono/, 'Nama dosen harus tampil di tab Tentang');
    assert.ok(!aboutCheck.footerText.includes('240533600012'), 'Footer ringkas tidak boleh memuat daftar NIM panjang');
    console.log('[PASS] Navigasi tab & pemindahan kredit tim ke tab Tentang terverifikasi.');

    // 6. Verifikasi nol error & nol permintaan eksternal
    assert.equal(consoleErrors.length, 0, `Tidak boleh ada console/page error: ${consoleErrors.join('; ')}`);
    assert.equal(externalInfReqs.length, 0, `Tidak boleh ada network request eksternal saat inferensi: ${externalInfReqs.join(', ')}`);
    console.log('[PASS] 0 console error, 0 network request eksternal saat inferensi.');
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((err) => {
  console.error('[E2E GAGAL]', err);
  process.exit(1);
});
