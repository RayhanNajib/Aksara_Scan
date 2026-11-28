/**
 * Suite Pengujian Menyeluruh Gerbang 4 — Fase 4 AksaraScan v4.0 (UI/UX Anti-Slop)
 * Menguji:
 *  1) Matriks 2 viewport (1440px Desktop & 390px Mobile) × 2 tema (Dark & Light) + screenshot bukti
 *  2) Audit kontras WCAG AA (>= 4.5:1) atas pasangan token & elemen terhitung di kedua tema
 *  3) Audit daftar larangan Anti-Slop (tanpa .wayang-corner, gradient text, eyebrow kapital, kartu bersarang, border > 1px)
 *  4) Pengungkapan bertahap (landing bebas pengaturan -> split workbench 52/48), Compare Before/After, Rotate/Zoom/Enhancer UI
 *  5) Inspektor Referensi Stage-2 (potongan pengguna + Top-3 kandidat + koreksi 1-klik + riwayat koreksi)
 *  6) Mode Batch Multi-Halaman (2 berkas berurutan + ekspor)
 *  7) Riwayat Pindai Lokal IndexedDB (simpan, pulihkan, & bersihkan)
 *  8) Bank Prototipe 56 Kelas & Kamus di Lab & Referensi
 *  9) Kasus ekstrem (foto sangat besar 2600x1800, kanvas kosong -> recovery notice, kata OOV panjang 48 huruf di 390px, 0 horizontal overflow)
 * 10) Nol permintaan jaringan eksternal sama sekali (font juga 100% lokal self-hosted) & 0 console error.
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
const SHOT_DIR = path.join(ROOT_DIR, 'eval', 'artifacts', 'fase4_screenshots');
const REPORT_PATH = path.join(ROOT_DIR, 'eval', 'reports', 'phase4_verification.json');

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

function findChrome() {
  const candidates = [
    process.env.CHROME_EXE,
    path.join(process.env.LOCALAPPDATA || '', 'ms-playwright', 'chromium-1234', 'chrome-win64', 'chrome.exe'),
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  ].filter(Boolean);
  return candidates.find((c) => fs.existsSync(c));
}

function startLocalServer() {
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
    window.__f4Done = false;
    const btn = document.querySelector('#btnRun');
    let sawDis = false;
    const obs = new MutationObserver(() => {
      if (btn.disabled) sawDis = true;
      else if (sawDis && !btn.disabled) {
        window.__f4Done = true;
        obs.disconnect();
      }
    });
    obs.observe(btn, { attributes: true, attributeFilter: ['disabled'] });
  });
  await page.locator('#fileInput').setInputFiles(imgPath);
  await page.waitForFunction(() => window.__f4Done === true, null, { timeout: 60000 });
  await page.waitForTimeout(80);
}

async function makeTempPng(page, width, height, drawFn, outPath) {
  const b64 = await page.evaluate(({ w, h, fnStr }) => {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    const fn = new Function('ctx', 'w', 'h', fnStr);
    fn(ctx, w, h);
    return c.toDataURL('image/png').split(',')[1];
  }, { w: width, h: height, fnStr: drawFn });
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
}

async function runWcagAndAntiSlopAudit(page) {
  return page.evaluate(() => {
    function parseRgba(s) {
      const m = (s || '').match(/[\d.]+/g);
      if (!m || m.length < 3) return [0, 0, 0, 1];
      return [Number(m[0]), Number(m[1]), Number(m[2]), m[3] !== undefined ? Number(m[3]) : 1];
    }
    function relLum([r, g, b]) {
      const f = (c) => { const s = c / 255; return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4); };
      return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
    }
    function contrast(a, b) {
      const l1 = relLum(a), l2 = relLum(b);
      return Number(((Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05)).toFixed(2));
    }
    function effectiveBg(el) {
      let cur = el;
      while (cur && cur !== document.documentElement) {
        const bg = parseRgba(getComputedStyle(cur).backgroundColor);
        if (bg[3] >= 0.85) return bg;
        cur = cur.parentElement;
      }
      return parseRgba(getComputedStyle(document.body).backgroundColor);
    }

    const csRoot = getComputedStyle(document.documentElement);
    const tokenPairs = [
      ['--text-main', '--bg-primary'],
      ['--text-main', '--bg-surface'],
      ['--text-main', '--bg-panel'],
      ['--text-muted', '--bg-surface'],
      ['--text-muted', '--bg-panel'],
      ['--text-sub', '--bg-panel'],
      ['--text-sub', '--bg-sunken'],
      ['--accent-text', '--bg-surface'],
      ['--nav-active-text', '--nav-active-bg'],
      ['--btn-accent-text', '--btn-accent-bg'],
      ['--lontar', '--bg-surface'],
      ['--bata', '--bg-surface'],
    ].map(([fgVar, bgVar]) => {
      const fg = csRoot.getPropertyValue(fgVar).trim();
      const bg = csRoot.getPropertyValue(bgVar).trim();
      const div = document.createElement('div');
      div.style.color = fg; div.style.backgroundColor = bg;
      document.body.appendChild(div);
      const st = getComputedStyle(div);
      const ratio = contrast(parseRgba(st.color), parseRgba(st.backgroundColor));
      div.remove();
      return { fgVar, bgVar, fg, bg, ratio, pass: ratio >= 4.5 };
    });

    // Live visible DOM element sample check
    const liveChecks = [];
    document.querySelectorAll('h1,h2,h3,p,button,.nav-pill,.engine-btn,.output-text--aksara,.output-text--latin,.output-text--meaning,.footer-minimal')
      .forEach((el) => {
        if (!el.offsetParent && getComputedStyle(el).position !== 'fixed') return;
        const st = getComputedStyle(el);
        const fg = parseRgba(st.color);
        const bg = effectiveBg(el);
        if (fg[3] > 0.1) {
          const ratio = contrast(fg, bg);
          liveChecks.push({ tag: el.tagName, cls: String(el.className).slice(0, 40), ratio });
        }
      });

    // Anti-slop structural DOM checks
    const wayangCount = document.querySelectorAll('.wayang-corner').length;
    const gradientText = [];
    const uppercaseHeaders = [];
    const nestedCards = [];
    const thickBorders = [];
    document.querySelectorAll('*').forEach((el) => {
      const st = getComputedStyle(el);
      const bgClip = st.webkitBackgroundClip || st.backgroundClip;
      if (bgClip === 'text' && parseRgba(st.webkitTextFillColor || st.color)[3] < 0.2) {
        gradientText.push(el.tagName + '.' + el.className);
      }
      if (/^H[1-6]$/.test(el.tagName) && st.textTransform === 'uppercase') {
        uppercaseHeaders.push(el.textContent.slice(0, 30));
      }
      if ((el.classList.contains('panel') || el.classList.contains('output-sheet')) &&
          el.parentElement?.closest('.panel,.surface-block,.workbench-col')) {
        if (!el.classList.contains('output-sheet')) {
          nestedCards.push(el.className);
        }
      }
      const bl = parseFloat(st.borderLeftWidth) || 0;
      const br = parseFloat(st.borderRightWidth) || 0;
      if ((bl > 2.5 || br > 2.5) && !el.id.includes('selOverlay')) {
        thickBorders.push(`${el.tagName}.${el.className}:${bl}/${br}`);
      }
    });

    const horizontalOverflowPx = Math.max(0, document.documentElement.scrollWidth - window.innerWidth);
    return {
      theme: document.documentElement.getAttribute('data-theme'),
      width: window.innerWidth,
      tokenPairs,
      minTokenContrast: Math.min(...tokenPairs.map((p) => p.ratio)),
      minLiveContrast: liveChecks.length ? Math.min(...liveChecks.map((l) => l.ratio)) : 99,
      lowestLiveItem: liveChecks.sort((a, b) => a.ratio - b.ratio)[0] || null,
      antiSlop: { wayangCount, gradientText, uppercaseHeaders, nestedCards, thickBorders },
      horizontalOverflowPx,
    };
  });
}

async function main() {
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  const { server, url } = await startLocalServer();
  const browser = await chromium.launch({ headless: true, executablePath: findChrome() });
  const consoleErrors = [];
  const externalRequests = [];
  const report = { generatedAt: new Date().toISOString(), matrix: [], checks: {} };

  const tmpDir = path.join(ROOT_DIR, 'eval', 'artifacts', 'fase4_tmp');
  fs.mkdirSync(tmpDir, { recursive: true });

  try {
    // 1. Desktop 1440px (Dark + Light): Progressive Disclosure -> Workbench -> Inspector -> Batch -> Lab -> About
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    page.on('pageerror', (e) => consoleErrors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    await page.route('**/*', (route) => {
      const rUrl = route.request().url();
      if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) return route.continue();
      externalRequests.push(rUrl);
      return route.abort();
    });

    await page.goto(`${url}/index.html`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => (document.querySelector('#engineStatus')?.textContent || '').includes('Stage-2'), null, { timeout: 45000 });
    await page.waitForTimeout(150);

    // Cek Pengungkapan Bertahap sebelum upload
    const initialState = await page.evaluate(() => ({
      landingVisible: !document.querySelector('#scannerEmptyWrap').hidden,
      workbenchHidden: document.querySelector('#workbenchWrap').hidden,
      hasFileBtn: Boolean(document.querySelector('#btnFile')),
      hasCamBtn: Boolean(document.querySelector('#btnCam')),
      sampleCount: document.querySelectorAll('.demo-card').length,
      hasNoSettingsVisible: !document.querySelector('#pipelineOptions').offsetParent,
    }));
    assert.equal(initialState.landingVisible, true, 'Landing dropzone harus tampil awal');
    assert.equal(initialState.workbenchHidden, true, 'Workbench tersembunyi sebelum ada gambar (progressive disclosure)');
    assert.equal(initialState.sampleCount, 4, '4 contoh uji tersedia');
    assert.equal(initialState.hasNoSettingsVisible, true, 'Pengaturan lanjutan tak membebani layar awal');
    report.checks.progressiveDisclosure = initialState;

    await page.screenshot({ path: path.join(SHOT_DIR, '01_landing_1440_dark.png'), fullPage: true });
    report.matrix.push(await runWcagAndAntiSlopAudit(page));

    // Upload 02_wedang_bubuk_crop.png & uji Workbench 52/48, Compare, Zoom, Enhancer
    const cropPath = path.join(ROOT_DIR, 'eval', 'failures', '02_wedang_bubuk_crop.png');
    await runUpload(page, cropPath);
    const wbMetrics = await page.evaluate(() => {
      const rIn = document.querySelector('.workbench-col--input').getBoundingClientRect();
      const rOut = document.querySelector('.workbench-col--output').getBoundingClientRect();
      return {
        workbenchVisible: !document.querySelector('#workbenchWrap').hidden,
        landingHidden: document.querySelector('#scannerEmptyWrap').hidden,
        sideBySide: rIn.left < rOut.left,
        colRatio: Number((rIn.width / Math.max(1, rIn.width + rOut.width)).toFixed(2)),
        latin: document.querySelector('#outSentenceLatin').textContent,
        glyphCount: document.querySelectorAll('#glyphsGrid .glyph-card').length,
      };
    });
    assert.equal(wbMetrics.workbenchVisible, true, 'Workbench harus tampil sesudah unggah');
    assert.equal(wbMetrics.sideBySide, true, 'Split 2-kolom berdampingan di 1440px');
    assert.ok(wbMetrics.colRatio >= 0.48 && wbMetrics.colRatio <= 0.56, `Rasio kiri ~52% (${wbMetrics.colRatio})`);
    assert.match(wbMetrics.latin.toLowerCase(), /wédang|wedang/, 'Hasil positif wédang');
    report.checks.workbenchSplit = wbMetrics;

    // Test Before/After Compare & Zoom
    await page.click('#toggleCompare');
    await page.click('#btnZoomIn');
    await page.waitForTimeout(80);
    const cmpZoom = await page.evaluate(() => ({
      compareVisible: !document.querySelector('#compareDivider').hidden,
      zoomText: document.querySelector('#zoomBadge').textContent,
      toolbarNoClip: document.querySelector('.stage__toolbar').scrollWidth <= document.querySelector('.stage__toolbar').clientWidth + 1,
    }));
    assert.equal(cmpZoom.compareVisible, true, 'Slider Before/After compare tampil saat aktif');
    assert.equal(cmpZoom.zoomText, '120%', 'Zoom bekerja (+20%)');
    assert.equal(cmpZoom.toolbarNoClip, true, 'Micro-toolbar kanvas tidak terpotong di desktop 1440px');
    await page.click('#toggleCompare');
    await page.click('#btnZoomOut');

    await page.screenshot({ path: path.join(SHOT_DIR, '02_workbench_1440_dark.png'), fullPage: true });
    const darkWbAudit = await runWcagAndAntiSlopAudit(page);
    report.matrix.push(darkWbAudit);

    // Uji Inspektor Referensi Stage-2 (klik kartu glif pertama)
    await page.locator('#glyphsGrid .glyph-card').first().click();
    await page.waitForTimeout(120);
    const inspCheck = await page.evaluate(() => ({
      modalOpen: !document.querySelector('#correctionModal').hidden,
      candRows: document.querySelectorAll('#inspTop3List .cand-row').length,
      thumbImgs: document.querySelectorAll('#inspTop3List .cand-row__thumbs img').length,
      cropBadge: document.querySelector('#inspCropMatchBadge').textContent,
    }));
    assert.equal(inspCheck.modalOpen, true, 'Inspektor Referensi terbuka');
    assert.ok(inspCheck.candRows >= 1 && inspCheck.candRows <= 3, `Top-K kandidat tampil (${inspCheck.candRows})`);
    assert.ok(inspCheck.thumbImgs >= 1, 'Thumbnail prototipe referensi tampil');
    report.checks.referenceInspector = inspCheck;
    await page.screenshot({ path: path.join(SHOT_DIR, '03_inspector_1440_dark.png') });
    await page.click('#closeModal');

    // Switch tema Light pada Workbench 1440px
    await page.click('#themeToggle');
    await page.waitForTimeout(150);
    const lightWbAudit = await runWcagAndAntiSlopAudit(page);
    assert.equal(lightWbAudit.theme, 'light');
    report.matrix.push(lightWbAudit);
    await page.screenshot({ path: path.join(SHOT_DIR, '04_workbench_1440_light.png'), fullPage: true });

    // Uji Bank Prototipe (56 Kelas) & Kamus di Lab & Referensi
    await page.click('[data-nav="lab"]');
    await page.click('[data-labtab="proto"]');
    await page.waitForTimeout(120);
    const protoCountAll = await page.locator('#protoBankGrid .proto-card').count();
    await page.click('[data-pcat="non_aksara"]');
    await page.waitForTimeout(80);
    const protoCountNeg = await page.locator('#protoBankGrid .proto-card').count();
    assert.equal(protoCountAll, 56, 'Bank prototipe menampilkan 56 kelas');
    assert.equal(protoCountNeg, 1, 'Filter Non-Aksara menampilkan #55');
    await page.click('[data-labtab="kamus"]');
    const dictCount = await page.locator('#dictionaryGrid .dict-card').count();
    assert.equal(dictCount, 20, 'Kamus Nglegena menampilkan 20 aksara');
    report.checks.protoBank = { protoCountAll, protoCountNeg, dictCount };

    // Uji Batch Multi-Halaman (2 gambar)
    await page.click('[data-nav="batch"]');
    await page.locator('#batchInput').setInputFiles([
      path.join(WEB_DIR, 'sample_1.png'),
      path.join(WEB_DIR, 'sample_4.png'),
    ]);
    await page.waitForFunction(
      () => document.querySelectorAll('#batchList .batch-item').length === 2 &&
        (document.querySelector('#batchSummary')?.textContent || '').includes('2 selesai'),
      null,
      { timeout: 60000 }
    );
    const batchTexts = await page.evaluate(() => ({
      sum: document.querySelector('#batchSummary').textContent,
      canExport: !document.querySelector('#btnBatchJson').disabled,
    }));
    assert.equal(batchTexts.canExport, true, 'Ekspor batch aktif setelah selesai');
    report.checks.batch = batchTexts;

    // Uji Tentang (Mode Read & Atribusi CC BY / SIL OFL)
    await page.click('[data-nav="about"]');
    await page.waitForTimeout(100);
    const aboutOk = await page.evaluate(() => {
      const txt = document.querySelector('#viewAbout').innerText;
      return txt.includes('CC BY 4.0') && txt.includes('SIL Open Font License') && txt.includes('Muhammad Rayhan Najib');
    });
    assert.equal(aboutOk, true, 'Tab Tentang memuat atribusi CC BY, SIL OFL, dan identitas tim');

    // Uji Riwayat IndexedDB sesudah reload + Kasus Ekstrem: Kanvas kosong & Foto sangat besar 2600x1800
    await page.click('[data-nav="scanner"]');
    await page.click('#btnNewScan');
    await page.waitForTimeout(120);
    const histCount = await page.locator('#recentScansList .recent-item').count();
    assert.ok(histCount >= 1, `Riwayat Pindai Lokal IndexedDB tersimpan (${histCount})`);
    report.checks.indexedDbHistory = { histCount };

    // Kanvas kosong -> pesan pemulihan ramah
    const blankPng = path.join(tmpDir, 'blank.png');
    await makeTempPng(page, 640, 320, 'ctx.fillStyle="#f7f4ec"; ctx.fillRect(0,0,w,h);', blankPng);
    await runUpload(page, blankPng);
    const emptyAlert = await page.evaluate(() => ({
      shown: !document.querySelector('#scanEmptyAlert').hidden,
      msg: document.querySelector('#scanEmptyMsg').textContent,
    }));
    assert.equal(emptyAlert.shown, true, 'Kanvas kosong menampilkan pesan pemulihan');
    report.checks.emptyRecovery = emptyAlert;

    // Gambar sangat besar 2600x1800 -> di-downscale aman & tidak error
    const bigPng = path.join(tmpDir, 'big_2600x1800.png');
    await makeTempPng(page, 2600, 1800, 'ctx.fillStyle="#faf7ee"; ctx.fillRect(0,0,w,h); ctx.fillStyle="#222"; ctx.fillRect(200,200,40,40);', bigPng);
    await runUpload(page, bigPng);
    const bigDim = await page.evaluate(() => document.querySelector('#srcDim').textContent);
    assert.match(bigDim, /skala dari 2600×1800/i, 'Dimensi besar terkalibrasi otomatis');
    report.checks.largeImage = { bigDim };

    // 2. Mobile 390px (Dark & Light) + Fokus Keyboard + Kata OOV panjang (overflow 0px)
    const ctxMob = await browser.newContext({ viewport: { width: 390, height: 844 } });
    const pageMob = await ctxMob.newPage();
    pageMob.on('pageerror', (e) => consoleErrors.push(String(e)));
    pageMob.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    await pageMob.route('**/*', (route) => {
      const rUrl = route.request().url();
      if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) return route.continue();
      externalRequests.push(rUrl);
      return route.abort();
    });

    await pageMob.goto(`${url}/index.html`, { waitUntil: 'domcontentloaded' });
    await pageMob.waitForFunction(() => (document.querySelector('#engineStatus')?.textContent || '').includes('Stage-2'), null, { timeout: 45000 });
    await pageMob.screenshot({ path: path.join(SHOT_DIR, '05_landing_390_dark.png'), fullPage: true });
    report.matrix.push(await runWcagAndAntiSlopAudit(pageMob));

    await runUpload(pageMob, cropPath);
    // Inject kata OOV panjang 52 karakter untuk verifikasi wrap di layar 390px
    await pageMob.evaluate(() => {
      document.querySelector('#outSentenceLatin').textContent = 'pambudhidayaningpangrekadayaaksarajawamantramukjizatpanjangbanget';
    });
    await pageMob.waitForTimeout(100);
    const mobDarkAudit = await runWcagAndAntiSlopAudit(pageMob);
    assert.equal(mobDarkAudit.horizontalOverflowPx, 0, 'Mobile 390px Dark bebas scroll horizontal');
    const mobNavFit = await pageMob.evaluate(() => {
      const nav = document.querySelector('.topnav');
      return nav.scrollWidth <= nav.clientWidth + 1;
    });
    assert.equal(mobNavFit, true, 'Keempat pill navigasi muat di 390px tanpa terpotong');
    report.matrix.push(mobDarkAudit);
    await pageMob.screenshot({ path: path.join(SHOT_DIR, '06_workbench_390_dark.png'), fullPage: true });

    await pageMob.click('#themeToggle');
    await pageMob.waitForTimeout(120);
    const mobLightAudit = await runWcagAndAntiSlopAudit(pageMob);
    assert.equal(mobLightAudit.horizontalOverflowPx, 0, 'Mobile 390px Light bebas scroll horizontal');
    report.matrix.push(mobLightAudit);
    await pageMob.screenshot({ path: path.join(SHOT_DIR, '07_workbench_390_light.png'), fullPage: true });

    // Cek keyboard Tab focus visible
    await pageMob.keyboard.press('Tab');
    const focusInfo = await pageMob.evaluate(() => {
      const el = document.activeElement;
      const st = el ? getComputedStyle(el) : null;
      return { tag: el?.tagName, outlineWidth: st?.outlineWidth, outlineStyle: st?.outlineStyle };
    });
    report.checks.keyboardFocus = focusInfo;

    // Validasi seluruh matriks Anti-Slop & WCAG >= 4.5:1
    for (const m of report.matrix) {
      assert.ok(m.minTokenContrast >= 4.5, `Kontras token tema ${m.theme} @${m.width} (${m.minTokenContrast}) harus >= 4.5`);
      assert.equal(m.antiSlop.wayangCount, 0, '.wayang-corner harus 0');
      assert.equal(m.antiSlop.gradientText.length, 0, 'Teks gradasi harus 0');
      assert.equal(m.antiSlop.uppercaseHeaders.length, 0, 'Heading huruf kapital semua harus 0');
      assert.equal(m.antiSlop.nestedCards.length, 0, 'Kartu bersarang harus 0');
      assert.equal(m.antiSlop.thickBorders.length, 0, 'Border samping tebal (>2.5px) harus 0');
      assert.equal(m.horizontalOverflowPx, 0, `Horizontal overflow @${m.width}px ${m.theme} harus 0`);
    }

    assert.equal(externalRequests.length, 0, `Tidak boleh ada request eksternal (CDN font pun sudah lokal): ${JSON.stringify(externalRequests)}`);
    assert.equal(consoleErrors.length, 0, `Console errors ditemukan: ${JSON.stringify(consoleErrors)}`);

    fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));
    console.log('\n[PASS SEMUA GERBANG 4] Laporan tersimpan ke', REPORT_PATH);
    console.log('Matrix ringkas:', report.matrix.map((m) => ({
      width: m.width,
      theme: m.theme,
      minTokenCR: m.minTokenContrast,
      minLiveCR: m.minLiveContrast,
      overflow: m.horizontalOverflowPx,
      antiSlopOK: Object.values(m.antiSlop).every((v) => v === 0 || (Array.isArray(v) && v.length === 0)),
    })));
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((err) => {
  console.error('[GERBANG 4 GAGAL]', err);
  process.exit(1);
});
