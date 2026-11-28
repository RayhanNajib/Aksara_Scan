/**
 * Audit Pra-Rilis Vercel, Ukuran Aset, Cache Service Worker, & Waktu Muat Pertama
 * -------------------------------------------------------------------------------
 * Menguji kesiapan deploy statis sesuai `vercel.json` di localhost dengan peramban
 * Chromium baru (tanpa cache), menghitung ukuran total aset produksi, memverifikasi
 * Service Worker cache-version & asset-list, serta mengukur cold first-load latency.
 *
 * Penggunaan:
 *   node eval/eval_phase5_prerelease.mjs
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
const VERCEL_JSON = path.join(ROOT_DIR, 'vercel.json');
const OUT_PATH = path.join(__dirname, 'reports', 'phase5_prerelease.json');

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
  '.bat': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
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

function collectAssets(dir, prefix = '') {
  const rows = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      rows.push(...collectAssets(abs, rel));
    } else if (entry.isFile()) {
      rows.push({ rel, bytes: fs.statSync(abs).size });
    }
  }
  return rows.sort((a, b) => b.bytes - a.bytes);
}

function categorizeAsset(rel) {
  if (rel.endsWith('.onnx')) return 'onnx_models';
  if (rel.startsWith('vendor/ort/')) return 'ort_wasm_runtime';
  if (rel.startsWith('vendor/fonts/')) return 'fonts_sil_ofl';
  if (rel === 'reference_prototypes.json' || rel === 'stage2_calibration.json' || rel === 'yolo_meta.json' || rel.startsWith('model.js') || rel === 'javanese_lm_data.js') {
    return 'prototypes_and_lm_data';
  }
  if (rel.endsWith('.js') || rel.endsWith('.html') || rel.endsWith('.css') || rel === 'manifest.webmanifest') {
    return 'app_code_styles';
  }
  if (rel.startsWith('sample_') || rel === 'uji_halaman.png' || rel.includes('icon') || rel.startsWith('favicon')) {
    return 'samples_and_icons';
  }
  return 'misc_repo_files';
}

function parseSwConfig() {
  const swText = fs.readFileSync(path.join(WEB_DIR, 'sw.js'), 'utf-8');
  const cacheMatch = swText.match(/CACHE_NAME\s*=\s*['"]([^'"]+)['"]/);
  assert.ok(cacheMatch, 'CACHE_NAME harus didefinisikan di sw.js');
  const cacheName = cacheMatch[1];

  const listMatch = swText.match(/ASSETS_TO_CACHE\s*=\s*\[([\s\S]*?)\]/);
  assert.ok(listMatch, 'ASSETS_TO_CACHE harus didefinisikan di sw.js');
  const assets = [...listMatch[1].matchAll(/['"]([^'"]+)['"]/g)].map(m => m[1]);
  const missingOnDisk = [];
  for (const a of assets) {
    if (a === './') continue;
    const fp = path.join(WEB_DIR, a.replace(/^\.\//, ''));
    if (!fs.existsSync(fp)) missingOnDisk.push(a);
  }
  assert.equal(missingOnDisk.length, 0, `Aset SW hilang di disk: ${missingOnDisk.join(', ')}`);
  return { cacheName, cachedAssetsCount: assets.length, assets };
}

function startVercelParityServer(vercelCfg) {
  const swHeaders = (vercelCfg.headers || []).find(h => h.source === '/sw.js')?.headers || [];
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      let reqPath = (req.url || '/').split('?')[0];
      // Emulasi vercel.json rewrite: /app/web_camscanner/(.*) -> /$1
      if (reqPath.startsWith('/app/web_camscanner/')) {
        reqPath = '/' + reqPath.slice('/app/web_camscanner/'.length);
      }
      const cleanRel = reqPath === '/' ? 'index.html' : reqPath.replace(/^\/+/, '');
      const filePath = path.join(WEB_DIR, cleanRel);
      if (!filePath.startsWith(WEB_DIR) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.statusCode = 404;
        res.end('Not found');
        return;
      }
      const ext = path.extname(filePath).toLowerCase();
      res.setHeader('Content-Type', MIME_MAP[ext] || 'application/octet-stream');
      if (reqPath === '/sw.js') {
        for (const h of swHeaders) res.setHeader(h.key, h.value);
      }
      fs.createReadStream(filePath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => {
      resolve({ server, url: `http://127.0.0.1:${server.address().port}` });
    });
  });
}

async function main() {
  const vercelCfg = JSON.parse(fs.readFileSync(VERCEL_JSON, 'utf-8'));
  assert.equal(vercelCfg.outputDirectory, 'app/web_camscanner', 'vercel.json outputDirectory harus app/web_camscanner');

  const swInfo = parseSwConfig();
  const assets = collectAssets(WEB_DIR);
  const totalBytes = assets.reduce((a, b) => a + b.bytes, 0);
  const byCategory = {};
  for (const row of assets) {
    const cat = categorizeAsset(row.rel);
    if (!byCategory[cat]) byCategory[cat] = { files: 0, bytes: 0 };
    byCategory[cat].files += 1;
    byCategory[cat].bytes += row.bytes;
  }
  for (const v of Object.values(byCategory)) {
    v.mb_decimal = Number((v.bytes / 1e6).toFixed(2));
    v.mib_binary = Number((v.bytes / (1024 * 1024)).toFixed(2));
  }

  const { server, url } = await startVercelParityServer(vercelCfg);
  const browser = await chromium.launch({ headless: true, executablePath: findChrome() });
  const consoleErrors = [];
  const externalRequests = [];

  try {
    // 1. Uji Cold First Load dengan Vercel-parity server (SW block untuk mengukur initial render murni vs penuh)
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    page.on('pageerror', e => consoleErrors.push(String(e)));
    page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });

    const reqLog = [];
    page.on('response', async (resp) => {
      try {
        const u = resp.url();
        if (!u.startsWith(url)) return;
        const cl = resp.headers()['content-length'];
        const bufLen = cl ? parseInt(cl, 10) : (await resp.body().then(b => b.length).catch(() => 0));
        reqLog.push({ path: u.replace(url, ''), status: resp.status(), bytes: bufLen });
      } catch (_e) { /* ignore aborted */ }
    });

    await page.route('**/*', (route) => {
      const rUrl = route.request().url();
      if (rUrl.startsWith(url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) return route.continue();
      externalRequests.push(rUrl);
      return route.abort();
    });

    const t0 = performance.now();
    await page.goto(`${url}/`, { waitUntil: 'load' });
    const loadWallMs = Math.round(performance.now() - t0);
    await page.waitForFunction(
      () => document.querySelector('#engineStatus')?.textContent.includes('Stage-2'),
      null,
      { timeout: 45000 }
    );
    const engineReadyMs = Math.round(performance.now() - t0);

    const navPerf = await page.evaluate(() => {
      const n = performance.getEntriesByType('navigation')[0];
      return {
        domContentLoadedMs: n ? Math.round(n.domContentLoadedEventEnd) : null,
        loadEventMs: n ? Math.round(n.loadEventEnd) : null,
        ttfbMs: n ? Math.round(n.responseStart) : null,
      };
    });

    // Pastikan rewrite & Cache-Control /sw.js aktif sesuai vercel.json
    const swHead = await page.evaluate(async () => {
      const r1 = await fetch('/sw.js');
      const r2 = await fetch('/app/web_camscanner/yolo_meta.json');
      const swReg = await navigator.serviceWorker.ready;
      const cache = await caches.open((await caches.keys())[0]);
      const keys = await cache.keys();
      return {
        swStatus: r1.status,
        swCacheControl: r1.headers.get('cache-control'),
        rewriteStatus: r2.status,
        activeCacheKeys: keys.length,
        swScriptUrl: swReg.active?.scriptURL || '',
      };
    });

    assert.equal(swHead.swStatus, 200);
    assert.match(swHead.swCacheControl || '', /no-cache/i, '/sw.js harus Cache-Control: no-cache');
    assert.equal(swHead.rewriteStatus, 200, 'Rewrite /app/web_camscanner/* harus mengembalikan 200');
    assert.equal(consoleErrors.length, 0, `Console error harus 0: ${consoleErrors.join('; ')}`);
    assert.equal(externalRequests.length, 0, `Permintaan eksternal harus 0: ${externalRequests.join(', ')}`);

    const transferredBytes = reqLog.reduce((a, b) => a + b.bytes, 0);
    const report = {
      generatedAt: new Date().toISOString(),
      vercelConfig: {
        outputDirectory: vercelCfg.outputDirectory,
        rewriteVerified: swHead.rewriteStatus === 200,
        swNoCacheVerified: /no-cache/i.test(swHead.swCacheControl || ''),
        note: 'Uji pra-rilis lokal setara aturan vercel.json (Vercel live preview URL mengikuti push ke remote GitHub yang tertaut).',
      },
      serviceWorker: {
        cacheName: swInfo.cacheName,
        declaredAssetsCount: swInfo.cachedAssetsCount,
        verifiedOfflineCacheEntries: swHead.activeCacheKeys,
      },
      deployedAssets: {
        totalFiles: assets.length,
        totalBytes,
        totalMB_decimal: Number((totalBytes / 1e6).toFixed(2)),
        totalMiB_binary: Number((totalBytes / (1024 * 1024)).toFixed(2)),
        byCategory,
        largestFiles: assets.slice(0, 10),
      },
      coldFirstLoad: {
        ...navPerf,
        windowLoadWallMs: loadWallMs,
        fullOnnxPlusStage2ReadyMs: engineReadyMs,
        httpRequestsObserved: reqLog.length,
        transferredBytesObserved: transferredBytes,
        transferredMB_decimal: Number((transferredBytes / 1e6).toFixed(2)),
        externalRequestsCount: externalRequests.length,
        consoleErrorsCount: consoleErrors.length,
      },
    };

    fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
    fs.writeFileSync(OUT_PATH, JSON.stringify(report, null, 2), 'utf-8');
    console.log('[PRERELEASE AUDIT PASS]', JSON.stringify(report, null, 2));
    await ctx.close();
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((err) => {
  console.error('[PRERELEASE AUDIT FAILED]', err);
  process.exit(1);
});
