/**
 * Benchmark End-to-End Browser WASM Fase 5 (`v3.1` vs Ablasi `v4.0`)
 * -----------------------------------------------------------------
 * Menjalankan mesin peramban penuh (Chromium + ONNX Runtime Web 1.20.1 WASM) pada:
 *   1. `eval/heldout_real/manifest.jsonl` (n=42, terkunci SHA256SUMS.lock)
 *   2. `eval/failures/manifest.jsonl` (n=8, set diagnostik & kasus gagal nyata)
 *   3. `dataset/external/synth_v2/manifest.jsonl` split `val` bersih (n=40 deterministik)
 *   4. `dataset/external/manifests_v4/synth_v4.jsonl` split `val` negatif murni (n=29, Latin/garis tembok)
 *
 * Across 4 mode:
 *   - `v3_1`           : Commit baseline `eb19337` langsung dari Git object store (in-memory HTTP server)
 *   - `v4_enh_only`    : Branch saat ini, Worker Enhancer AKTIF, Stage-2 diblokir (Stage-1 fallback)
 *   - `v4_stage2_only` : Branch saat ini, Stage-2 (176 prototipe) AKTIF, Enhancer dilewati (`skipEnhance: true`)
 *   - `v4_full`        : Branch saat ini, Pipeline produksi default penuh (Enhancer + Stage-2 + Ortho v4 + LM)
 *
 * Penggunaan:
 *   node eval/eval_phase5_browser.mjs
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const WEB_DIR = path.join(ROOT_DIR, 'app', 'web_camscanner');
const ORT_DIST = path.join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');
const OUT_PATH = path.join(__dirname, 'artifacts', 'phase5_browser_raw.json');

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
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.webmanifest': 'application/manifest+json',
};

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

function verifyHeldoutLock() {
  const lockPath = path.join(ROOT_DIR, 'eval', 'heldout_real', 'SHA256SUMS.lock');
  if (!fs.existsSync(lockPath)) {
    throw new Error(`Lock file missing: ${lockPath}`);
  }
  const lines = fs.readFileSync(lockPath, 'utf-8').split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  let checked = 0;
  for (const ln of lines) {
    const m = ln.match(/^([0-9a-f]{64})\s+(.+)$/i);
    if (!m) throw new Error(`Invalid lock entry: ${ln}`);
    const [, expected, relPath] = m;
    const fp = path.join(ROOT_DIR, 'eval', 'heldout_real', relPath.trim());
    if (!fs.existsSync(fp)) throw new Error(`Heldout locked file missing: ${fp}`);
    const actual = crypto.createHash('sha256').update(fs.readFileSync(fp)).digest('hex');
    if (actual !== expected.toLowerCase()) {
      throw new Error(`SHA256 mismatch on ${relPath}: ${actual} != ${expected}`);
    }
    checked++;
  }
  console.log(`[HELDOUT LOCK OK] ${checked} files verified, mismatches=0`);
  return checked;
}

function loadBenchmarkItems() {
  const readJsonl = (rel) => fs.readFileSync(path.join(ROOT_DIR, rel), 'utf-8')
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(Boolean)
    .map(l => JSON.parse(l));

  const heldout = readJsonl('eval/heldout_real/manifest.jsonl').map(r => ({
    ...r,
    dataset: 'heldout_real',
    imageAbs: path.join(ROOT_DIR, r.image.replace(/\//g, path.sep)),
  }));

  const failures = readJsonl('eval/failures/manifest.jsonl').map(r => ({
    ...r,
    dataset: 'failures',
    imageAbs: path.join(ROOT_DIR, r.image.replace(/\//g, path.sep)),
  }));

  const synth2All = readJsonl('dataset/external/synth_v2/manifest.jsonl');
  const synth2Clean40 = synth2All
    .filter(r => r.split === 'val' && !r.distractor && r.unicode_gt)
    .map(r => ({
      ...r,
      dataset: 'clean_synth_v2_val',
      subset: 'synth_v2_val_clean',
      condition: r.bg_style || 'clean',
      image: r.image.replace(/\\/g, '/'),
      imageAbs: path.join(ROOT_DIR, r.image.replace(/[\\/]/g, path.sep)),
    }))
    .filter(r => fs.existsSync(r.imageAbs))
    .slice(0, 40);

  const synth4All = readJsonl('dataset/external/manifests_v4/synth_v4.jsonl');
  const synth4Neg29 = synth4All
    .filter(r => r.split === 'val' && (!r.boxes || r.boxes.length === 0))
    .map(r => ({
      ...r,
      id: r.id || r.image_id || path.basename(r.image),
      dataset: 'distractors_v4_val',
      subset: 'synth_v4_val_neg',
      condition: r.mode || r.bg_style || 'distractor',
      unicode_gt: '',
      image: r.image.replace(/\\/g, '/'),
      imageAbs: path.join(ROOT_DIR, r.image.replace(/[\\/]/g, path.sep)),
    }))
    .filter(r => fs.existsSync(r.imageAbs));

  const all = [...heldout, ...failures, ...synth2Clean40, ...synth4Neg29];
  console.log('[DATASETS LOADED]', {
    heldout_real: heldout.length,
    failures: failures.length,
    clean_synth_v2_val: synth2Clean40.length,
    distractors_v4_val: synth4Neg29.length,
    total: all.length,
  });
  return all;
}

function startWorkspaceServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = (req.url || '/').split('?')[0];
      const cleanRel = reqPath === '/' ? 'index.html' : reqPath.replace(/^\/+/, '');
      const filePath = path.join(WEB_DIR, cleanRel);
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
      const addr = server.address();
      resolve({ server, url: `http://127.0.0.1:${addr.port}` });
    });
  });
}

function startGitCommitServer(commitHash) {
  const blobCache = new Map();
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = (req.url || '/').split('?')[0];
      const cleanRel = reqPath === '/' ? 'index.html' : reqPath.replace(/^\/+/, '');
      if (cleanRel.includes('..')) {
        res.statusCode = 403;
        res.end('Forbidden');
        return;
      }
      let buf = blobCache.get(cleanRel);
      if (buf === undefined) {
        try {
          buf = execFileSync(
            'git',
            ['-C', ROOT_DIR, 'cat-file', 'blob', `${commitHash}:app/web_camscanner/${cleanRel}`],
            { maxBuffer: 64 * 1024 * 1024 }
          );
          blobCache.set(cleanRel, buf);
        } catch (_e) {
          blobCache.set(cleanRel, null);
          buf = null;
        }
      }
      if (!buf) {
        res.statusCode = 404;
        res.end('Not found');
        return;
      }
      const ext = path.extname(cleanRel).toLowerCase();
      res.setHeader('Content-Type', MIME_MAP[ext] || 'application/octet-stream');
      res.end(buf);
    });
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, url: `http://127.0.0.1:${addr.port}` });
    });
  });
}

async function runSingleImage(page, imgPath, modeCfg) {
  const t0 = performance.now();

  if (modeCfg.skipEnhance) {
    await page.evaluate(() => {
      window.aksaraEnhanceParams = { skipEnhance: true };
    });
  }

  await page.evaluate(() => {
    window.__aksEvalRunDone = false;
    window.__aksaraLastEnhanceReport = null;
    window.__aksaraLastStage2Stats = null;
    window.__aksaraLastKeptDetections = null;
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
  await page.waitForFunction(() => window.__aksEvalRunDone === true, null, { timeout: 90000 });
  const totalMs = Math.round((performance.now() - t0) * 10) / 10;

  const snap = await page.evaluate(async () => {
    const unicodeText = document.querySelector('#outJavanese')?.textContent || '—';
    const sentenceLatin = document.querySelector('#outSentenceLatin')?.textContent || '—';
    const sentenceIndonesian = document.querySelector('#outIndonesian')?.textContent || '—';
    const phoneticLatin = document.querySelector('#outLatin')?.textContent || '—';
    const engineStatus = document.querySelector('#engineStatus')?.textContent || '';
    const canvasW = document.querySelector('#mainCanvas')?.width || 0;
    const canvasH = document.querySelector('#mainCanvas')?.height || 0;
    const glyphCount = document.querySelectorAll('#glyphsGrid .glyph-card').length;

    let detections = [];
    if (Array.isArray(window.__aksaraLastKeptDetections)) {
      detections = window.__aksaraLastKeptDetections.map(d => ({
        classIdx: d.classIdx,
        label: d.label,
        category: d.category || 'nglegena',
        unicode: d.unicode || '',
        confidence: Number((d.confidence || 0).toFixed(4)),
        uncertain: Boolean(d.isUncertain),
        box: d.box ? d.box.map(v => Math.round(v)) : [],
        stage2Applied: Boolean(d.stage2?.applied),
        stage2Relabeled: Boolean(d.stage2?.relabeled),
      }));
    } else if (glyphCount > 0) {
      let capturedBlob = null;
      const origCreate = URL.createObjectURL;
      const origClick = HTMLAnchorElement.prototype.click;
      URL.createObjectURL = (blob) => { capturedBlob = blob; return 'blob:eval-capture'; };
      HTMLAnchorElement.prototype.click = () => {};
      try {
        document.querySelector('#btnDownload')?.click();
      } finally {
        HTMLAnchorElement.prototype.click = origClick;
        URL.createObjectURL = origCreate;
      }
      if (capturedBlob) {
        const parsed = JSON.parse(await capturedBlob.text());
        detections = (parsed.deteksi || []).map(d => ({
          label: d.label,
          category: d.kategori || 'nglegena',
          unicode: d.unicode || '',
          confidence: Number((d.confidence || 0).toFixed(4)),
          uncertain: Boolean(d.ragu),
          box: d.kotak ? d.kotak.map(v => Math.round(v)) : [],
        }));
      }
    }

    return {
      canvasW,
      canvasH,
      unicodeText,
      sentenceLatin,
      sentenceIndonesian,
      phoneticLatin,
      engineStatus,
      glyphCount,
      enhanceReport: window.__aksaraLastEnhanceReport || null,
      stage2Stats: window.__aksaraLastStage2Stats || null,
      detections,
    };
  });

  return { elapsedMs: totalMs, ...snap };
}

async function runModeSuite(browser, items, modeCfg) {
  console.log(`\n=== Running Browser Benchmark Mode: ${modeCfg.name} (${modeCfg.desc}) ===`);
  const srv = modeCfg.gitCommit
    ? await startGitCommitServer(modeCfg.gitCommit)
    : await startWorkspaceServer();

  const consoleErrors = [];
  const externalReqs = [];
  let inInference = false;

  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      serviceWorkers: 'block',
    });
    const page = await context.newPage();
    page.on('pageerror', e => consoleErrors.push(String(e)));
    page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });

    await page.route('**/*', async (route) => {
      const rUrl = route.request().url();
      if (rUrl.startsWith(srv.url) || rUrl.startsWith('data:') || rUrl.startsWith('blob:')) {
        if (modeCfg.abortStage2 && (
          rUrl.endsWith('/aksara_embed.onnx') ||
          rUrl.endsWith('/reference_prototypes.json') ||
          rUrl.endsWith('/stage2_calibration.json')
        )) {
          await route.abort();
          return;
        }
        await route.continue();
      } else if (modeCfg.gitCommit && rUrl.startsWith('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/')) {
        // Commit eb19337 loaded ORT 1.20.1 from CDN; satisfy deterministically from local vendored eval dist
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
      } else if (modeCfg.gitCommit && (rUrl.includes('fonts.googleapis.com') || rUrl.includes('fonts.gstatic.com'))) {
        await route.fulfill({ status: 200, contentType: 'text/css', body: '' });
      } else {
        if (inInference) externalReqs.push(rUrl);
        await route.abort();
      }
    });

    const tNav0 = performance.now();
    await page.goto(`${srv.url}/index.html`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.querySelector('#engineStatus')?.textContent.includes('YOLOv8 Siap'),
      null,
      { timeout: 60000 }
    );
    if (!modeCfg.gitCommit && !modeCfg.abortStage2) {
      // Wait for Stage-2 session to become ready too
      await page.waitForFunction(
        () => window.AksaraStage2 && (window.AksaraStage2.getStatus().ready || window.AksaraStage2.getStatus().fallbackReason),
        null,
        { timeout: 30000 }
      );
    }
    const initReadyMs = Math.round(performance.now() - tNav0);
    const runtimeInfo = await page.evaluate(() => ({
      engineStatus: document.querySelector('#engineStatus')?.textContent || '',
      crossOriginIsolated: Boolean(window.crossOriginIsolated),
      stage2Ready: Boolean(window.AksaraStage2?.getStatus().ready),
    }));
    console.log(`  Ready in ${initReadyMs} ms | status="${runtimeInfo.engineStatus}" | coi=${runtimeInfo.crossOriginIsolated}`);

    // Once at default landing on v4_full, derive deterministic canonical Latin GT from Unicode GT for Word Accuracy checks
    let canonicalLatinMap = null;
    if (modeCfg.extractCanonicalLatin) {
      const payload = items.map(it => ({ id: it.id, unicode_gt: it.unicode_gt || '' }));
      canonicalLatinMap = await page.evaluate((list) => {
        const ortho = window.AksaraOrthography;
        const PADA_CHARS = new Set('\uA9C1\uA9C2\uA9C3\uA9C4\uA9C5\uA9C6\uA9C7\uA9C8\uA9C9\uA9CA\uA9CB\uA9CC\uA9CD\uA9CF\uA9DE\uA9DF');
        const COMBINING = new Set([
          '\uA980', '\uA981', '\uA982', '\uA983',
          '\uA9B3', '\uA9B4', '\uA9B5', '\uA9B6', '\uA9B7', '\uA9B8',
          '\uA9B9', '\uA9BA', '\uA9BB', '\uA9BC', '\uA9BD', '\uA9BE', '\uA9BF',
        ]);
        const splitGraphemes = (line) => {
          const s = (line || '').replace(/\s+/g, '');
          const clusters = [];
          let i = 0;
          while (i < s.length) {
            const ch = s[i];
            const cp = ch.codePointAt(0);
            if (PADA_CHARS.has(ch)) {
              clusters.push(ch);
              i += 1;
              continue;
            }
            if (cp >= 0xA984 && cp <= 0xA9B2) {
              let cl = ch;
              i += 1;
              while (i < s.length) {
                const nxt = s[i];
                const ncp = nxt.codePointAt(0);
                if (nxt === '\uA9C0') {
                  cl += nxt;
                  i += 1;
                  if (i < s.length) {
                    const bcp = s[i].codePointAt(0);
                    if (bcp >= 0xA984 && bcp <= 0xA9B2) {
                      cl += s[i];
                      i += 1;
                      continue;
                    }
                  }
                } else if (COMBINING.has(nxt) || (ncp >= 0xA9B3 && ncp <= 0xA9BF)) {
                  cl += nxt;
                  i += 1;
                } else {
                  break;
                }
              }
              clusters.push(cl);
            } else {
              clusters.push(ch);
              i += 1;
            }
          }
          return clusters;
        };
        const out = {};
        for (const row of list) {
          if (!row.unicode_gt) {
            out[row.id] = '';
            continue;
          }
          const lines = row.unicode_gt.split(/\r?\n/).filter(Boolean);
          const latParts = lines.map((ln) => {
            const cls = splitGraphemes(ln);
            return cls.length ? ortho.decodeClustersToLatin(cls) : '';
          }).filter(Boolean);
          out[row.id] = latParts.join(' ').replace(/\s+/g, ' ').replace(/\s+([,.;:!?])/g, '$1').trim();
        }
        return out;
      }, payload);
    }

    inInference = true;
    const results = [];
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      const res = await runSingleImage(page, it.imageAbs, modeCfg);
      results.push({
        id: it.id,
        dataset: it.dataset,
        subset: it.subset || it.dataset,
        condition: it.condition || it.domain || '',
        ...res,
      });
      if ((i + 1) % 25 === 0 || i + 1 === items.length) {
        console.log(`  [${modeCfg.name}] ${i + 1}/${items.length} done (last=${it.id} ${res.elapsedMs}ms)`);
      }
    }

    await context.close();
    return {
      mode: modeCfg.name,
      desc: modeCfg.desc,
      initReadyMs,
      runtimeInfo,
      consoleErrors,
      externalRequestsDuringInference: externalReqs,
      canonicalLatinMap,
      results,
    };
  } finally {
    srv.server.close();
  }
}

async function main() {
  verifyHeldoutLock();
  const items = loadBenchmarkItems();
  const exe = findChromeExecutable();
  const browser = await chromium.launch({ headless: true, executablePath: exe });

  try {
    const modeConfigs = [
      {
        name: 'v3_1',
        desc: 'Baseline v3.1 (git eb19337: YOLOv8n + old preproc + old orthography/beam)',
        gitCommit: 'eb19337',
        abortStage2: false,
        skipEnhance: false,
      },
      {
        name: 'v4_enh_only',
        desc: 'v4.0 Worker Enhancer + Orthography v4 + LM Beam (Stage-2 disabled fallback)',
        gitCommit: null,
        abortStage2: true,
        skipEnhance: false,
      },
      {
        name: 'v4_stage2_only',
        desc: 'v4.0 Stage-2 Metric Verifier (176 prototypes) + Orthography v4 + LM Beam (Enhancer skipped)',
        gitCommit: null,
        abortStage2: false,
        skipEnhance: true,
      },
      {
        name: 'v4_full',
        desc: 'v4.0 Full Default (Worker Enhancer + Stage-2 176 Prototypes + Orthography v4 + LM Beam)',
        gitCommit: null,
        abortStage2: false,
        skipEnhance: false,
        extractCanonicalLatin: true,
      },
    ];

    const modeOutputs = {};
    let canonicalLatinMap = {};
    for (const cfg of modeConfigs) {
      const out = await runModeSuite(browser, items, cfg);
      if (out.canonicalLatinMap) {
        canonicalLatinMap = out.canonicalLatinMap;
        delete out.canonicalLatinMap;
      }
      modeOutputs[cfg.name] = out;
    }

    const cpuInfo = os.cpus();
    const payload = {
      generatedAt: new Date().toISOString(),
      environment: {
        os: `${os.type()} ${os.release()} (${os.arch()})`,
        cpuModel: cpuInfo[0]?.model || 'unknown',
        cpuLogicalCores: cpuInfo.length,
        totalRamGB: Number((os.totalmem() / (1024 ** 3)).toFixed(2)),
        nodeVersion: process.version,
        browserVersion: browser.version(),
        ortWebVersion: '1.20.1',
      },
      itemsMeta: items.map(it => ({
        id: it.id,
        dataset: it.dataset,
        subset: it.subset || it.dataset,
        condition: it.condition || it.domain || '',
        image: it.image,
        unicode_gt: it.unicode_gt || '',
        latin_gt: it.latin_gt || null,
        canonical_latin_gt: canonicalLatinMap[it.id] || '',
        boxes: it.boxes || [],
      })),
      modes: modeOutputs,
    };

    fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
    fs.writeFileSync(OUT_PATH, JSON.stringify(payload, null, 2), 'utf-8');
    console.log(`\n[SAVED RAW PHASE 5 BROWSER BENCHMARK] -> ${OUT_PATH}`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error('[PHASE 5 BROWSER BENCHMARK FAILED]', err);
  process.exit(1);
});
