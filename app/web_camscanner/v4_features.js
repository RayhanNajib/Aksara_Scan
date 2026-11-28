/**
 * AksaraScan v4.0 — Fitur Interaktif Fase 4:
 *  1) Inspektor Referensi Stage-2 & Top-3 Kandidat (+ riwayat koreksi lokalStorage)
 *  2) Mode Batch Multi-Halaman dengan ekspor .txt / .md / .json / Salin Semua
 *  3) Riwayat Pindai Lokal IndexedDB (simpan & bersihkan 1-klik)
 *  4) Penjelajah Bank Prototipe 56 Kelas di Lab & Referensi
 */
(function () {
  'use strict';

  const CORR_KEY = 'aksarascan_v3_corrections_v1';
  const DB_NAME = 'aksarascan_history_db';
  const DB_STORE = 'scans';

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  function loadCorrList() {
    try {
      const raw = localStorage.getItem(CORR_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (_e) {
      return [];
    }
  }

  function appendCorrEntry(entry) {
    try {
      const arr = loadCorrList();
      arr.push({ ...entry, savedAt: new Date().toISOString() });
      localStorage.setItem(CORR_KEY, JSON.stringify(arr.slice(-300)));
      const badge = document.getElementById('v3CorrectionCount');
      if (badge) badge.textContent = String(Math.min(300, arr.length));
    } catch (_e) { /* ignore */ }
  }

  // ==========================================================================
  // 1. INSPEKTOR REFERENSI STAGE-2 (Mengapa X dan bukan Y?)
  // ==========================================================================
  let lastInspectedIdx = -1;

  function drawCroppedPatch(srcEl, box, targetCanvas) {
    if (!targetCanvas) return;
    const ctx = targetCanvas.getContext('2d');
    const W = targetCanvas.width, H = targetCanvas.height;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);
    if (!srcEl || !box) return;
    const [x0, y0, x1, y1] = box;
    const sw = Math.max(1, x1 - x0), sh = Math.max(1, y1 - y0);
    const pad = Math.max(4, Math.round(Math.max(sw, sh) * 0.12));
    const sx = Math.max(0, x0 - pad), sy = Math.max(0, y0 - pad);
    const sx1 = Math.min(srcEl.naturalWidth || srcEl.width, x1 + pad);
    const sy1 = Math.min(srcEl.naturalHeight || srcEl.height, y1 + pad);
    const cw = Math.max(1, sx1 - sx), ch = Math.max(1, sy1 - sy);
    const sc = Math.min((W - 10) / cw, (H - 10) / ch);
    const dw = cw * sc, dh = ch * sc;
    ctx.drawImage(srcEl, sx, sy, cw, ch, (W - dw) / 2, (H - dh) / 2, dw, dh);
  }

  function renderCorrectionHistoryPanel() {
    const pan = $('#corrHistoryPanel');
    if (!pan) return;
    const items = loadCorrList();
    pan.replaceChildren();
    if (!items.length) {
      pan.textContent = 'Belum ada koreksi manual tersimpan di perangkat ini.';
      return;
    }
    const header = document.createElement('div');
    header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:8px;';
    const title = document.createElement('strong');
    title.textContent = `Riwayat koreksi tersimpan (${items.length})`;
    const clearBtn = document.createElement('button');
    clearBtn.type = 'button';
    clearBtn.className = 'btn btn--sm';
    clearBtn.textContent = 'Hapus Riwayat';
    clearBtn.addEventListener('click', () => {
      try { localStorage.removeItem(CORR_KEY); } catch (_e) {}
      const badge = document.getElementById('v3CorrectionCount');
      if (badge) badge.textContent = '0';
      renderCorrectionHistoryPanel();
    });
    header.append(title, clearBtn);
    pan.appendChild(header);

    const list = document.createElement('div');
    list.style.cssText = 'display:flex;flex-direction:column;gap:4px;max-height:140px;overflow-y:auto;';
    items.slice(-12).reverse().forEach((it) => {
      const row = document.createElement('div');
      row.className = 'telemetry-mono';
      row.style.fontSize = '11.5px';
      row.textContent = `${it.from || '?'} → ${it.to || '?'} (${it.type || 'manual'})`;
      list.appendChild(row);
    });
    pan.appendChild(list);
  }

  function renderInspector(det, idx, sourceImage, enhancedCanvas) {
    lastInspectedIdx = idx;
    const cropCanvas = $('#inspUserCrop');
    const srcForCrop = enhancedCanvas || sourceImage;
    drawCroppedPatch(srcForCrop, det.box, cropCanvas);

    const bank = window.AksaraStage2?.getBank ? window.AksaraStage2.getBank() : null;
    const badge = $('#inspCropMatchBadge');
    const refs = Array.isArray(det.referenceMatches) ? det.referenceMatches : [];
    const topMatch = refs.find((r) => r.label === det.label) || refs[0] || null;
    if (badge) {
      if (topMatch) {
        const styleName = topMatch.style === 'handwritten' ? 'tulisan tangan' : 'kanonik cetak';
        badge.textContent = `Cocok Ref ${topMatch.label} · ${styleName} (${topMatch.similarityPct}%)`;
      } else {
        badge.textContent = `Keyakinan ${Math.round((det.confidence || 0) * 100)}%`;
      }
    }

    const sub = $('#inspSubMeta');
    if (sub) {
      sub.textContent = topMatch
        ? `Cocok dengan Referensi "${topMatch.label}" (${topMatch.similarityPct}%). Bandingkan Top-3 kandidat di bawah atau ketik koreksi kelas lain.`
        : 'Bandingkan potongan citra Anda dengan kandidat detektor teratas atau pilih kelas pengganti.';
    }

    const list = $('#inspTop3List');
    if (!list) return;
    list.replaceChildren();

    const cands = (det.topK && det.topK.length) ? det.topK.slice(0, 3) : [{
      classIdx: det.classIdx,
      label: det.label,
      category: det.category,
      prob: det.confidence,
    }];

    cands.forEach((c) => {
      const row = document.createElement('div');
      row.className = 'cand-row';

      const left = document.createElement('div');
      left.className = 'cand-row__left';

      const thumbsWrap = document.createElement('div');
      thumbsWrap.className = 'cand-row__thumbs';
      const refEntry = refs.find((r) => r.label === c.label);
      const bankEntry = bank && bank.classes ? bank.classes.find((bc) => bc.label === c.label) : null;
      const thumbItems = bankEntry && Array.isArray(bankEntry.thumbs) && bankEntry.thumbs.length
        ? bankEntry.thumbs.slice(0, 2)
        : (refEntry && refEntry.thumbPng ? [{ style: refEntry.style, png: refEntry.thumbPng }] : []);

      thumbItems.forEach((t) => {
        if (!t.png) return;
        const im = document.createElement('img');
        im.src = t.png;
        im.alt = `Referensi ${c.label} (${t.style})`;
        im.title = t.style === 'handwritten' ? 'Variasi tulisan tangan' : 'Kanonik cetak';
        thumbsWrap.appendChild(im);
      });

      const txt = document.createElement('div');
      const lbl = document.createElement('div');
      lbl.className = 'cand-row__label';
      const uniSpan = document.createElement('span');
      uniSpan.textContent = (window.AksaraOrthography?.BASE_CHARS && det.unicode) ? '' : '';
      // Ambil unicode dari tombol correction-grid yang sudah disinkronkan
      const gridBtn = Array.from(document.querySelectorAll('#correctionGrid .correction-btn'))
        .find((b) => b.dataset.label === c.label.toLowerCase());
      const uniText = gridBtn ? (gridBtn.querySelector('.aksara-glyph')?.textContent || '') : det.unicode;
      uniSpan.textContent = uniText;
      lbl.append(uniSpan, document.createTextNode(c.label));

      const meta = document.createElement('div');
      meta.className = 'cand-row__meta';
      const probPct = Math.round((c.prob || 0) * 100);
      const protoPct = c.protoSim !== undefined ? Math.round(c.protoSim * 100) : (refEntry ? refEntry.similarityPct : null);
      meta.textContent = protoPct !== null
        ? `Skor gabung ${probPct}% · Kemiripan referensi ${protoPct}%`
        : `Skor detektor ${probPct}%`;

      txt.append(lbl, meta);
      left.append(thumbsWrap, txt);

      const act = document.createElement('button');
      act.type = 'button';
      act.className = c.label === det.label ? 'btn btn--sm' : 'btn btn--sm btn--primary';
      act.textContent = c.label === det.label ? 'Aktif' : 'Pilih';
      act.addEventListener('click', () => {
        if (c.label !== det.label) {
          appendCorrEntry({ type: 'inspector_topk', from: det.label, to: c.label, index: lastInspectedIdx });
          window.AksaraApp?.applyCandidateCorrection(lastInspectedIdx, {
            label: c.label,
            category: c.category,
            prob: c.prob,
            unicode: uniText,
          });
        }
        window.AksaraApp?.closeCorrectionModal();
      });

      row.append(left, act);
      list.appendChild(row);
    });

    const histPan = $('#corrHistoryPanel');
    if (histPan) histPan.hidden = true;
    const filterEl = $('#inspClassFilter');
    if (filterEl) {
      filterEl.value = '';
      filterGridButtons('');
    }
  }

  function filterGridButtons(q) {
    const query = (q || '').trim().toLowerCase();
    $$('#correctionGrid .correction-btn').forEach((btn) => {
      const lbl = btn.dataset.label || '';
      btn.hidden = Boolean(query && !lbl.includes(query));
    });
  }

  function initInspectorHooks() {
    const filterEl = $('#inspClassFilter');
    if (filterEl) {
      filterEl.addEventListener('input', (e) => filterGridButtons(e.target.value));
    }
    const histBtn = $('#btnViewCorrHistory');
    if (histBtn) {
      histBtn.addEventListener('click', () => {
        const pan = $('#corrHistoryPanel');
        if (!pan) return;
        pan.hidden = !pan.hidden;
        if (!pan.hidden) renderCorrectionHistoryPanel();
      });
    }
    window.addEventListener('aksara:correction', (ev) => {
      if (ev.detail) appendCorrEntry(ev.detail);
    });
  }

  // ==========================================================================
  // 2. MODE BATCH MULTI-HALAMAN
  // ==========================================================================
  const batchQueue = [];
  let batchRunning = false;

  function refreshBatchUI() {
    const listEl = $('#batchList');
    const sumEl = $('#batchSummary');
    if (!listEl || !sumEl) return;

    const doneItems = batchQueue.filter((it) => it.status === 'done');
    const hasDone = doneItems.length > 0;
    ['#btnBatchCopyAll', '#btnBatchTxt', '#btnBatchMd', '#btnBatchJson', '#btnBatchClear'].forEach((sel) => {
      const b = $(sel);
      if (b) b.disabled = sel === '#btnBatchClear' ? (batchQueue.length === 0 || batchRunning) : !hasDone;
    });

    if (!batchQueue.length) {
      sumEl.textContent = 'Belum ada berkas dalam antrean.';
      listEl.replaceChildren();
      return;
    }

    const total = batchQueue.length;
    const errCnt = batchQueue.filter((it) => it.status === 'error').length;
    sumEl.textContent = batchRunning
      ? `Memproses ${doneItems.length + 1} / ${total} halaman…`
      : `${doneItems.length} selesai dari ${total} halaman${errCnt ? ` (${errCnt} gagal)` : ''}.`;

    listEl.replaceChildren();
    batchQueue.forEach((it) => {
      const card = document.createElement('div');
      card.className = 'batch-item';

      const img = document.createElement('img');
      img.className = 'batch-item__thumb';
      img.alt = it.file.name;
      if (it.thumbDataUrl) img.src = it.thumbDataUrl;

      const body = document.createElement('div');
      const head = document.createElement('div');
      head.className = 'batch-item__head';
      const nm = document.createElement('span');
      nm.className = 'batch-item__name';
      nm.textContent = it.file.name;
      const st = document.createElement('span');
      st.className = 'batch-item__state';
      if (it.status === 'done') st.textContent = `Selesai · ${it.detectionsCount} glif · ${it.latencyMs} ms`;
      else if (it.status === 'processing') st.textContent = 'Memproses…';
      else if (it.status === 'error') st.textContent = 'Gagal';
      else st.textContent = 'Menunggu';
      head.append(nm, st);

      const uEl = document.createElement('div');
      uEl.className = 'batch-item__aksara';
      uEl.textContent = it.status === 'done' ? (it.unicodeText || '—') : (it.errorMsg || '—');

      const lEl = document.createElement('div');
      lEl.className = 'batch-item__latin';
      lEl.textContent = it.status === 'done' ? (it.sentenceLatin || '—') : '';

      const mEl = document.createElement('div');
      mEl.className = 'batch-item__meaning';
      mEl.textContent = it.status === 'done' ? (it.sentenceIndonesian || '') : '';

      body.append(head, uEl, lEl, mEl);
      card.append(img, body);
      listEl.appendChild(card);
    });
  }

  async function processBatchQueue() {
    if (batchRunning) return;
    batchRunning = true;
    try {
      for (const item of batchQueue) {
        if (item.status !== 'pending') continue;
        item.status = 'processing';
        refreshBatchUI();
        try {
          const out = await window.AksaraApp.processImageFileBatch(item.file);
          Object.assign(item, out, { status: 'done' });
        } catch (err) {
          item.status = 'error';
          item.errorMsg = err?.message || 'Gagal membaca gambar';
        }
        refreshBatchUI();
      }
    } finally {
      batchRunning = false;
      refreshBatchUI();
    }
  }

  function queueBatchFiles(fileList) {
    const files = Array.from(fileList || []).filter((f) => f && f.type && f.type.startsWith('image/'));
    if (!files.length) return;
    files.forEach((f) => {
      batchQueue.push({
        id: `${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
        file: f,
        status: 'pending',
        thumbDataUrl: '',
        unicodeText: '',
        sentenceLatin: '',
        sentenceIndonesian: '',
        detectionsCount: 0,
        latencyMs: 0,
      });
    });
    refreshBatchUI();
    processBatchQueue();
  }

  function buildBatchExport(kind) {
    const done = batchQueue.filter((it) => it.status === 'done');
    if (kind === 'json') {
      return JSON.stringify({
        aplikasi: 'AksaraScan Batch v4.0',
        exportedAt: new Date().toISOString(),
        totalHalaman: done.length,
        hasil: done.map((d, i) => ({
          halaman: i + 1,
          berkas: d.fileName,
          unicode: d.unicodeText,
          latin: d.sentenceLatin,
          indonesia: d.sentenceIndonesian,
          jumlahGlif: d.detectionsCount,
          keyakinanRata: d.avgConf,
          blur: d.blurLevel,
        })),
      }, null, 2);
    }
    if (kind === 'md') {
      return `# Hasil Batch AksaraScan (${done.length} Halaman)\n\n` + done.map((d, i) =>
        `## Halaman ${i + 1}: ${d.fileName}\n\n- **Unicode:** ${d.unicodeText}\n- **Latin:** ${d.sentenceLatin}\n- **Arti:** ${d.sentenceIndonesian}\n`
      ).join('\n');
    }
    return done.map((d, i) =>
      `[${i + 1}] ${d.fileName}\nUnicode: ${d.unicodeText}\nLatin: ${d.sentenceLatin}\nArti: ${d.sentenceIndonesian}\n`
    ).join('\n---\n');
  }

  function triggerDownload(text, filename, mime = 'text/plain;charset=utf-8') {
    const blob = new Blob([text], { type: mime });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function initBatchUI() {
    const bInp = $('#batchInput');
    const bDrop = $('#batchDropzone');
    if ($('#btnBatchPick')) $('#btnBatchPick').addEventListener('click', () => bInp?.click());
    if ($('#btnBatchBrowse')) $('#btnBatchBrowse').addEventListener('click', () => bInp?.click());
    if (bInp) {
      bInp.addEventListener('change', (e) => {
        if (e.target.files?.length) queueBatchFiles(e.target.files);
        e.target.value = '';
      });
    }
    if (bDrop) {
      bDrop.addEventListener('dragover', (e) => { e.preventDefault(); bDrop.classList.add('dragover'); });
      bDrop.addEventListener('dragleave', () => bDrop.classList.remove('dragover'));
      bDrop.addEventListener('drop', (e) => {
        e.preventDefault();
        bDrop.classList.remove('dragover');
        if (e.dataTransfer?.files?.length) queueBatchFiles(e.dataTransfer.files);
      });
    }
    if ($('#btnBatchCopyAll')) {
      $('#btnBatchCopyAll').addEventListener('click', () => {
        const text = buildBatchExport('txt');
        if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).catch(() => {});
      });
    }
    if ($('#btnBatchTxt')) $('#btnBatchTxt').addEventListener('click', () => triggerDownload(buildBatchExport('txt'), `AksaraScan_Batch_${Date.now()}.txt`));
    if ($('#btnBatchMd')) $('#btnBatchMd').addEventListener('click', () => triggerDownload(buildBatchExport('md'), `AksaraScan_Batch_${Date.now()}.md`, 'text/markdown;charset=utf-8'));
    if ($('#btnBatchJson')) $('#btnBatchJson').addEventListener('click', () => triggerDownload(buildBatchExport('json'), `AksaraScan_Batch_${Date.now()}.json`, 'application/json'));
    if ($('#btnBatchClear')) {
      $('#btnBatchClear').addEventListener('click', () => {
        if (batchRunning) return;
        batchQueue.length = 0;
        refreshBatchUI();
      });
    }
  }

  // ==========================================================================
  // 3. RIWAYAT PINDAI LOKAL (IndexedDB)
  // ==========================================================================
  function openHistoryDb() {
    return new Promise((resolve, reject) => {
      if (typeof indexedDB === 'undefined') return reject(new Error('IndexedDB unavailable'));
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(DB_STORE)) {
          db.createObjectStore(DB_STORE, { keyPath: 'id' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async function saveHistoryRecord(rec) {
    try {
      const db = await openHistoryDb();
      const tx = db.transaction(DB_STORE, 'readwrite');
      const store = tx.objectStore(DB_STORE);
      store.put(rec);
      await new Promise((res) => { tx.oncomplete = res; tx.onerror = res; });

      // Pangkas bila > 12 entri
      const rtx = db.transaction(DB_STORE, 'readwrite');
      const rstore = rtx.objectStore(DB_STORE);
      const allReq = rstore.getAll();
      allReq.onsuccess = () => {
        const rows = (allReq.result || []).sort((a, b) => b.savedAtMs - a.savedAtMs);
        rows.slice(12).forEach((old) => rstore.delete(old.id));
      };
      renderRecentScans();
    } catch (_e) { /* IndexedDB fallback silent */ }
  }

  async function renderRecentScans() {
    const wrap = $('#recentScansWrap');
    const listEl = $('#recentScansList');
    if (!wrap || !listEl) return;
    try {
      const db = await openHistoryDb();
      const tx = db.transaction(DB_STORE, 'readonly');
      const req = tx.objectStore(DB_STORE).getAll();
      const rows = await new Promise((res) => {
        req.onsuccess = () => res(req.result || []);
        req.onerror = () => res([]);
      });
      rows.sort((a, b) => b.savedAtMs - a.savedAtMs);
      if (!rows.length) {
        wrap.hidden = true;
        listEl.replaceChildren();
        return;
      }
      wrap.hidden = false;
      listEl.replaceChildren();
      rows.slice(0, 6).forEach((row) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'recent-item';
        const im = document.createElement('img');
        im.src = row.thumbDataUrl || './favicon.png';
        im.alt = row.fileName || 'Riwayat pindaian';

        const b = document.createElement('div');
        b.className = 'recent-item__body';
        const lat = document.createElement('div');
        lat.className = 'recent-item__latin';
        lat.textContent = row.sentenceLatin || row.unicodeText || row.fileName;
        const meta = document.createElement('div');
        meta.className = 'recent-item__meta';
        const dt = new Date(row.savedAtMs);
        const hm = `${String(dt.getHours()).padStart(2, '0')}:${String(dt.getMinutes()).padStart(2, '0')}`;
        meta.textContent = `${row.fileName} · ${hm} · ${row.avgConf}%`;
        b.append(lat, meta);
        btn.append(im, b);

        btn.addEventListener('click', () => {
          if (row.previewDataUrl) {
            const img = new Image();
            img.onload = () => {
              const evtDrop = document.getElementById('scannerEmptyWrap');
              const wb = document.getElementById('workbenchWrap');
              if (evtDrop) evtDrop.hidden = true;
              if (wb) wb.hidden = false;
              const mc = document.getElementById('mainCanvas');
              if (mc) {
                mc.width = img.width; mc.height = img.height;
                mc.getContext('2d').drawImage(img, 0, 0);
              }
              if ($('#outJavanese')) $('#outJavanese').textContent = row.unicodeText || '—';
              if ($('#outSentenceLatin')) $('#outSentenceLatin').textContent = row.sentenceLatin || '—';
              if ($('#outIndonesian')) $('#outIndonesian').textContent = row.sentenceIndonesian || '—';
              if ($('#srcDim')) $('#srcDim').textContent = `${row.fileName} (riwayat lokal)`;
              if ($('#meterWrap')) $('#meterWrap').hidden = false;
              if ($('#meterVal')) $('#meterVal').textContent = `${row.avgConf || 0}%`;
              if ($('#meterFill')) $('#meterFill').style.width = `${row.avgConf || 0}%`;
            };
            img.src = row.previewDataUrl;
          }
        });
        listEl.appendChild(btn);
      });
    } catch (_e) {
      wrap.hidden = true;
    }
  }

  function initHistoryUI() {
    renderRecentScans();
    if ($('#btnClearHistory')) {
      $('#btnClearHistory').addEventListener('click', async () => {
        try {
          const db = await openHistoryDb();
          const tx = db.transaction(DB_STORE, 'readwrite');
          tx.objectStore(DB_STORE).clear();
          await new Promise((res) => { tx.oncomplete = res; });
        } catch (_e) {}
        renderRecentScans();
      });
    }
    window.addEventListener('aksara:scancomplete', (ev) => {
      const d = ev.detail || {};
      if (!d.detectionsCount) return;
      const snap = window.AksaraApp?.getStateSnapshot ? window.AksaraApp.getStateSnapshot() : {};
      const srcEl = snap.sourceImage;
      let thumbDataUrl = '';
      let previewDataUrl = '';
      if (srcEl) {
        try {
          const sw = srcEl.naturalWidth || srcEl.width || 120;
          const sh = srcEl.naturalHeight || srcEl.height || 80;
          const tc = document.createElement('canvas');
          const sc = Math.min(1, 120 / Math.max(sw, sh));
          tc.width = Math.max(1, Math.round(sw * sc));
          tc.height = Math.max(1, Math.round(sh * sc));
          tc.getContext('2d').drawImage(srcEl, 0, 0, tc.width, tc.height);
          thumbDataUrl = tc.toDataURL('image/jpeg', 0.72);

          const pc = document.createElement('canvas');
          const psc = Math.min(1, 720 / Math.max(sw, sh));
          pc.width = Math.max(1, Math.round(sw * psc));
          pc.height = Math.max(1, Math.round(sh * psc));
          pc.getContext('2d').drawImage(srcEl, 0, 0, pc.width, pc.height);
          previewDataUrl = pc.toDataURL('image/jpeg', 0.76);
        } catch (_e) {}
      }
      saveHistoryRecord({
        id: `${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        savedAtMs: Date.now(),
        fileName: d.fileName || 'Pindaian',
        unicodeText: d.unicodeText,
        sentenceLatin: d.sentenceLatin,
        sentenceIndonesian: d.sentenceIndonesian,
        avgConf: d.avgConf || 0,
        blurLevel: d.blurLevel || '—',
        thumbDataUrl,
        previewDataUrl,
      });
    });
  }

  // ==========================================================================
  // 4. PENJELAJAH BANK PROTOTIPE STAGE-2 (56 Kelas)
  // ==========================================================================
  let protoCategory = 'all';

  function getProtoCat(label) {
    if (label === 'non_aksara') return 'non_aksara';
    if (label.startsWith('p_')) return 'pasangan';
    if (label.startsWith('pada_') || ['wulu', 'suku', 'pepet', 'taling', 'tarung', 'layar', 'cecak', 'wignyan', 'pangkon', 'cakra', 'keret', 'pengkal'].includes(label)) {
      return 'sandhangan';
    }
    return 'nglegena';
  }

  function renderProtoBank() {
    const grid = $('#protoBankGrid');
    const note = $('#protoStatusNote');
    if (!grid || !note) return;

    const bank = window.AksaraStage2?.getBank ? window.AksaraStage2.getBank() : null;
    if (!bank || !Array.isArray(bank.classes)) {
      note.textContent = 'Bank prototipe sedang dimuat atau Stage-2 dalam mode fallback.';
      grid.replaceChildren();
      return;
    }

    const q = ($('#protoSearchInput')?.value || '').trim().toLowerCase();
    const filtered = bank.classes.filter((cls) => {
      const cat = getProtoCat(cls.label);
      if (protoCategory !== 'all' && cat !== protoCategory) return false;
      if (q && !cls.label.toLowerCase().includes(q)) return false;
      return true;
    });

    note.textContent = `Menampilkan ${filtered.length} dari ${bank.classes.length} kelas (${bank.total_prototypes || 176} prototipe acuan train-only).`;
    grid.replaceChildren();

    filtered.forEach((cls) => {
      const card = document.createElement('div');
      card.className = 'proto-card';

      const top = document.createElement('div');
      top.className = 'proto-card__top';
      const lbl = document.createElement('strong');
      lbl.textContent = `#${cls.cid} · ${cls.label}`;
      const gl = document.createElement('span');
      gl.className = 'proto-card__glyph';
      const gridBtn = Array.from(document.querySelectorAll('#correctionGrid .correction-btn'))
        .find((b) => b.dataset.label === cls.label.toLowerCase());
      gl.textContent = cls.label === 'non_aksara' ? '—' : (gridBtn?.querySelector('.aksara-glyph')?.textContent || '');
      top.append(lbl, gl);

      const thumbsRow = document.createElement('div');
      thumbsRow.className = 'proto-card__thumbs';
      (cls.thumbs || []).forEach((t) => {
        if (!t.png) return;
        const wrap = document.createElement('div');
        wrap.className = 'proto-thumb-wrap';
        const im = document.createElement('img');
        im.src = t.png;
        im.alt = `${cls.label} ${t.style}`;
        const cap = document.createElement('span');
        cap.textContent = t.style === 'handwritten' ? 'Tulis tangan' : t.style === 'printed_tps' ? 'Cetak' : 'Negatif';
        wrap.append(im, cap);
        thumbsRow.appendChild(wrap);
      });

      const statsEl = document.createElement('div');
      statsEl.className = 'proto-card__stats';
      const nTr = cls.stats?.n_train ?? '—';
      const mCos = cls.stats?.mean_cos !== undefined ? Number(cls.stats.mean_cos).toFixed(2) : '—';
      statsEl.textContent = `Sampel train: ${nTr} · mean cos ${mCos}`;

      card.append(top, thumbsRow, statsEl);
      grid.appendChild(card);
    });
  }

  function initProtoExplorerUI() {
    $$('.proto-filter-pill').forEach((btn) => {
      btn.addEventListener('click', () => {
        $$('.proto-filter-pill').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        protoCategory = btn.dataset.pcat || 'all';
        renderProtoBank();
      });
    });
    if ($('#protoSearchInput')) {
      $('#protoSearchInput').addEventListener('input', renderProtoBank);
    }
  }

  // ==========================================================================
  // INIT
  // ==========================================================================
  window.addEventListener('DOMContentLoaded', () => {
    initInspectorHooks();
    initBatchUI();
    initHistoryUI();
    initProtoExplorerUI();
  });

  window.AksaraV4 = {
    renderInspector,
    queueBatchFiles,
    renderProtoBank,
    renderRecentScans,
  };
})();
