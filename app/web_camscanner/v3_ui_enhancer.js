/**
 * Peningkatan Antarmuka AksaraScan v3.0 (Fase 4g):
 *  - Menampilkan transkripsi Unicode mentah (top-1 detektor) di samping hasil Dekoder Beam + LM
 *  - Garis bawah putus-putus pada kata berkeyakinan rendah / OOV dengan menu klik-untuk-tukar top-K
 *  - Pencatatan koreksi lokal ke localStorage + ekspor JSON lokal untuk fine-tuning putaran berikutnya
 *  - Toggle opsional MT daring opt-in (mati secara baku, tanpa unggahan citra ke peladen)
 */
(function () {
  'use strict';

  const STORAGE_KEY = 'aksarascan_v3_corrections_v1';

  function loadCorrections() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (_e) {
      return [];
    }
  }

  function saveCorrectionEntry(entry) {
    try {
      const arr = loadCorrections();
      arr.push({ ...entry, savedAt: new Date().toISOString() });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(arr.slice(-300)));
      updateExportBadge(arr.length);
    } catch (_e) { /* quota ignore */ }
  }

  function updateExportBadge(count) {
    const badge = document.getElementById('v3CorrectionCount');
    if (badge) badge.textContent = String(count);
  }

  function injectStyles() {
    if (document.getElementById('aksaraV3Styles')) return;
    const st = document.createElement('style');
    st.id = 'aksaraV3Styles';
    st.textContent = `
      .v3-meta-bar {
        display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
        margin-top: 8px; padding: 8px 10px; border-radius: 8px;
        background: rgba(18, 25, 38, 0.05); border: 1px solid rgba(0, 0, 0, 0.08);
        font-size: 0.82rem; color: inherit;
      }
      body.dark-mode .v3-meta-bar {
        background: rgba(255, 255, 255, 0.05); border-color: rgba(255, 255, 255, 0.12);
      }
      .v3-pill {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 2px 8px; border-radius: 999px; font-weight: 600; font-size: 0.75rem;
        background: #e2e8f0; color: #1e293b;
      }
      .v3-pill.ok { background: #dcfce7; color: #166534; }
      .v3-pill.warn { background: #fef3c7; color: #92400e; }
      .v3-raw-line { font-family: 'Noto Sans Javanese', serif; opacity: 0.85; word-break: break-word; }
      .v3-word-uncertain {
        border-bottom: 2px dotted #d97706; cursor: pointer; padding: 0 2px; border-radius: 3px;
        transition: background 0.15s;
      }
      .v3-word-uncertain:hover { background: rgba(217, 119, 6, 0.14); }
      .v3-controls-row {
        display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between;
        margin-top: 10px; font-size: 0.8rem;
      }
      .v3-btn-mini {
        padding: 5px 10px; border-radius: 6px; border: 1px solid rgba(100,116,139,0.4);
        background: transparent; color: inherit; cursor: pointer; font-weight: 600; font-size: 0.78rem;
      }
      .v3-btn-mini:hover { background: rgba(100,116,139,0.12); }
      .v3-swap-popover {
        position: fixed; z-index: 9999; background: #1e293b; color: #f8fafc;
        border: 1px solid #475569; border-radius: 8px; padding: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        font-size: 0.82rem; min-width: 190px;
      }
      .v3-swap-popover button {
        display: block; width: 100%; text-align: left; padding: 5px 8px; margin-top: 4px;
        border: 0; border-radius: 4px; background: #334155; color: #f8fafc; cursor: pointer;
      }
      .v3-swap-popover button:hover { background: #475569; }
    `;
    document.head.appendChild(st);
  }

  function ensureMetaContainers() {
    injectStyles();
    const outJav = document.getElementById('outJavanese');
    if (outJav && !document.getElementById('v3RawMetaBar')) {
      const bar = document.createElement('div');
      bar.id = 'v3RawMetaBar';
      bar.className = 'v3-meta-bar';
      bar.style.display = 'none';
      bar.innerHTML = `
        <span id="v3OrthoBadge" class="v3-pill ok">Ortografi Valid</span>
        <span id="v3BeamBadge" class="v3-pill" style="display:none">Koreksi Beam + LM</span>
        <span style="opacity:0.75">Mentah detektor:</span>
        <span id="outRawJavanese" class="v3-raw-line"></span>
      `;
      outJav.parentNode.insertBefore(bar, outJav.nextSibling);
    }

    const outIndo = document.getElementById('outIndonesian');
    if (outIndo && !document.getElementById('v3FooterControls')) {
      const ctrl = document.createElement('div');
      ctrl.id = 'v3FooterControls';
      ctrl.className = 'v3-controls-row';
      ctrl.innerHTML = `
        <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;" title="Baku: MATI. Hanya aktifkan jika Anda mengizinkan teks terdeteksi dikirim ke layanan terjemahan daring.">
          <input type="checkbox" id="chkOptInOnlineMT" />
          <span>Terjemahan Daring Opsional (opt-in, baku mati)</span>
        </label>
        <div style="display:flex;gap:8px;align-items:center;">
          <button type="button" id="btnExportV3Corrections" class="v3-btn-mini">
            Ekspor Koreksi Lokal (<span id="v3CorrectionCount">0</span>)
          </button>
        </div>
      `;
      outIndo.parentNode.insertBefore(ctrl, outIndo.nextSibling);
      updateExportBadge(loadCorrections().length);

      const expBtn = document.getElementById('btnExportV3Corrections');
      if (expBtn) {
        expBtn.addEventListener('click', () => {
          const items = loadCorrections();
          const blob = new Blob([JSON.stringify({ version: '3.0', exportedAt: new Date().toISOString(), items }, null, 2)], { type: 'application/json' });
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = `aksarascan_corrections_${Date.now()}.json`;
          a.click();
          URL.revokeObjectURL(a.href);
        });
      }

      const mtChk = document.getElementById('chkOptInOnlineMT');
      if (mtChk) {
        mtChk.addEventListener('change', async () => {
          if (!mtChk.checked) return;
          const res = window.__aksaraLastOrthoResult;
          if (!res || !res.sentenceLatin || !window.AksaraLexicon || !window.AksaraLexicon.optionalOnlineTranslate) return;
          const tr = await window.AksaraLexicon.optionalOnlineTranslate(res.sentenceLatin, true);
          if (tr) {
            outIndo.textContent = `${tr} (MT daring opt-in)`;
          }
        });
      }
    }
  }

  function closePopover() {
    const old = document.getElementById('v3SwapPopover');
    if (old) old.remove();
  }

  function showWordSwapPopover(anchorEl, wordMeta, res) {
    closePopover();
    const pop = document.createElement('div');
    pop.id = 'v3SwapPopover';
    pop.className = 'v3-swap-popover';
    const rect = anchorEl.getBoundingClientRect();
    pop.style.left = `${Math.max(8, Math.min(window.innerWidth - 220, rect.left))}px`;
    pop.style.top = `${Math.min(window.innerHeight - 180, rect.bottom + 6)}px`;

    const title = document.createElement('div');
    title.style.fontWeight = '700';
    title.textContent = `Perbaiki "${wordMeta.word}"`;
    pop.appendChild(title);

    // Ambil alternatif top-K dari deteksi yang relevan
    const dets = window.__aksaraLastKeptDetections || [];
    const topAlts = [];
    dets.forEach((d) => {
      if (d.topK && d.topK.length > 1 && d.confidence < 0.72) {
        d.topK.slice(0, 3).forEach((tk) => {
          if (tk.label !== d.label) {
            topAlts.push({ det: d, alt: tk });
          }
        });
      }
    });

    if (topAlts.length > 0) {
      topAlts.slice(0, 3).forEach(({ det, alt }) => {
        const b = document.createElement('button');
        b.textContent = `Tukar ${det.label} -> ${alt.label} (${(alt.prob * 100).toFixed(0)}%)`;
        b.addEventListener('click', () => {
          const prevLabel = det.label;
          det.label = alt.label;
          det.category = alt.category;
          det.confidence = Math.max(0.8, alt.prob);
          saveCorrectionEntry({ type: 'topk_swap', from: prevLabel, to: alt.label, rawWord: wordMeta.word });
          closePopover();
          if (window.AksaraOrthography) {
            const updated = window.AksaraOrthography.reconstructJavaneseText(dets);
            const oj = document.getElementById('outJavanese');
            const ol = document.getElementById('outSentenceLatin');
            const oi = document.getElementById('outIndonesian');
            if (oj) oj.textContent = updated.unicodeText;
            if (ol) ol.textContent = updated.sentenceLatin;
            if (oi) oi.textContent = updated.sentenceIndonesian;
          }
        });
        pop.appendChild(b);
      });
    }

    const customBtn = document.createElement('button');
    customBtn.textContent = 'Ketik koreksi kata manual...';
    customBtn.addEventListener('click', () => {
      closePopover();
      const edited = window.prompt(`Koreksi kata "${wordMeta.word}":`, wordMeta.word);
      if (edited && edited.trim() && edited.trim() !== wordMeta.word) {
        saveCorrectionEntry({ type: 'manual_word', from: wordMeta.word, to: edited.trim(), unicode: res.unicodeText });
        anchorEl.textContent = edited.trim();
        anchorEl.classList.remove('v3-word-uncertain');
        const ol = document.getElementById('outSentenceLatin');
        const oi = document.getElementById('outIndonesian');
        if (ol && oi && window.AksaraLexicon && window.AksaraLexicon.translateJavaneseTokens) {
          oi.textContent = window.AksaraLexicon.translateJavaneseTokens(ol.textContent);
        }
      }
    });
    pop.appendChild(customBtn);

    document.body.appendChild(pop);
    setTimeout(() => {
      document.addEventListener('click', function handler(ev) {
        if (!pop.contains(ev.target) && ev.target !== anchorEl) {
          closePopover();
          document.removeEventListener('click', handler);
        }
      });
    }, 0);
  }

  function renderV3Annotations(res) {
    ensureMetaContainers();
    if (!res) return;

    const bar = document.getElementById('v3RawMetaBar');
    const rawSpan = document.getElementById('outRawJavanese');
    const orthoBadge = document.getElementById('v3OrthoBadge');
    const beamBadge = document.getElementById('v3BeamBadge');
    if (bar && rawSpan && orthoBadge && beamBadge) {
      bar.style.display = res.unicodeText ? 'flex' : 'none';
      rawSpan.textContent = res.rawUnicodeText || res.unicodeText || '';
      orthoBadge.className = `v3-pill ${res.orthoValid ? 'ok' : 'warn'}`;
      orthoBadge.textContent = res.orthoValid ? 'Ortografi Valid' : `Catatan Ortografi (${(res.orthoErrors || []).length})`;
      beamBadge.style.display = res.beamUsed ? 'inline-flex' : 'none';
    }

    const outLat = document.getElementById('outSentenceLatin');
    if (outLat && Array.isArray(res.wordDetails) && res.wordDetails.length > 0 && res.sentenceLatin) {
      // Beri garis bawah putus-putus HANYA pada kata OOV agar pengguna tahu mana yang perlu ditinjau
      const metaByClean = new Map();
      res.wordDetails.forEach((wm) => metaByClean.set(wm.word, wm));

      const parts = res.sentenceLatin.split(/(\s+)/);
      outLat.innerHTML = '';
      parts.forEach((part) => {
        if (/^\s+$/.test(part)) {
          outLat.appendChild(document.createTextNode(part));
          return;
        }
        const clean = part.toLowerCase().replace(/^[.,;:!?\"'()]+|[.,;:!?\"'()]+$/g, '');
        const wm = metaByClean.get(clean);
        if (wm && !wm.known && clean.length >= 2) {
          const sp = document.createElement('span');
          sp.className = 'v3-word-uncertain';
          sp.title = 'Kata tidak ditemukan di kamus/korpus Jawa — klik untuk tinjau atau perbaiki';
          sp.textContent = part;
          sp.addEventListener('click', (e) => {
            e.stopPropagation();
            showWordSwapPopover(sp, wm, res);
          });
          outLat.appendChild(sp);
        } else {
          outLat.appendChild(document.createTextNode(part));
        }
      });
    }
  }

  function updateVersionCopy() {
    const btnYolo = document.getElementById('btnEngineYolo');
    if (btnYolo) {
      btnYolo.innerHTML = btnYolo.innerHTML
        .replace(/52\s*Kelas/gi, '55 Kelas')
        .replace(/v2\.5/gi, 'v3.0');
    }
    const subtitle = document.querySelector('.brand-sub, .subtitle, header p');
    if (subtitle && subtitle.textContent.includes('52 Kelas')) {
      subtitle.textContent = subtitle.textContent.replace(/52 Kelas/g, '55 Kelas');
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    updateVersionCopy();
    ensureMetaContainers();
  });

  window.addEventListener('aksara:reconstructed', (ev) => {
    // Tunda 0ms agar berjalan sesudah app.js selesai mengisi textContent DOM
    setTimeout(() => renderV3Annotations(ev.detail || window.__aksaraLastOrthoResult), 0);
  });
})();
