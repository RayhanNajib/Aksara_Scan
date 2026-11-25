/**
 * Peningkatan Antarmuka AksaraScan v3.0:
 *  - Garis bawah putus-putus pada kata OOV/ragu dengan menu klik-untuk-tukar top-K
 *  - Pencatatan koreksi lokal ke localStorage + ekspor JSON lokal
 *  - Toggle opsional MT daring opt-in (baku mati)
 *  - Sepenuhnya selaras token tema Terang & Gelap (tanpa lencana Ortografi Valid / Mentah detektor)
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
      .v3-word-uncertain {
        border-bottom: 2px dotted var(--prada);
        cursor: pointer;
        padding: 0 2px;
        border-radius: 3px;
        transition: background-color 0.15s;
      }
      .v3-word-uncertain:hover {
        background: var(--prada-glow);
      }
      .v3-controls-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid var(--border-dim);
        font-size: 12px;
        color: var(--text-muted);
      }
      .v3-controls-row input[type="checkbox"] {
        accent-color: var(--prada);
      }
      .v3-btn-mini {
        padding: 5px 10px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-dim);
        background: var(--bg-surface);
        color: var(--text-main);
        cursor: pointer;
        font-weight: 600;
        font-size: 11.5px;
        transition: border-color 0.15s, background-color 0.15s;
      }
      .v3-btn-mini:hover {
        border-color: var(--prada);
        background: var(--bg-panel-hover);
      }
      .v3-swap-popover {
        position: fixed;
        z-index: 9999;
        background: var(--popover-bg);
        color: var(--popover-text);
        border: 1px solid var(--popover-border);
        border-radius: var(--radius-md);
        padding: 10px;
        box-shadow: var(--shadow-card);
        font-size: 12.5px;
        min-width: 205px;
      }
      .v3-swap-popover button {
        display: block;
        width: 100%;
        text-align: left;
        padding: 6px 9px;
        margin-top: 5px;
        border: 1px solid var(--border-dim);
        border-radius: var(--radius-sm);
        background: var(--popover-item);
        color: var(--popover-text);
        cursor: pointer;
        font-size: 12px;
      }
      .v3-swap-popover button:hover {
        border-color: var(--prada);
      }
    `;
    document.head.appendChild(st);
  }

  function ensureMetaContainers() {
    injectStyles();
    const outIndo = document.getElementById('outIndonesian');
    if (outIndo && !document.getElementById('v3FooterControls')) {
      const ctrl = document.createElement('div');
      ctrl.id = 'v3FooterControls';
      ctrl.className = 'v3-controls-row';
      ctrl.innerHTML = `
        <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;" title="Baku: MATI. Aktifkan hanya bila ingin menerjemahkan lewat layanan daring.">
          <input type="checkbox" id="chkOptInOnlineMT" />
          <span>Terjemahan Daring Opsional</span>
        </label>
        <button type="button" id="btnExportV3Corrections" class="v3-btn-mini">
          Ekspor Koreksi (<span id="v3CorrectionCount">0</span>)
        </button>
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
            outIndo.textContent = `${tr} (MT daring)`;
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
    pop.style.left = `${Math.max(8, Math.min(window.innerWidth - 225, rect.left))}px`;
    pop.style.top = `${Math.min(window.innerHeight - 190, rect.bottom + 6)}px`;

    const title = document.createElement('div');
    title.style.fontWeight = '700';
    title.textContent = `Perbaiki "${wordMeta.word}"`;
    pop.appendChild(title);

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
        b.textContent = `Tukar ${det.label} → ${alt.label} (${(alt.prob * 100).toFixed(0)}%)`;
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
    customBtn.textContent = 'Ketik koreksi manual…';
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

    const outLat = document.getElementById('outSentenceLatin');
    if (outLat && Array.isArray(res.wordDetails) && res.wordDetails.length > 0 && res.sentenceLatin) {
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
          sp.title = 'Klik untuk meninjau atau memperbaiki kata ini';
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
      btnYolo.innerHTML = btnYolo.innerHTML.replace(/52\s*Kelas/gi, '55 Kelas');
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    updateVersionCopy();
    ensureMetaContainers();
  });

  window.addEventListener('aksara:reconstructed', (ev) => {
    setTimeout(() => renderV3Annotations(ev.detail || window.__aksaraLastOrthoResult), 0);
  });
})();
