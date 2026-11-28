/**
 * AksaraScan — Pemindai Aksara Jawa Berbasis Web
 * Pipeline Citra Klasik: Grayscale -> Otsu -> Median -> Morfologi -> Proyeksi -> Normalisasi 96x96 -> SVM Linear
 * Kelompok 4 · PTIN236021 Universitas Negeri Malang
 * Penulis: Muhammad Rayhan Najib & M. Ikhsan Ridho A.
 */

// Mapping 20 Aksara Nglegena ke Unicode dan Latin
const AKSARA_DATA = [
  { latin: 'ha', unicode: 'ꦲ', name: 'Ha' },
  { latin: 'na', unicode: 'ꦤ', name: 'Na' },
  { latin: 'ca', unicode: 'ꦕ', name: 'Ca' },
  { latin: 'ra', unicode: 'ꦫ', name: 'Ra' },
  { latin: 'ka', unicode: 'ꦏ', name: 'Ka' },
  { latin: 'da', unicode: 'ꦢ', name: 'Da' },
  { latin: 'ta', unicode: 'ꦠ', name: 'Ta' },
  { latin: 'sa', unicode: 'ꦱ', name: 'Sa' },
  { latin: 'wa', unicode: 'ꦮ', name: 'Wa' },
  { latin: 'la', unicode: 'ꦭ', name: 'La' },
  { latin: 'pa', unicode: 'ꦥ', name: 'Pa' },
  { latin: 'dha', unicode: 'ꦝ', name: 'Dha' },
  { latin: 'ja', unicode: 'ꦗ', name: 'Ja' },
  { latin: 'ya', unicode: 'ꦪ', name: 'Ya' },
  { latin: 'nya', unicode: 'ꦚ', name: 'Nya' },
  { latin: 'ma', unicode: 'ꦩ', name: 'Ma' },
  { latin: 'ga', unicode: 'ꦒ', name: 'Ga' },
  { latin: 'ba', unicode: 'ꦧ', name: 'Ba' },
  { latin: 'tha', unicode: 'ꦛ', name: 'Tha' },
  { latin: 'nga', unicode: 'ꦔ', name: 'Nga' }
];

const UNICODE_MAP = {};
AKSARA_DATA.forEach(a => { UNICODE_MAP[a.latin] = a.unicode; });

// State Aplikasi
let model = null;
let modelWeights = null;
let yoloMeta = null;
let yoloSession = null;
let activeEngine = 'svm'; // 'yolo' otomatis aktif bila model ONNX siap
let yoloReconstructed = null;
let sourceImage = null;
let currentDetections = [];
let currentThreshold = 128;
let manualThreshold = null; // digeser user -> override Otsu; reset tiap gambar baru
let selectedGlyphIndex = -1;
let lastSourceName = null;

// Cache Citra Antara untuk Lab Pipeline
const labStages = {
  original: null,
  grayscale: null,
  otsu: null,
  median: null,
  morphology: null,
  projection: null,
  histogram: null
};

// Elemen DOM
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

const fileInput = $('#fileInput');
const camInput = $('#camInput');
const dropzone = $('#dropzone');
const stageWrap = $('#stageWrap');
const mainCanvas = $('#mainCanvas');
const mainCtx = mainCanvas.getContext('2d', { willReadFrequently: true });
const btnRun = $('#btnRun');
const modelBadge = $('#modelBadge');
const progressBox = $('#progressBox');
const progressFill = $('#progressFill');
const progressText = $('#progressText');

// Fallback Float16 Decoder (Kompatibel dengan semua peramban)
function decodeFloat16(b0, b1) {
  const h = (b1 << 8) | b0;
  const s = (h >> 15) & 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function parseModelWeights(model) {
  const raw = atob(model.w_f16_b64);
  const total = raw.length / 2;
  const weights = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    const b0 = raw.charCodeAt(i * 2);
    const b1 = raw.charCodeAt(i * 2 + 1);
    weights[i] = decodeFloat16(b0, b1);
  }
  return weights;
}

// Inisialisasi Model & PWA
async function initApp() {
  try {
    if (window.AKSARA_MODEL) {
      model = window.AKSARA_MODEL;
    } else {
      const res = await fetch('./model.json');
      if (!res.ok) throw new Error('Gagal memuat model.json');
      model = await res.json();
    }
    modelWeights = parseModelWeights(model);
    modelBadge.textContent = `SVM Siap · ${model.accuracy}% Uji`;
    modelBadge.style.color = 'var(--lontar)';
  } catch (err) {
    console.error(err);
    modelBadge.textContent = 'Model Offline';
    modelBadge.style.color = 'var(--bata)';
  }

  // Registrasi Service Worker PWA
  if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
    navigator.serviceWorker.register('./sw.js').catch(e => console.log('SW reg error:', e));
  }

  // Muat YOLOv8-Nano ONNX secara paralel bila dibuka melalui HTTP(S)
  initYoloEngine();

  // Setup Pengalih Mesin
  const btnYolo = $('#btnEngineYolo');
  const btnSvm = $('#btnEngineSvm');
  if (btnYolo) btnYolo.addEventListener('click', () => setActiveEngine('yolo'));
  if (btnSvm) btnSvm.addEventListener('click', () => setActiveEngine('svm'));

  // Setup Tema
  const savedTheme = localStorage.getItem('aksarascan_theme') || 'dark';
  applyTheme(savedTheme);
  $('#themeToggle').addEventListener('click', toggleTheme);

  // Setup Navigasi Atas & Tab Kamus
  $$('[data-nav]').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      switchMainView(el.dataset.nav);
    });
  });
  $$('.kamus-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.kamus-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const cat = btn.dataset.kcat;
      if ($('#kamusNglegenaWrap')) $('#kamusNglegenaWrap').hidden = (cat !== 'nglegena');
      if ($('#kamusPasanganWrap')) $('#kamusPasanganWrap').hidden = (cat !== 'pasangan');
      if ($('#kamusSandhanganWrap')) $('#kamusSandhanganWrap').hidden = (cat !== 'sandhangan');
    });
  });

  // Setup Events
  // Kamera: mobile pakai input capture; desktop pakai MediaDevices getUserMedia.
  const hasCapture = camInput.capture !== undefined && camInput.capture !== '';
  $('#btnCam').addEventListener('click', () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia && !hasCapture) startDesktopCamera();
    else camInput.click();
  });
  $('#btnFile').addEventListener('click', () => fileInput.click());
  if ($('#btnReplaceImg')) $('#btnReplaceImg').addEventListener('click', () => fileInput.click());
  // Reset value agar file yang sama bisa diunggah ulang tanpa refresh.
  fileInput.addEventListener('change', e => {
    const f = e.target.files[0];
    if (f) handleImageUpload(f);
    e.target.value = '';
  });
  camInput.addEventListener('change', e => {
    const f = e.target.files[0];
    if (f) handleImageUpload(f);
    e.target.value = '';
  });

  if ($('#optConfRange')) {
    $('#optConfRange').addEventListener('input', e => {
      if ($('#optConfVal')) $('#optConfVal').textContent = parseFloat(e.target.value).toFixed(2);
    });
    $('#optConfRange').addEventListener('change', () => {
      if (sourceImage && activeEngine === 'yolo') executePipeline();
    });
  }

  initSelection();
  initPasteAndDrop();

  // Drag and drop & keyboard dropzone
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', e => { e.preventDefault(); dropzone.classList.remove('dragover'); });
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 1 && window.AksaraV4?.queueBatchFiles) {
      switchMainView('batch');
      window.AksaraV4.queueBatchFiles(e.dataTransfer.files);
    } else if (e.dataTransfer.files[0]) {
      handleImageUpload(e.dataTransfer.files[0]);
    }
  });
  dropzone.addEventListener('click', e => {
    if (!e.target.closest('button')) fileInput.click();
  });

  // Topbar Pindai Baru
  if ($('#btnNewScan')) {
    $('#btnNewScan').addEventListener('click', () => {
      if (sourceImage) resetToLanding();
      else {
        switchMainView('scanner');
        fileInput.click();
      }
    });
  }

  // Tombol aksi & toolbar kanvas
  btnRun.addEventListener('click', () => { clearSelection(); executePipeline(); });
  $('#showBoxes').addEventListener('change', redrawMainCanvas);
  $('#showBin').addEventListener('change', redrawMainCanvas);
  if ($('#toggleCompare')) $('#toggleCompare').addEventListener('change', redrawMainCanvas);
  if ($('#compareSplitRange')) $('#compareSplitRange').addEventListener('input', redrawMainCanvas);
  if ($('#btnRotate90')) $('#btnRotate90').addEventListener('click', rotateSourceImage90);
  if ($('#btnZoomIn')) $('#btnZoomIn').addEventListener('click', () => setCanvasZoom((window.__aksaraCanvasZoom || 1) + 0.2));
  if ($('#btnZoomOut')) $('#btnZoomOut').addEventListener('click', () => setCanvasZoom((window.__aksaraCanvasZoom || 1) - 0.2));

  // Pre-Scan Enhancer Controls terhubung langsung ke image_enhancer.js
  const syncEnhanceFromUI = (rerun = true) => {
    const autoSh = $('#enhAutoSharpen') ? $('#enhAutoSharpen').checked : true;
    const flatSh = $('#enhFlattenShadow') ? $('#enhFlattenShadow').checked : true;
    const remLine = $('#optRemoveLines') ? $('#optRemoveLines').checked : true;
    const deskew = $('#btnAutoDeskew') ? $('#btnAutoDeskew').checked : false;
    const shVal = $('#enhSharpenRange') ? parseFloat($('#enhSharpenRange').value) : -0.1;
    const ctVal = $('#enhContrastRange') ? parseFloat($('#enhContrastRange').value) : 0;

    if ($('#enhSharpenVal')) $('#enhSharpenVal').textContent = shVal < 0 ? 'Auto' : shVal.toFixed(2);
    if ($('#enhContrastVal')) $('#enhContrastVal').textContent = `${ctVal >= 0 ? '+' : ''}${Math.round(ctVal * 100)}%`;

    window.aksaraEnhanceParams = {
      ...window.aksaraEnhanceParams,
      autoSharpen: autoSh,
      sharpen: shVal >= 0 ? shVal : undefined,
      flattenShadow: flatSh ? undefined : false,
      removeLines: remLine ? 'auto' : false,
      autoDeskew: deskew,
      contrast: ctVal,
    };
    if (rerun && sourceImage && activeEngine === 'yolo') executePipeline();
  };

  ['#enhAutoSharpen', '#enhFlattenShadow', '#optRemoveLines', '#btnAutoDeskew'].forEach(sel => {
    if ($(sel)) $(sel).addEventListener('change', () => syncEnhanceFromUI(true));
  });
  ['#enhSharpenRange', '#enhContrastRange'].forEach(sel => {
    if ($(sel)) {
      $(sel).addEventListener('input', () => syncEnhanceFromUI(false));
      $(sel).addEventListener('change', () => syncEnhanceFromUI(true));
    }
  });

  if ($('#btnRecoverSharpen')) {
    $('#btnRecoverSharpen').addEventListener('click', () => {
      if ($('#enhAutoSharpen')) $('#enhAutoSharpen').checked = true;
      if ($('#enhSharpenRange')) $('#enhSharpenRange').value = '0.65';
      syncEnhanceFromUI(true);
    });
  }
  if ($('#btnRecoverLowerConf')) {
    $('#btnRecoverLowerConf').addEventListener('click', () => {
      if ($('#optConfRange')) {
        $('#optConfRange').value = '0.18';
        if ($('#optConfVal')) $('#optConfVal').textContent = '0.18';
      }
      if (sourceImage) executePipeline();
    });
  }

  $('#btnCopyUnicode').addEventListener('click', () => copyToClipboard($('#outJavanese').textContent, 'Aksara Jawa disalin!'));
  $('#btnCopyLatin').addEventListener('click', () => {
    const sent = $('#outSentenceLatin')?.textContent;
    const raw = $('#outLatin').textContent;
    copyToClipboard(sent && sent !== '—' ? sent : raw, 'Kalimat Latin disalin!');
  });
  if ($('#btnDownloadTxt')) $('#btnDownloadTxt').addEventListener('click', () => downloadTextFile(buildExportText('txt'), `AksaraScan_${Date.now()}.txt`));
  if ($('#btnDownloadMd')) $('#btnDownloadMd').addEventListener('click', () => downloadTextFile(buildExportText('md'), `AksaraScan_${Date.now()}.md`, 'text/markdown;charset=utf-8'));
  $('#btnDownload').addEventListener('click', downloadJson);

  // Klik langsung pada kanvas untuk membuka Inspektor Referensi
  mainCanvas.addEventListener('click', handleCanvasClick);

  // Demo Card Coba 1-Klik
  $$('.demo-card').forEach(card => {
    const trigger = (e) => {
      e.preventDefault();
      switchMainView('scanner');
      loadDemoImage(card.dataset.src);
    };
    card.addEventListener('click', trigger);
  });

  // Sub-tab dalam Lab & Referensi
  $$('[data-labtab]').forEach(btn => {
    btn.addEventListener('click', () => switchLabSubTab(btn.dataset.labtab));
  });

  // Setup Kamus Aksara & Inspektor Referensi Modal
  renderDictionary();
  setupCorrectionModal();
}

function applyTheme(theme) {
  const mode = theme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', mode);
  localStorage.setItem('aksarascan_theme', mode);
  const metaTheme = $('#metaThemeColor');
  if (metaTheme) metaTheme.setAttribute('content', mode === 'light' ? '#f6efe0' : '#0e1422');
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'light' ? 'dark' : 'light');
}

function switchLabSubTab(subKey) {
  const target = subKey || 'pcd';
  $$('[data-labtab]').forEach(b => b.classList.toggle('active', b.dataset.labtab === target));
  $$('[data-labpanel]').forEach(p => { p.hidden = (p.dataset.labpanel !== target); });
  if (target === 'pcd') {
    const activeStage = document.querySelector('.lab-stage-btn.active')?.dataset.stage || 'otsu';
    renderLabStage(activeStage);
  } else if (target === 'proto' && window.AksaraV4?.renderProtoBank) {
    window.AksaraV4.renderProtoBank();
  }
}

function switchMainView(viewKey) {
  let effectiveView = viewKey;
  if (viewKey === 'kamus') {
    effectiveView = 'lab';
  }
  const views = {
    scanner: $('#viewScanner'),
    batch: $('#viewBatch'),
    lab: $('#viewLab'),
    about: $('#viewAbout')
  };
  if (!views[effectiveView]) return;
  Object.entries(views).forEach(([k, el]) => {
    if (el) el.hidden = (k !== effectiveView);
  });
  $$('.nav-pill').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.nav === effectiveView);
  });
  if (effectiveView === 'lab') {
    switchLabSubTab(viewKey === 'kamus' ? 'kamus' : (document.querySelector('[data-labtab].active')?.dataset.labtab || 'pcd'));
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Muat Citra ke Kanvas
function handleImageUpload(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    loadImageElement(img, file.name);
  };
  img.src = url;
}

// Kamera desktop via HTML5 MediaDevices API: pratinjau live + tangkap frame.
// Di mobile, jalur ini dilewati dan input capture="environment" dipakai.
function startDesktopCamera() {
  if (document.getElementById('camDialog')) return;
  const dlg = document.createElement('div');
  dlg.id = 'camDialog';
  dlg.style.cssText = 'position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.82);display:flex;align-items:center;justify-content:center;padding:16px';
  dlg.innerHTML = `
    <div class="card" style="max-width:720px;width:100%;padding:14px">
      <video id="camVideo" autoplay playsinline style="width:100%;border-radius:6px;background:#000"></video>
      <div style="display:flex;gap:8px;margin-top:10px;justify-content:center">
        <button class="btn btn--primary" id="camSnap">Tangkap</button>
        <button class="btn" id="camCancel">Batal</button>
      </div>
    </div>`;
  document.body.appendChild(dlg);
  let stream = null;
  const close = () => { if (stream) stream.getTracks().forEach(t => t.stop()); dlg.remove(); };
  navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
    .then(s => {
      stream = s;
      dlg.querySelector('#camVideo').srcObject = s;
      dlg.querySelector('#camSnap').addEventListener('click', () => {
        const v = dlg.querySelector('#camVideo');
        // Rata-ratakan 6 frame: exposure/autofokus kamera berfluktuasi antar frame;
        // satu frame acak = hasil scan berbeda-beda saat ulang. Rata-rata menstabilkan.
        const fw = v.videoWidth, fh = v.videoHeight;
        const fc = document.createElement('canvas');
        fc.width = fw; fc.height = fh;
        const fctx = fc.getContext('2d', { willReadFrequently: true });
        const acc = new Float64Array(fw * fh * 3);
        let grabbed = 0;
        const N = 6;
        const grabOne = () => {
          fctx.drawImage(v, 0, 0);
          const d = fctx.getImageData(0, 0, fw, fh).data;
          for (let i = 0, p3 = 0; i < d.length; i += 4, p3 += 3) {
            acc[p3] += d[i]; acc[p3 + 1] += d[i + 1]; acc[p3 + 2] += d[i + 2];
          }
          if (++grabbed < N) requestAnimationFrame(grabOne);
          else {
            const out = document.createElement('canvas');
            out.width = fw; out.height = fh;
            const octx = out.getContext('2d');
            const oimg = octx.createImageData(fw, fh);
            const od = oimg.data;
            for (let i = 0, p3 = 0; i < od.length; i += 4, p3 += 3) {
              od[i] = acc[p3] / N; od[i + 1] = acc[p3 + 1] / N; od[i + 2] = acc[p3 + 2] / N; od[i + 3] = 255;
            }
            octx.putImageData(oimg, 0, 0);
            close();
            const img = new Image();
            img.onload = () => loadImageElement(img, 'kamera.jpg');
            img.src = out.toDataURL('image/jpeg', 0.92);
          }
        };
        grabOne();
      });
      dlg.querySelector('#camCancel').addEventListener('click', close);
    })
    .catch(() => { close(); camInput.click(); });
}

function loadDemoImage(src) {
  modelBadge.textContent = 'Memuat Sampel...';
  const img = new Image();
  img.onload = () => {
    // loadImageElement kini auto-run pipeline (dengan antri jika model belum siap).
    loadImageElement(img, src);
  };
  img.onerror = () => {
    alert('Gagal memuat berkas sampel: ' + src);
  };
  img.src = src;
}

function handleCanvasClick(e) {
  if (!currentDetections.length) return;
  const rect = mainCanvas.getBoundingClientRect();
  const scaleX = mainCanvas.width / rect.width;
  const scaleY = mainCanvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;

  for (let i = 0; i < currentDetections.length; i++) {
    const [x0, y0, x1, y1] = currentDetections[i].box;
    if (px >= x0 && px <= x1 && py >= y0 && py <= y1) {
      openCorrectionModal(i);
      return;
    }
  }
}

function setCanvasZoom(z) {
  const zClamped = Math.max(0.6, Math.min(2.4, Math.round(z * 10) / 10));
  window.__aksaraCanvasZoom = zClamped;
  mainCanvas.style.width = `${Math.round(zClamped * 100)}%`;
  mainCanvas.style.maxHeight = zClamped > 1 ? 'none' : '440px';
  const badge = $('#zoomBadge');
  if (badge) badge.textContent = `${Math.round(zClamped * 100)}%`;
}

function resetToLanding() {
  switchMainView('scanner');
  clearSelection();
  sourceImage = null;
  currentDetections = [];
  window.__aksaraLastEnhancedCanvas = null;
  if ($('#workbenchWrap')) $('#workbenchWrap').hidden = true;
  if ($('#scannerEmptyWrap')) $('#scannerEmptyWrap').hidden = false;
  if (dropzone) dropzone.hidden = false;
  if ($('#btnReplaceImg')) $('#btnReplaceImg').hidden = true;
  if ($('#scanEmptyAlert')) $('#scanEmptyAlert').hidden = true;
  btnRun.disabled = true;
  resetResults();
}

function rotateSourceImage90() {
  if (!sourceImage) return;
  const sw = sourceImage.naturalWidth || sourceImage.width;
  const sh = sourceImage.naturalHeight || sourceImage.height;
  const rc = document.createElement('canvas');
  rc.width = sh;
  rc.height = sw;
  const rctx = rc.getContext('2d');
  rctx.translate(sh, 0);
  rctx.rotate(Math.PI / 2);
  rctx.drawImage(sourceImage, 0, 0, sw, sh);
  const im = new Image();
  im.onload = () => loadImageElement(im, lastSourceName || 'diputar');
  im.src = rc.toDataURL('image/png');
}

function loadImageElement(img, label) {
  sourceImage = img;
  manualThreshold = null;
  window.__aksaraLastEnhancedCanvas = null;
  lastSourceName = typeof label === 'string' ? label : String(label || 'gambar');
  const maxDim = 1500;
  const nw = img.naturalWidth || img.width;
  const nh = img.naturalHeight || img.height;
  const scale = Math.min(1, maxDim / Math.max(nw, nh));
  mainCanvas.width = Math.round(nw * scale);
  mainCanvas.height = Math.round(nh * scale);
  mainCtx.drawImage(img, 0, 0, mainCanvas.width, mainCanvas.height);

  if ($('#scannerEmptyWrap')) $('#scannerEmptyWrap').hidden = true;
  if ($('#workbenchWrap')) $('#workbenchWrap').hidden = false;
  dropzone.hidden = true;
  stageWrap.hidden = false;
  if ($('#btnReplaceImg')) $('#btnReplaceImg').hidden = false;
  if ($('#toggleCompare')) $('#toggleCompare').checked = false;
  if ($('#compareDivider')) $('#compareDivider').hidden = true;
  if ($('#compareSplitRange')) $('#compareSplitRange').hidden = true;
  if ($('#scanEmptyAlert')) $('#scanEmptyAlert').hidden = true;
  setCanvasZoom(1.0);
  clearSelection();

  const scaledNote = scale < 1 ? ` · skala dari ${nw}×${nh}` : '';
  $('#srcDim').textContent = `${mainCanvas.width}×${mainCanvas.height} px${scaledNote}`;
  btnRun.disabled = false;
  currentDetections = [];
  resetResults();
  // Auto-deteksi: begitu gambar masuk langsung scan. Model belum siap -> tunggu lalu jalan sendiri.
  if (model || yoloSession) {
    setTimeout(() => { if (sourceImage === img) executePipeline(); }, 100);
  } else {
    const wait = setInterval(() => {
      if (model || yoloSession) { clearInterval(wait); if (sourceImage === img) executePipeline(); }
    }, 100);
  }
}

function resetResults() {
  $('#outJavanese').textContent = '—';
  $('#outLatin').textContent = '—';
  if ($('#outSentenceLatin')) $('#outSentenceLatin').textContent = '—';
  if ($('#outIndonesian')) $('#outIndonesian').textContent = '—';
  if ($('#glyphCountBadge')) $('#glyphCountBadge').textContent = '0';
  if ($('#stage2Summary')) $('#stage2Summary').textContent = '';
  const meterVal = $('#meterVal');
  if (meterVal) meterVal.textContent = '0%';
  const meterFill = $('#meterFill');
  if (meterFill) meterFill.style.width = '0%';
  const meterWrap = $('#meterWrap');
  if (meterWrap) meterWrap.hidden = true;
  $('#glyphsGrid').replaceChildren();
}

// ============================================================================
// PIPELINE PENGOLAHAN CITRA DIGITAL (8 TAHAPAN)
// ============================================================================

// Overlay loading berisi aksara Jawa yang berputar saat pipeline berjalan.
const AKSARA_RING = ['\uA780','\uA781','\uA782','\uA783','\uA784','\uA785','\uA786','\uA787',
  '\uA788','\uA789','\uA78A','\uA78B','\uA78C','\uA78D','\uA78E','\uA78F','\uA790','\uA791','\uA792','\uA793'];
let fxTimer = null;
function showScanFx(label) {
  const fx = document.getElementById('scanFx'), ring = document.getElementById('scanfxRing');
  if (!fx) return;
  fx.hidden = false;
  document.body.classList.add('scanning');
  let i = 0;
  ring.textContent = AKSARA_RING[0];
  clearInterval(fxTimer);
  fxTimer = setInterval(() => { ring.textContent = AKSARA_RING[++i % AKSARA_RING.length]; }, 90);
  if (label) document.getElementById('scanfxLabel').textContent = label;
}
function hideScanFx() {
  const fx = document.getElementById('scanFx');
  if (fx) fx.hidden = true;
  document.body.classList.remove('scanning');
  clearInterval(fxTimer);
}

function updateProgress(step, text) {
  if (progressBox) progressBox.hidden = false;
  if (progressText) progressText.textContent = text;
  if (progressFill) progressFill.style.width = `${(step / 8) * 100}%`;
}

// ============================================================================
// MESIN DETEKSI YOLOv8-NANO ONNX (NGLEGENA + PASANGAN + SANDHANGAN)
// ============================================================================

async function initYoloEngine() {
  const statusEl = $('#engineStatus');
  const btnYolo = $('#btnEngineYolo');
  if (typeof ort === 'undefined') {
    if (statusEl) statusEl.textContent = 'Offline · Mode SVM Aktif';
    if (btnYolo) btnYolo.disabled = true;
    return;
  }
  if (!location.protocol.startsWith('http')) {
    if (statusEl) statusEl.textContent = 'file:// · Jalankan HTTP untuk YOLOv8';
    if (btnYolo) btnYolo.disabled = true;
    return;
  }
  try {
    if (statusEl) statusEl.textContent = 'Memuat YOLOv8 ONNX…';
    const metaRes = await fetch('./yolo_meta.json');
    if (!metaRes.ok) throw new Error('yolo_meta.json belum tersedia');
    yoloMeta = await metaRes.json();

    ort.env.wasm.wasmPaths = new URL('./vendor/ort/', window.location.href).href;
    ort.env.wasm.numThreads = (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated)
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;
    yoloSession = await ort.InferenceSession.create('./aksara_yolo.onnx', {
      executionProviders: ['wasm']
    });

    // Muat Stage-2 Metric Verifier + Bank Prototipe (fallback aman bila gagal)
    let stage2Note = '';
    if (window.AksaraStage2) {
      const s2Res = await window.AksaraStage2.init({
        onnxPath: './aksara_embed.onnx',
        prototypesPath: './reference_prototypes.json',
        calibrationPath: './stage2_calibration.json',
      });
      if (s2Res.ok) {
        stage2Note = ` + Stage-2 (${s2Res.totalPrototypes} prototipe)`;
      } else {
        stage2Note = ' · Stage-1 Fallback (Stage-2 tidak tersedia)';
        if (typeof console !== 'undefined') {
          console.warn('Stage-2 tidak aktif, memakai detektor saja:', s2Res.fallbackReason);
        }
      }
    }

    const m50 = yoloMeta.metrics && yoloMeta.metrics.map50 ? `${yoloMeta.metrics.map50}% mAP50` : '55 Kelas';
    if (statusEl) statusEl.textContent = `YOLOv8 Siap · ${m50}${stage2Note}`;
    if (btnYolo) btnYolo.disabled = false;
    setActiveEngine('yolo', false);
    setupCorrectionModal();
    if (sourceImage && !btnRun.disabled) executePipeline();
  } catch (err) {
    console.warn('YOLOv8 ONNX fallback ke SVM:', err);
    if (statusEl) statusEl.textContent = 'SVM Aktif (ONNX opsional)';
    if (btnYolo) btnYolo.disabled = true;
  }
}

function updateModelBadge() {
  if (activeEngine === 'yolo' && yoloMeta) {
    const m50 = yoloMeta.metrics && yoloMeta.metrics.map50 ? `${yoloMeta.metrics.map50}% mAP50` : '55 Kelas';
    modelBadge.textContent = `YOLOv8 · ${m50}`;
    modelBadge.style.color = 'var(--accent-text)';
  } else if (model) {
    modelBadge.textContent = `SVM Siap · ${model.accuracy}% Uji`;
    modelBadge.style.color = 'var(--lontar)';
  }
}

function setActiveEngine(mode, reRun = true) {
  if (mode === 'yolo' && !yoloSession) {
    alert('Model YOLOv8 ONNX belum siap atau halaman dibuka tanpa HTTP server.');
    return;
  }
  activeEngine = mode;
  const btnYolo = $('#btnEngineYolo');
  const btnSvm = $('#btnEngineSvm');
  const legend = $('#yoloLegend');
  if (btnYolo) btnYolo.classList.toggle('active', mode === 'yolo');
  if (btnSvm) btnSvm.classList.toggle('active', mode === 'svm');
  if (legend) legend.hidden = (mode !== 'yolo');

  updateModelBadge();
  setupCorrectionModal();
  if (reRun && sourceImage) executePipeline();
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
    const areaC = Math.max(1, (cand.box[2] - cand.box[0]) * (cand.box[3] - cand.box[1]));
    for (const k of kept) {
      const ix0 = Math.max(cand.box[0], k.box[0]), iy0 = Math.max(cand.box[1], k.box[1]);
      const ix1 = Math.min(cand.box[2], k.box[2]), iy1 = Math.min(cand.box[3], k.box[3]);
      const iw = Math.max(0, ix1 - ix0), ih = Math.max(0, iy1 - iy0);
      const inter = iw * ih;
      if (inter <= 0) continue;
      const areaK = Math.max(1, (k.box[2] - k.box[0]) * (k.box[3] - k.box[1]));
      const iou = inter / Math.max(1, areaC + areaK - inter);
      const contain = inter / Math.min(areaC, areaK);

      // nglegena, pasangan, & pada tidak boleh bertumpuk di posisi sama; hanya sandhangan yang boleh overlap dengan induknya
      const sameExclusiveGroup = (cand.category !== 'sandhangan' && k.category !== 'sandhangan');
      const thr = (cand.category === k.category || sameExclusiveGroup) ? sameCatIou : crossCatIou;
      if (iou > thr) { suppress = true; break; }
      // Cegah fragmen kecil di dalam huruf lebar yang SAMA kelompok eksklusifnya (misal sub-lengkungan ra di dalam ha/na/ka)
      if ((cand.category === k.category || sameExclusiveGroup) && contain > 0.60) {
        suppress = true; break;
      }
    }
    if (!suppress) kept.push(cand);
  }
  return kept;
}

function estimatePaperBgCss(srcCanvasOrImg, w, h) {
  try {
    const tmp = document.createElement('canvas');
    const sw = 64, sh = 64;
    tmp.width = sw; tmp.height = sh;
    const tctx = tmp.getContext('2d', { willReadFrequently: true });
    tctx.drawImage(srcCanvasOrImg, 0, 0, sw, sh);
    const d = tctx.getImageData(0, 0, sw, sh).data;
    const rs = [], gs = [], bs = [];
    for (let i = 0; i < d.length; i += 4) {
      const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      if (lum > 140) {
        rs.push(d[i]); gs.push(d[i + 1]); bs.push(d[i + 2]);
      }
    }
    if (rs.length >= 40) {
      rs.sort((a, b) => a - b); gs.sort((a, b) => a - b); bs.sort((a, b) => a - b);
      const mid = Math.floor(rs.length * 0.65);
      return `rgb(${rs[mid]}, ${gs[mid]}, ${bs[mid]})`;
    }
  } catch (_e) { /* fallback */ }
  return 'rgb(248, 245, 238)';
}

window.aksaraEnhanceParams = window.aksaraEnhanceParams || {
  autoSharpen: true,
  removeLines: 'auto',
  autoDeskew: false,
  rotateDeg: 0,
  contrast: 0,
};

let enhancerWorkerInstance = null;
let enhancerReqSeq = 0;

function runEnhancerWorker(imageData, params) {
  return new Promise((resolve) => {
    const canWorker = typeof Worker !== 'undefined' && typeof location !== 'undefined' && location.protocol.startsWith('http');
    if (!canWorker) {
      if (window.AksaraEnhancer) resolve(window.AksaraEnhancer.enhance(imageData, params));
      else resolve({ imageData, report: { mode: 'fallback_raw' } });
      return;
    }
    try {
      if (!enhancerWorkerInstance) {
        enhancerWorkerInstance = new Worker('./enhancer_worker.js');
      }
      const reqId = ++enhancerReqSeq;
      const copyBuf = new Uint8ClampedArray(imageData.data).buffer;
      const onMsg = (ev) => {
        const d = ev.data || {};
        if (d.id !== reqId) return;
        enhancerWorkerInstance.removeEventListener('message', onMsg);
        enhancerWorkerInstance.removeEventListener('error', onErr);
        if (d.ok && d.buffer) {
          resolve({
            imageData: { data: new Uint8ClampedArray(d.buffer), width: d.width, height: d.height },
            report: d.report,
          });
        } else if (window.AksaraEnhancer) {
          try {
            resolve(window.AksaraEnhancer.enhance(imageData, params));
          } catch (err) {
            console.warn('Enhancer fallback:', err);
            resolve({ imageData, report: { mode: 'error_fallback' } });
          }
        } else {
          resolve({ imageData, report: { mode: 'fallback_raw' } });
        }
      };
      const onErr = () => {
        if (enhancerWorkerInstance) {
          enhancerWorkerInstance.removeEventListener('message', onMsg);
          enhancerWorkerInstance.removeEventListener('error', onErr);
          enhancerWorkerInstance = null;
        }
        if (window.AksaraEnhancer) {
          try {
            resolve(window.AksaraEnhancer.enhance(imageData, params));
          } catch (err) {
            console.warn('Enhancer fallback:', err);
            resolve({ imageData, report: { mode: 'error_fallback' } });
          }
        } else {
          resolve({ imageData, report: { mode: 'fallback_raw' } });
        }
      };
      enhancerWorkerInstance.addEventListener('message', onMsg);
      enhancerWorkerInstance.addEventListener('error', onErr);
      enhancerWorkerInstance.postMessage(
        { id: reqId, op: 'enhance', width: imageData.width, height: imageData.height, buffer: copyBuf, params },
        [copyBuf]
      );
    } catch (_e) {
      if (window.AksaraEnhancer) resolve(window.AksaraEnhancer.enhance(imageData, params));
      else resolve({ imageData, report: { mode: 'fallback_raw' } });
    }
  });
}

/**
 * Praproses adaptif non-destruktif berbasis Worker (`AksaraEnhancer`):
 * menggantikan early bailout & kurva pow(1.32) lama dengan estimasi blur,
 * perata pencahayaan Sauvola, pemisah tinta warna CIELAB, penekan garis buku/tembok,
 * serta penajaman sadar-blur + Richardson-Lucy ringan anti-ringing.
 */
function formatEnhanceTelemetry(report) {
  if (!report || !report.blurBefore) return 'Siap';
  const b = report.blurBefore;
  const parts = [`Blur: ${b.level}`];
  if (report.usedRL) parts.push('RL');
  if (report.usedClahe) parts.push('CLAHE');
  if (report.suppressedLinePixels > 0) parts.push(`Garis diredam`);
  if (Number.isFinite(report.elapsedMs)) parts.push(`${report.elapsedMs} ms`);
  return parts.join(' · ');
}

async function prepareYoloInputCanvas(srcEl, CW, CH, customParams, opts = {}) {
  try {
    const c = document.createElement('canvas');
    c.width = CW;
    c.height = CH;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(srcEl, 0, 0, CW, CH);
    const img = ctx.getImageData(0, 0, CW, CH);
    const params = customParams || window.aksaraEnhanceParams || {};
    const { imageData: outData, report } = await runEnhancerWorker(img, params);

    if (outData.width !== CW || outData.height !== CH) {
      c.width = outData.width;
      c.height = outData.height;
    }
    const targetImg = ctx.createImageData(outData.width, outData.height);
    targetImg.data.set(outData.data);
    ctx.putImageData(targetImg, 0, 0);

    if (!opts.silent) {
      window.__aksaraLastEnhanceReport = report;
      window.__aksaraLastEnhancedCanvas = c;
      const pill = $('#enhTelemetryPill');
      if (pill) pill.textContent = formatEnhanceTelemetry(report);
    }
    c.__aksaraEnhanceReport = report;
    return c;
  } catch (_e) {
    return srcEl;
  }
}

async function runYoloInferenceCore(CW, CH, rx, ry, w, h, sel, morphBin, yoloInputEl, opts = {}) {
  const silent = Boolean(opts.silent);
  const rawSrcEl = yoloInputEl || sourceImage;
  const srcEl = await prepareYoloInputCanvas(rawSrcEl, CW, CH, window.aksaraEnhanceParams, { silent });
  const enhanceReport = srcEl.__aksaraEnhanceReport || window.__aksaraLastEnhanceReport || null;
  const srcW = srcEl.naturalWidth || srcEl.width || CW;
  const srcH = srcEl.naturalHeight || srcEl.height || CH;
  const imgsz = yoloMeta.imgsz || 416;
  const off = document.createElement('canvas');
  off.width = imgsz; off.height = imgsz;
  const octx = off.getContext('2d', { willReadFrequently: true });
  const padBgCss = estimatePaperBgCss(srcEl, CW, CH);

  // Temukan kotak pembatas tinta bersih di ROI untuk membuang margin kosong besar
  let minX = rx + w, minY = ry + h, maxX = rx, maxY = ry;
  let inkFound = 0;
  for (let yy = ry; yy < ry + h; yy++) {
    const row = yy * CW;
    for (let xx = rx; xx < rx + w; xx++) {
      if (morphBin[row + xx]) {
        inkFound++;
        if (xx < minX) minX = xx;
        if (xx > maxX) maxX = xx;
        if (yy < minY) minY = yy;
        if (yy > maxY) maxY = yy;
      }
    }
  }

  let baseX0 = rx, baseY0 = ry, baseX1 = rx + w, baseY1 = ry + h;
  if (inkFound > 30) {
    const pad = 36;
    baseX0 = Math.max(0, minX - pad);
    baseY0 = Math.max(0, minY - pad);
    baseX1 = Math.min(CW, maxX + pad);
    baseY1 = Math.min(CH, maxY + pad);
  }

  const bwArea = Math.max(1, baseX1 - baseX0);
  const bhArea = Math.max(1, baseY1 - baseY0);
  const tiles = [];

  if (Math.max(bwArea, bhArea) <= 580 && (bwArea / Math.max(1, bhArea)) <= 4.6) {
    tiles.push([baseX0, baseY0, baseX1, baseY1]);
  } else {
    // Deteksi pita baris horizontal dari morphBin agar kalimat memanjang tetap beresolusi tinggi per jendela
    const lineBands = [];
    let lStart = -1;
    let emptyRun = 0;
    const minRowInk = Math.max(2, Math.round(bwArea * 0.004));
    for (let yy = baseY0; yy < baseY1; yy++) {
      let s = 0;
      const row = yy * CW;
      for (let xx = baseX0; xx < baseX1; xx++) s += morphBin[row + xx];
      if (s >= minRowInk) {
        if (lStart < 0) lStart = yy;
        emptyRun = 0;
      } else if (lStart >= 0) {
        emptyRun++;
        if (emptyRun > 10) {
          if (yy - emptyRun - lStart >= 14) lineBands.push([lStart, yy - emptyRun]);
          lStart = -1;
          emptyRun = 0;
        }
      }
    }
    if (lStart >= 0 && baseY1 - lStart >= 14) lineBands.push([lStart, baseY1]);

    const maxBandH = lineBands.length ? Math.max(...lineBands.map(b => b[1] - b[0])) : 0;
    if (lineBands.length && lineBands.length <= 8 && maxBandH <= 260) {
      // Jangan buang baris valid yang tingginya 25-40px (baris tanpa pasangan!)
      const activeBands = lineBands.filter(b => (b[1] - b[0]) >= 18);
      const chosenBands = activeBands.length ? activeBands : lineBands;

      // Untuk halaman berkarakter besar renggang (mis. grid tulisan tangan 1-4 baris),
      // sertakan jendela makro agar aksara lebar (ha/na/ka) tidak terpecah menjadi 'ra'.
      const sortedBandHs = chosenBands.map(b => b[1] - b[0]).sort((a, b) => a - b);
      const medianBandH = sortedBandHs.length ? sortedBandHs[Math.floor(sortedBandHs.length / 2)] : 0;
      if (medianBandH >= 82 && Math.max(bwArea, bhArea) <= 1550) {
        tiles.push([baseX0, baseY0, baseX1, baseY1]);
      }

      for (let bIdx = 0; bIdx < chosenBands.length; bIdx++) {
        const [ly0, ly1] = chosenBands[bIdx];
        const rawH = ly1 - ly0;
        const lh = Math.max(36, rawH);
        const padV = Math.min(28, Math.max(14, Math.round(lh * 0.28)));

        // Jangan biarkan padding meluber masuk ke pita baris atas/bawah (atau teks keterangan Latin tipis)
        let minTopBound = 0;
        for (const [allY0, allY1] of lineBands) {
          if (allY1 <= ly0) minTopBound = Math.max(minTopBound, Math.min(ly0 - 2, allY1 + 2));
        }
        let maxBotBound = CH;
        for (const [allY0, allY1] of lineBands) {
          if (allY0 >= ly1) { maxBotBound = Math.min(maxBotBound, Math.max(ly1 + 2, allY0 - 2)); break; }
        }

        const ty0 = Math.max(minTopBound, ly0 - padV);
        const ty1 = Math.min(maxBotBound, ly1 + padV);
        const bandH = Math.max(1, ty1 - ty0);
        // Karakter tinggi >= 96px diberi petak lebih lebar agar skala dalam imgsz 416px tetap ideal (~50-75px)
        const tileW = rawH >= 96
          ? Math.min(bwArea, Math.max(440, Math.round(rawH * 5.2)))
          : Math.min(440, Math.max(260, Math.round(bandH * 2.8)));
        if (bwArea <= tileW * 1.15) {
          tiles.push([baseX0, ty0, baseX1, ty1]);
        } else {
          const step = Math.round(tileW * 0.65);
          for (let tx = baseX0; tx < baseX1; ) {
            const txEnd = Math.min(baseX1, tx + tileW);
            const txStart = (txEnd === baseX1) ? Math.max(baseX0, txEnd - tileW) : tx;
            tiles.push([txStart, ty0, txEnd, ty1]);
            if (txEnd >= baseX1 || tiles.length >= 24) break;
            tx += step;
          }
        }
        if (tiles.length >= 24) break;
      }
    } else {
      if (Math.max(bwArea, bhArea) <= 1550) {
        tiles.push([baseX0, baseY0, baseX1, baseY1]);
      }
      const ts = 440;
      const step = Math.round(ts * 0.68);
      for (let ty = baseY0; ty < baseY1; ) {
        const tyEnd = Math.min(baseY1, ty + ts);
        const tyStart = (tyEnd === baseY1 && tyEnd - ty < ts * 0.5) ? Math.max(baseY0, tyEnd - ts) : ty;
        for (let tx = baseX0; tx < baseX1; ) {
          const txEnd = Math.min(baseX1, tx + ts);
          const txStart = (txEnd === baseX1 && txEnd - tx < ts * 0.5) ? Math.max(baseX0, txEnd - ts) : tx;
          tiles.push([txStart, tyStart, txEnd, tyEnd]);
          if (txEnd >= baseX1 || tiles.length >= 24) break;
          tx += step;
        }
        if (tyEnd >= baseY1 || tiles.length >= 24) break;
        ty += step;
      }
    }
  }

  const nc = yoloMeta.classes.length;
  const uiConfEl = document.getElementById('optConfRange');
  const confThr = uiConfEl ? (parseFloat(uiConfEl.value) || 0.25) : (yoloMeta.conf_thresh || 0.25);
  const inpName = yoloSession.inputNames[0];
  const plane = imgsz * imgsz;
  const chw = new Float32Array(3 * plane);
  const natScaleX = srcW / CW;
  const natScaleY = srcH / CH;
  const cands = [];

  for (let tIdx = 0; tIdx < tiles.length; tIdx++) {
    const [tx0, ty0, tx1, ty1] = tiles[tIdx];
    const tw = Math.max(1, tx1 - tx0), th = Math.max(1, ty1 - ty0);
    if (!silent) updateProgress(6, `Tahap 6 · Inferensi YOLOv8 ONNX (${tIdx + 1}/${tiles.length})…`);

    octx.fillStyle = tiles.length === 1 ? padBgCss : 'rgb(114, 114, 114)';
    octx.fillRect(0, 0, imgsz, imgsz);
    const scale = Math.min(imgsz / tw, imgsz / th);
    const nw = Math.round(tw * scale), nh = Math.round(th * scale);
    const padX = Math.floor((imgsz - nw) / 2), padY = Math.floor((imgsz - nh) / 2);

    octx.drawImage(srcEl, tx0 * natScaleX, ty0 * natScaleY, tw * natScaleX, th * natScaleY, padX, padY, nw, nh);
    const rgba = octx.getImageData(0, 0, imgsz, imgsz).data;
    for (let i = 0; i < plane; i++) {
      const p = i * 4;
      chw[i] = rgba[p] / 255.0;
      chw[plane + i] = rgba[p + 1] / 255.0;
      chw[2 * plane + i] = rgba[p + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', chw, [1, 3, imgsz, imgsz]);
    const results = await yoloSession.run({ [inpName]: tensor });
    const outTensor = results[yoloSession.outputNames[0]];
    const numAnchors = outTensor.dims[2];
    const raw = outTensor.data;

    const SANDHANGAN_SET = new Set(['wulu', 'suku', 'pepet', 'taling', 'tarung', 'layar', 'cecak', 'wignyan', 'pangkon', 'cakra', 'keret', 'pengkal']);
    const s2St = window.AksaraStage2 ? window.AksaraStage2.getStatus() : null;
    const entryConf = (s2St && s2St.ready && s2St.cal && s2St.cal.recover_s2_min <= 1.0)
      ? Math.min(confThr, s2St.cal.trigger_lo || 0.18)
      : confThr;

    for (let a = 0; a < numAnchors; a++) {
      let bestCls = -1, bestScore = entryConf;
      const topCands = [];
      for (let c = 0; c < nc; c++) {
        const sc = raw[(4 + c) * numAnchors + a];
        if (sc > bestScore) { bestScore = sc; bestCls = c; }
        if (sc >= 0.05) topCands.push({ c, sc });
      }
      if (bestCls < 0) continue;
      const detVec = new Float32Array(nc);
      for (let c = 0; c < nc; c++) detVec[c] = raw[(4 + c) * numAnchors + a];

      topCands.sort((p, q) => q.sc - p.sc);
      const topK = topCands.slice(0, 5).map((it) => {
        const l = yoloMeta.classes[it.c];
        return {
          classIdx: it.c,
          label: l,
          category: l.startsWith('p_') ? 'pasangan' : l.startsWith('pada_') ? 'pada' : SANDHANGAN_SET.has(l) ? 'sandhangan' : 'nglegena',
          prob: it.sc,
        };
      });

      const cx = raw[a], cy = raw[numAnchors + a], bw = raw[2 * numAnchors + a], bh = raw[3 * numAnchors + a];
      const x0 = Math.max(0, (cx - bw / 2 - padX) / scale + tx0);
      const y0 = Math.max(0, (cy - bh / 2 - padY) / scale + ty0);
      const x1 = Math.min(CW, (cx + bw / 2 - padX) / scale + tx0);
      const y1 = Math.min(CH, (cy + bh / 2 - padY) / scale + ty0);
      if (x1 - x0 < 3 || y1 - y0 < 3) continue;

      // Abaikan kotak terpotong di tepi irisan petak internal (petak tetangga menangkap glif utuh)
      if (tiles.length > 1) {
        if ((tx0 > baseX0 + 2 && x0 <= tx0 + 2) || (tx1 < baseX1 - 2 && x1 >= tx1 - 2)) continue;
      }

      if (sel) {
        const gx = (x0 + x1) / 2, gy = (y0 + y1) / 2;
        if (gx < sel.x0 || gx > sel.x1 || gy < sel.y0 || gy > sel.y1) continue;
      }

      const label = yoloMeta.classes[bestCls];
      const category = yoloMeta.categories[bestCls];
      cands.push({
        classIdx: bestCls,
        label,
        category,
        unicode: yoloMeta.unicode[label] || '',
        confidence: bestScore,
        isUncertain: bestScore < 0.45,
        topK,
        detVec,
        box: [Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1)]
      });
    }
  }

  let kept = yoloNms(cands, 0.42, 0.72);
  let stage2Stats = null;
  if (window.AksaraStage2 && window.AksaraStage2.getStatus().ready) {
    if (!silent) updateProgress(7, 'Tahap 7 · Verifikasi Stage-2 & Prototipe…');
    const s2Out = await window.AksaraStage2.verifyDetections(kept, srcEl, CW, CH, yoloMeta);
    kept = yoloNms(s2Out.detections, 0.42, 0.72);
    stage2Stats = s2Out.stats;
  } else {
    kept.forEach((d) => { delete d.detVec; });
    stage2Stats = {
      active: false,
      fallbackReason: window.AksaraStage2 ? window.AksaraStage2.getStatus().fallbackReason : 'unavailable',
    };
  }
  kept.sort((a, b) => (a.box[1] - b.box[1]) || (a.box[0] - b.box[0]));

  kept.forEach((det, idx) => {
    det.index = idx;
    const [bx0, by0, bx1, by1] = det.box;
    const bw = Math.max(1, bx1 - bx0), bh = Math.max(1, by1 - by0);
    const sz = model ? model.size : 96;
    const norm = new Float32Array(sz * sz);
    const sc = (sz - 16) / Math.max(bw, bh);
    const nw2 = Math.max(1, Math.round(bw * sc)), nh2 = Math.max(1, Math.round(bh * sc));
    const ox = Math.floor((sz - nw2) / 2), oy = Math.floor((sz - nh2) / 2);
    for (let yy = 0; yy < nh2; yy++) {
      const sy = Math.min(CH - 1, Math.max(0, by0 + Math.floor(yy / sc)));
      for (let xx = 0; xx < nw2; xx++) {
        const sx = Math.min(CW - 1, Math.max(0, bx0 + Math.floor(xx / sc)));
        norm[(oy + yy) * sz + ox + xx] = morphBin[sy * CW + sx] ? 1 : 0;
      }
    }
    det.normCanvas = norm;
  });

  const reconstructed = window.AksaraOrthography
    ? window.AksaraOrthography.reconstructJavaneseText(kept)
    : null;

  if (!silent) {
    window.__aksaraLastStage2Stats = stage2Stats;
    window.__aksaraLastKeptDetections = kept;
    currentDetections = kept;
    yoloReconstructed = reconstructed;
    cacheSvmMontage(kept, model ? model.size : 96);
  }
  return { kept, reconstructed, enhanceReport, stage2Stats };
}

async function executeYoloPipeline(CW, CH, rx, ry, w, h, sel, morphBin, yoloInputEl) {
  return runYoloInferenceCore(CW, CH, rx, ry, w, h, sel, morphBin, yoloInputEl, { silent: false });
}

function executePipeline(roi) {
  if (!sourceImage || (!model && !yoloSession)) return;
  btnRun.disabled = true;
  showScanFx(roi ? 'Memindai area pilihan…' : 'Menganalisis…');

  setTimeout(async () => {
    try {
      const CW = mainCanvas.width;
      const CH = mainCanvas.height;
      // Selalu mulai dari citra sumber BERSIH: draw overlay kotak/label run
      // sebelumnya tertinggal di mainCanvas dan akan ikut terbacanya sebagai tinta.
      mainCtx.drawImage(sourceImage, 0, 0, CW, CH);
      const imgData = mainCtx.getImageData(0, 0, CW, CH);

      // Seleksi area ala Google Lens: pipeline berjalan HANYA di dalam roi.
      let w = CW, h = CH, rx = 0, ry = 0, sel = null;
      if (roi) {
        // Simpan seleksi ASLI pemakai untuk memfilter hasil; area pemrosesan diperlebar
        // simetris supaya ambang lokal & morfologi punya konteks (ala Google Lens).
        sel = { x0: roi.x0, y0: roi.y0, x1: roi.x1, y1: roi.y1 };
        const MINROI = 160;
        const padx = Math.max(0, (MINROI - (roi.x1 - roi.x0)) / 2) + 12;
        const pady = Math.max(0, (MINROI - (roi.y1 - roi.y0)) / 2) + 12;
        rx = Math.max(0, Math.round(roi.x0 - padx));
        ry = Math.max(0, Math.round(roi.y0 - pady));
        const x1 = Math.min(CW, Math.round(roi.x1 + padx));
        const y1 = Math.min(CH, Math.round(roi.y1 + pady));
        w = x1 - rx; h = y1 - ry;
        if (w < 24 || h < 24) { rx = 0; ry = 0; w = CW; h = CH; roi = null; sel = null; }
      }

      // Tahap 1: Grayscale (ITU-R BT.601) + Deteksi Otomatis Latar Gelap (Dark Mode Rescue)
      const gray = new Uint8Array(w * h);
      const data = imgData.data;
      const lumHist = new Uint32Array(256);
      for (let y = 0; y < h; y++) {
        const srcRow = ((ry + y) * CW + rx) * 4;
        const dstRow = y * w;
        for (let x = 0; x < w; x++) {
          const i = srcRow + x * 4;
          const v = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
          gray[dstRow + x] = v;
          lumHist[v]++;
        }
      }

      // Hitung median luminans latar untuk mendeteksi gambar teks terang di atas latar gelap
      let cum = 0, medianLum = 255;
      const halfPx = (w * h) >> 1;
      for (let b = 0; b < 256; b++) {
        cum += lumHist[b];
        if (cum >= halfPx) { medianLum = b; break; }
      }
      let yoloInputEl = sourceImage;
      if (medianLum < 112) {
        for (let k = 0; k < gray.length; k++) gray[k] = 255 - gray[k];
        const invC = document.createElement('canvas');
        invC.width = CW; invC.height = CH;
        const ictx = invC.getContext('2d', { willReadFrequently: true });
        const invImg = ictx.createImageData(CW, CH);
        const dstD = invImg.data;
        for (let p = 0; p < data.length; p += 4) {
          dstD[p] = 255 - data[p];
          dstD[p + 1] = 255 - data[p + 1];
          dstD[p + 2] = 255 - data[p + 2];
          dstD[p + 3] = 255;
        }
        ictx.putImageData(invImg, 0, 0);
        yoloInputEl = invC;
      }
      cacheLabStage('grayscale', gray, w, h, 'gray');

      // Tahap 2: Binerisasi Otsu + koreksi pencahayaan lokal (integral image)
      updateProgress(2, 'Tahap 2: Binerisasi Ambang Batas...');
      const { threshold, hist } = calculateOtsu(gray, w, h);
      currentThreshold = threshold;
      if (manualThreshold == null) {
        $('#otsuValueDisplay').textContent = threshold;
        $('#otsuRange').value = threshold;
      } else {
        $('#otsuValueDisplay').textContent = manualThreshold + ' (manual)';
      }
      const binary = adaptiveBinary(gray, w, h, manualThreshold ?? threshold);
      // clearBorderNoise membuang komponen yang menyentuh tepi (bingkai meja gelap).
      // Di seleksi ROI, glif justru selalu menempel tepi -> lewati agar tidak terhapus.
      if (!roi) clearBorderNoise(binary, w, h);
      cacheLabStage('otsu', binary, w, h, 'bin');
      cacheHistogram(hist, threshold);

      // Tahap 3: Filter Median 3x3
      updateProgress(3, 'Tahap 3: Filter Spasial Median 3×3...');
      const medFilterOn = $('#optMedian').checked;
      const filtered = medFilterOn ? applyMedian3x3(binary, w, h) : binary;
      cacheLabStage('median', filtered, w, h, 'bin');

      // Tahap 4: Morfologi (Hapus Garis Buku & Closing)
      updateProgress(4, 'Tahap 4: Morfologi (Hapus Garis & Penutupan)...');
      const morphOn = $('#optMorph').checked;
      const morph = morphOn ? applyMorphology(filtered, w, h) : filtered;
      cacheLabStage('morphology', morph, w, h, 'bin');
      labStages.binary = roiFullBin(morph, CW, CH, rx, ry, w, h);

      if (activeEngine === 'yolo' && yoloSession) {
        yoloReconstructed = null;
        await executeYoloPipeline(CW, CH, rx, ry, w, h, sel, labStages.binary, yoloInputEl);
        const segForLab = currentDetections.map(d => ({
          box: [Math.max(0, d.box[0] - rx), Math.max(0, d.box[1] - ry), Math.min(w - 1, d.box[2] - rx), Math.min(h - 1, d.box[3] - ry)]
        }));
        cacheLabStage('segmentation', overlaySegmentation(morph, w, h, segForLab), w, h, 'bin');
      } else {
        yoloReconstructed = null;
        // Tahap 5 & 6: Segmentasi Baris & Karakter
        updateProgress(5, 'Tahap 5 & 6: Segmentasi Baris & Karakter (CCL)...');
        const segments = segmentDocument(morph, w, h, roi ? Math.max(CW, CH) : 0);
        cacheLabStage('segmentation', overlaySegmentation(morph, w, h, segments), w, h, 'bin');

        // Tahap 7 & 8: Normalisasi 96x96 & SVM Linear
        updateProgress(7, 'Tahap 7 & 8: Normalisasi Glif 96×96 & Klasifikasi SVM...');
        let detections = [];
        for (let i = 0; i < segments.length; i++) {
          const seg = segments[i];
          const norm = normCrop(seg.crop, seg.cw, seg.ch, model.size, model.inner);
          if (norm) {
            const smooth = model.blurSigma ? smoothFeatures(norm.out, model.size, model.blurSigma) : norm.out;
            const res = classifyGlif(smooth, modelWeights, model);
            detections.push({
              ...res,
              category: 'nglegena',
              box: seg.box,
              normCanvas: norm.out,
              index: i
            });
          }
        }

        if (rx || ry) {
          for (const d of detections) {
            d.box = [d.box[0] + rx, d.box[1] + ry, d.box[2] + rx, d.box[3] + ry];
          }
        }
        if (sel) {
          // Hanya glif yang PUSATNYA di dalam seleksi asli yang dilaporkan.
          detections = detections.filter(d => {
            const cx = (d.box[0] + d.box[2]) / 2, cy = (d.box[1] + d.box[3]) / 2;
            return cx >= sel.x0 && cx <= sel.x1 && cy >= sel.y0 && cy <= sel.y1;
          });
          detections.forEach((d, i) => { d.index = i; });
        }
        currentDetections = detections;
        cacheSvmMontage(detections, model.size);
      }

      // Selesai
      updateProgress(8, 'Selesai!');
      setTimeout(() => { progressBox.hidden = true; }, 300);
      hideScanFx();

      redrawMainCanvas();
      renderOutputs();
      if ($('#viewLab') && !$('#viewLab').hidden) {
        const activeStage = document.querySelector('.lab-stage-btn.active')?.dataset.stage || 'otsu';
        renderLabStage(activeStage);
      }
    } catch (err) {
      console.error(err);
      const alertEl = $('#scanEmptyAlert');
      const msgEl = $('#scanEmptyMsg');
      if (msgEl) msgEl.textContent = `Pemrosesan gagal: ${err.message}. Coba unggah ulang gambar atau segarkan halaman.`;
      if (alertEl) alertEl.hidden = false;
      progressBox.hidden = true;
      hideScanFx();
    } finally {
      btnRun.disabled = false;
      updateModelBadge();
    }
  }, 40);
}

// Salinan biner ROI ke kanvas penuh untuk tampilan hitam-putih & Lab.
function roiFullBin(morph, W, H, rx, ry, rw, rh) {
  if (!rx && !ry && rw === W && rh === H) return morph;
  const full = new Uint8Array(W * H);
  for (let y = 0; y < rh; y++) {
    full.set(morph.subarray(y * rw, y * rw + rw), (ry + y) * W + rx);
  }
  return full;
}

// ============================ SELEKSI AREA (ala Google Lens) =================
// Seret kotak pada citra -> "Pindai Area Pilihan" -> pipeline hanya di dalam
// seleksi: kontras lokal & ambang jadi lebih akurat untuk aksara kecil.
let selStart = null, selRect = null, selDrag = false;

function canvasPos(e) {
  const rect = mainCanvas.getBoundingClientRect();
  return {
    x: Math.min(mainCanvas.width, Math.max(0, (e.clientX - rect.left) * mainCanvas.width / rect.width)),
    y: Math.min(mainCanvas.height, Math.max(0, (e.clientY - rect.top) * mainCanvas.height / rect.height))
  };
}

function clearSelection() {
  selStart = selRect = null; selDrag = false;
  const ov = $('#selOverlay'), btn = $('#btnRescanSel');
  ov.hidden = true; btn.hidden = true;
}

function positionSel() {
  const ov = $('#selOverlay'), btn = $('#btnRescanSel');
  const rect = mainCanvas.getBoundingClientRect();
  const sx = rect.width / mainCanvas.width, sy = rect.height / mainCanvas.height;
  const [x0, y0, x1, y1] = selRect;
  ov.style.left = (x0 * sx) + 'px';
  ov.style.top = (y0 * sy) + 'px';
  ov.style.width = ((x1 - x0) * sx) + 'px';
  ov.style.height = ((y1 - y0) * sy) + 'px';
  ov.hidden = false;
  btn.style.left = (x0 * sx) + 'px';
  btn.style.top = (y1 * sy + 8) + 'px';
}

function initSelection() {
  mainCanvas.addEventListener('pointerdown', e => {
    if (!sourceImage || btnRun.disabled) return;
    e.preventDefault();
    mainCanvas.setPointerCapture(e.pointerId);
    selStart = canvasPos(e);
    selDrag = false;
  });
  mainCanvas.addEventListener('pointermove', e => {
    if (!selStart) return;
    const cur = canvasPos(e);
    if (!selDrag && Math.abs(cur.x - selStart.x) + Math.abs(cur.y - selStart.y) < 8) return;
    selDrag = true;
    selRect = [
      Math.min(selStart.x, cur.x), Math.min(selStart.y, cur.y),
      Math.max(selStart.x, cur.x), Math.max(selStart.y, cur.y)
    ];
    positionSel();
    $('#btnRescanSel').hidden = false;
  });
  const end = e => {
    if (!selStart) return;
    const wasDrag = selDrag;
    selStart = null;
    if (!wasDrag) {
      // klik: bila ada seleksi tersisa, bersihkan; bila klik glif -> modal koreksi
      if (selRect) { clearSelection(); return; }
      handleCanvasClick(e);
    }
    // bila drag: biarkan kotak + tombol "Pindai Area Pilihan" tampil
  };
  mainCanvas.addEventListener('pointerup', end);
  mainCanvas.addEventListener('pointercancel', () => { selStart = null; });
  $('#btnRescanSel').addEventListener('click', () => {
    if (!selRect) return;
    const roi = { x0: selRect[0], y0: selRect[1], x1: selRect[2], y1: selRect[3] };
    clearSelection();
    executePipeline(roi);
  });
}

// =================== TEMPEL (Ctrl+V) & SERET GAMBAR DI MANA SAJA ===================
function initPasteAndDrop() {
  document.addEventListener('paste', e => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const it of items) {
      if (it.type && it.type.startsWith('image/')) {
        const f = it.getAsFile();
        if (f) { handleImageUpload(f); e.preventDefault(); return; }
      }
    }
  });
  // Drop file di area mana pun (bukan cuma dropzone) — cegah browser membuka gambar.
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => {
    e.preventDefault();
    const files = e.dataTransfer && e.dataTransfer.files;
    if (!files || !files.length) return;
    if (files.length > 1 && window.AksaraV4?.queueBatchFiles) {
      switchMainView('batch');
      window.AksaraV4.queueBatchFiles(files);
    } else if (files[0] && files[0].type.startsWith('image/')) {
      handleImageUpload(files[0]);
    }
  });
}

// Algoritma Otsu dengan Evaluasi Sentral Tahan Bayangan/Batas
function calculateOtsu(gray, w, h) {
  const hist = new Uint32Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;

  // Evaluasi Otsu di area sentral dokumen (12% margin dari setiap tepi)
  // untuk mencegah bayangan meja gelap atau batas tepi foto merusak ambang bimodal
  const subHist = new Uint32Array(256);
  let subTotal = 0;
  let subSum = 0;

  if (w && h) {
    const yStart = Math.floor(h * 0.12);
    const yEnd = Math.floor(h * 0.88);
    const xStart = Math.floor(w * 0.12);
    const xEnd = Math.floor(w * 0.88);

    for (let y = yStart; y < yEnd; y++) {
      const row = y * w;
      for (let x = xStart; x < xEnd; x++) {
        const v = gray[row + x];
        subHist[v]++;
        subSum += v;
        subTotal++;
      }
    }
  }

  const targetHist = subTotal > 500 ? subHist : hist;
  const total = subTotal > 500 ? subTotal : gray.length;
  let sum = subTotal > 500 ? subSum : 0;
  if (total === gray.length) {
    for (let i = 0; i < 256; i++) sum += i * hist[i];
  }

  let sumB = 0;
  let wB = 0;
  let bestVar = -1;
  let threshold = 127;

  for (let t = 0; t < 256; t++) {
    wB += targetHist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;

    sumB += t * targetHist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const betweenVar = wB * wF * (mB - mF) * (mB - mF);

    if (betweenVar > bestVar) {
      bestVar = betweenVar;
      threshold = t;
    }
  }
  return { threshold, hist };
}

// Binerisasi adaptif: ambang per-pixel = min(globalT, rata-rata lokal * 0.88).
// Integral image O(1)/pixel. Bayangan gelap tidak lagi jadi tinta blob,
// goresan tipis di area terang tetap tertangkap. Deterministik.
function adaptiveBinary(gray, w, h, globalT) {
  const IW = w + 1, IH = h + 1;
  const ii = new Float64Array(IW * IH);
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    const cur = (y + 1) * IW, prev = y * IW;
    for (let x = 0; x < w; x++) {
      rowSum += gray[y * w + x];
      ii[cur + x + 1] = ii[prev + x + 1] + rowSum;
    }
  }
  const bin = new Uint8Array(w * h);
  const s = Math.max(15, Math.round(Math.min(w, h) / 12));
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - s), y1 = Math.min(h - 1, y + s);
    const rT = (y1 + 1) * IW, rB = y0 * IW;
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - s), x1 = Math.min(w - 1, x + s);
      const area = (y1 - y0 + 1) * (x1 - x0 + 1);
      const sum = ii[rT + x1 + 1] - ii[rB + x1 + 1] - ii[rT + x0] + ii[rB + x0];
      const th = Math.min(globalT, (sum / area) * 0.88);
      bin[y * w + x] = gray[y * w + x] < th ? 1 : 0;
    }
  }
  return bin;
}

// Pembersihan Artefak Batas Meja/Tepi Gelap
function clearBorderNoise(bin, w, h) {
  const visited = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      // Hanya telusuri komponen yang menyentuh batas tepi luar (1 piksel perimeter)
      if (!bin[idx] || visited[idx] || (x > 1 && x < w - 2 && y > 1 && y < h - 2)) continue;

      let qHead = 0, qTail = 0;
      queue[qTail++] = idx;
      visited[idx] = 1;
      let touchesBorder = false;
      let area = 0;

      while (qHead < qTail) {
        const curr = queue[qHead++];
        area++;
        const cy = Math.floor(curr / w);
        const cx = curr % w;
        if (cx <= 1 || cy <= 1 || cx >= w - 2 || cy >= h - 2) touchesBorder = true;

        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= h) continue;
          const nrow = ny * w;
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = cx + dx;
            if (nx < 0 || nx >= w) continue;
            const nidx = nrow + nx;
            if (bin[nidx] && !visited[nidx]) {
              visited[nidx] = 1;
              queue[qTail++] = nidx;
            }
          }
        }
      }

      // Jika komponen menyentuh tepi dan sangat besar (artefak meja gelap / margin scanner), hapus
      if (touchesBorder && (area > 1500 || area > w * h * 0.02)) {
        for (let i = 0; i < qTail; i++) {
          bin[queue[i]] = 0;
        }
      }
    }
  }
}

// Filter Spasial Median 3x3 Preservasi Goresan Tipis
function applyMedian3x3(bin, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      let ones = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const drow = (y + dy) * w;
        for (let dx = -1; dx <= 1; dx++) {
          if (bin[drow + (x + dx)]) ones++;
        }
      }
      const self = bin[row + x];
      // Ink dipertahankan selama >=1 tetangga (jangan hapus goresan tipis/sambungan 1px);
      // latar hanya jadi ink bila >=7 tetangga (hole-filling bintik)
      if (self) {
        out[row + x] = ones >= 1 ? 1 : 0;
      } else {
        out[row + x] = ones >= 7 ? 1 : 0;
      }
    }
  }
  return out;
}

// Operasi Morfologi (Penghapusan Garis Horizontal + Closing)
function applyMorphology(bin, w, h) {
  // 1. Deteksi & eliminasi garis buku tulis horizontal
  const lineRemovalOn = $('#optRemoveLines').checked;
  let clean = bin;

  if (lineRemovalOn) {
    const lines = new Uint8Array(w * h);
    const kLen = Math.max(13, Math.min(31, Math.round(w * 0.025))) | 1;
    const half = Math.floor(kLen / 2);

    // Erosi horizontal
    const eroded = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = half; x < w - half; x++) {
        let allOnes = 1;
        for (let k = -half; k <= half; k++) {
          if (!bin[row + x + k]) { allOnes = 0; break; }
        }
        eroded[row + x] = allOnes;
      }
    }
    // Dilasi horizontal
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = half; x < w - half; x++) {
        if (eroded[row + x]) {
          for (let k = -half; k <= half; k++) lines[row + x + k] = 1;
        }
      }
    }
    // Subtraksi garis
    clean = new Uint8Array(w * h);
    for (let i = 0; i < bin.length; i++) {
      clean[i] = bin[i] && !lines[i] ? 1 : 0;
    }
  }

  // 2. Closing 3x3 untuk menyambung goresan pulpen
  const closed = new Uint8Array(w * h);
  // Dilasi 3x3
  const dilated = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      if (clean[row + x]) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            dilated[(y + dy) * w + (x + dx)] = 1;
          }
        }
      }
    }
  }
  // Erosi 3x3
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      let match = 1;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dilated[(y + dy) * w + (x + dx)]) { match = 0; break; }
        }
        if (!match) break;
      }
      closed[row + x] = match;
    }
  }
  return closed;
}

// Segmentasi Komponen Terhubung 2D (Connected Component Labeling) & Pengelompokan Baris
function segmentDocument(bin, w, h, scaleRef) {
  const visited = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  const comps = [];
  // Skala relatif: foto resolusi tinggi tidak boleh membuang glif kecil;
  // foto kecil tidak boleh meloloskan debu. minArea 35px @1500px sebanding ~12px @900px.
  const pxScale = (scaleRef || Math.max(w, h)) / 1500;
  const minArea = Math.max(12, Math.round(30 * pxScale * pxScale));
  const maxArea = w * h * 0.35;

  // 1. Ekstraksi Komponen Terhubung 8-Arah (BFS)
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const idx = row + x;
      if (!bin[idx] || visited[idx]) continue;

      let qHead = 0, qTail = 0;
      queue[qTail++] = idx;
      visited[idx] = 1;
      let minX = x, maxX = x, minY = y, maxY = y;
      let area = 0;

      while (qHead < qTail) {
        const curr = queue[qHead++];
        area++;
        const cy = Math.floor(curr / w);
        const cx = curr % w;

        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= h) continue;
          const nrow = ny * w;
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = cx + dx;
            if (nx < 0 || nx >= w) continue;
            const nidx = nrow + nx;
            if (bin[nidx] && !visited[nidx]) {
              visited[nidx] = 1;
              queue[qTail++] = nidx;
            }
          }
        }
      }

      const cw = maxX - minX + 1;
      const ch = maxY - minY + 1;

      // Filter noise bintik kecil dan artefak dokumen raksasa.
      // Garis lurus panjang (tembok/pinggiran meja/kisi) dibuang: aspect ekstrem
      // + kepadatan tinta rendah + membentang hampir selebar/setinggi halaman.
      const elongated = cw > 8 * ch || ch > 8 * cw;
      const sparse = area < 0.12 * cw * ch;
      // Membentang = garis/kerangka HOLLOW (isi renggang). Batas absolut 300px
      // supaya glif utuh di dalam seleksi sempit tidak ikut terbuang.
      const spansPage = sparse && ((cw > 0.6 * w && cw > 300) || (ch > 0.6 * h && ch > 300));
      if (area >= minArea && cw >= Math.max(4, Math.round(6 * pxScale)) && ch >= Math.max(6, Math.round(9 * pxScale)) && area < maxArea &&
          !(elongated && sparse) && !spansPage) {
        comps.push({
          x0: minX,
          y0: minY,
          x1: maxX + 1,
          y1: maxY + 1,
          w: cw,
          h: ch,
          area,
          centerY: minY + ch / 2,
          centerX: minX + cw / 2
        });
      }
    }
  }

  if (!comps.length) return [];

  // 2. Pengelompokan Komponen ke dalam Baris Teks (Line Clustering)
  comps.sort((a, b) => a.centerY - b.centerY);
  const lines = [];

  for (const c of comps) {
    let placed = false;
    for (const line of lines) {
      const avgY = line.reduce((acc, it) => acc + it.centerY, 0) / line.length;
      const avgH = line.reduce((acc, it) => acc + it.h, 0) / line.length;
      if (Math.abs(c.centerY - avgY) < Math.max(14 * pxScale, avgH * 0.6)) {
        line.push(c);
        placed = true;
        break;
      }
    }
    if (!placed) {
      lines.push([c]);
    }
  }

  // Urutkan baris dari atas ke bawah
  lines.sort((l1, l2) => {
    const y1 = l1.reduce((acc, it) => acc + it.centerY, 0) / l1.length;
    const y2 = l2.reduce((acc, it) => acc + it.centerY, 0) / l2.length;
    return y1 - y2;
  });

  // 3. Pengurutan Kiri-ke-Kanan & Penggabungan Aksara Terpisah (Intra-Glyph Merge)
  const segments = [];

  for (const line of lines) {
    line.sort((a, b) => a.x0 - b.x0);
    const merged = [];

    for (const c of line) {
      if (!merged.length) {
        merged.push({ ...c });
      } else {
        const prev = merged[merged.length - 1];
        const gap = c.x0 - prev.x1;
        const avgH = (prev.h + c.h) / 2;
        const overlapY = Math.max(0, Math.min(prev.y1, c.y1) - Math.max(prev.y0, c.y0));

        // Jika jarak horizontal sangat rapat (stroke terpisah seperti pada ra, ka, ya, atau sandhangan)
        if (gap < Math.max(14 * pxScale, avgH * 0.22) && (overlapY > 0 || gap < 6 * pxScale)) {
          prev.x0 = Math.min(prev.x0, c.x0);
          prev.y0 = Math.min(prev.y0, c.y0);
          prev.x1 = Math.max(prev.x1, c.x1);
          prev.y1 = Math.max(prev.y1, c.y1);
          prev.w = prev.x1 - prev.x0;
          prev.h = prev.y1 - prev.y0;
          prev.area += c.area;
        } else {
          merged.push({ ...c });
        }
      }
    }

    for (const m of merged) {
      const cw = m.w;
      const ch = m.h;
      const crop = new Uint8Array(cw * ch);
      for (let y = 0; y < ch; y++) {
        crop.set(bin.subarray((m.y0 + y) * w + m.x0, (m.y0 + y) * w + m.x1), y * cw);
      }
      segments.push({
        crop,
        cw,
        ch,
        box: [m.x0, m.y0, m.x1, m.y1]
      });
    }
  }

  return segments.slice(0, 300);
}

// Normalisasi Glif Isotropik 96x96
function normCrop(crop, cw, ch, size = 96, inner = 76) {
  let x0 = cw, y0 = ch, x1 = -1, y1 = -1;
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      if (crop[y * cw + x]) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < x0 || (x1 - x0 + 1) * (y1 - y0 + 1) < 20) return null;

  const w = x1 - x0 + 1;
  const h = y1 - y0 + 1;
  const sc = inner / Math.max(w, h);
  const nw = Math.max(1, Math.round(w * sc));
  const nh = Math.max(1, Math.round(h * sc));
  const out = new Float32Array(size * size);
  const ox = Math.floor((size - nw) / 2);
  const oy = Math.floor((size - nh) / 2);

  for (let y = 0; y < nh; y++) {
    const sy = y0 + Math.min(h - 1, Math.floor(y / sc));
    for (let x = 0; x < nw; x++) {
      const sx = x0 + Math.min(w - 1, Math.floor(x / sc));
      out[(oy + y) * size + ox + x] = crop[sy * cw + sx] ? 1 : 0;
    }
  }
  return { out, box: [x0, y0, x1 + 1, y1 + 1] };
}

// Penghalusan fitur Gaussian separable 96x96, pad half-symmetric (reflect_101).
// Identik matematis dengan blur96() di train_svm_v2.py: kernel exp, radius ceil(3*sigma).
function smoothFeatures(feat, size, sigma) {
  const r = Math.ceil(3 * sigma);
  const n = 2 * r + 1;
  const W2 = size + 2 * r;
  const k = new Float64Array(n);
  let ksum = 0;
  for (let i = 0; i < n; i++) {
    const x = i - r;
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    ksum += k[i];
  }
  for (let i = 0; i < n; i++) k[i] /= ksum;
  const mirror = (v, hi) => { // half-symmetric: pantul tanpa duplikasi tepi, v di [-(W-1), 2W-2]
    if (v < 0) v = -v;
    if (v > hi) v = 2 * hi - v;
    return v;
  };
  // pass kolom: hi = size-1
  const t = new Float64Array(size * size);
  for (let y = 0; y < size; y++) {
    const srow = y * size;
    for (let x = 0; x < size; x++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += k[i] * feat[srow + mirror(x + i - r, size - 1)];
      t[srow + x] = s;
    }
  }
  // pass baris: hi = size-1 pada indeks y
  const out = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += k[i] * t[mirror(y + i - r, size - 1) * size + x];
      out[y * size + x] = s;
    }
  }
  return out;
}

// Klasifikasi SVM Linear Multi-Kelas One-vs-Rest
function classifyGlif(feat, weights, model) {
  const n = model.size * model.size;
  let best = -Infinity;
  let bestIdx = 0;
  const scores = [];

  for (let c = 0; c < model.classes.length; c++) {
    let score = model.bias[c];
    const off = c * n;
    for (let i = 0; i < n; i++) {
      score += feat[i] * weights[off + i];
    }
    scores.push(score);
    if (score > best) {
      best = score;
      bestIdx = c;
    }
  }

  // Softmax untuk kalibrasi skor keyakinan
  const expScores = scores.map(s => Math.exp(s / 1.5));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const confidence = expScores[bestIdx] / sumExp;

  const label = model.classes[bestIdx];
  const unicode = UNICODE_MAP[label] || '';

  return {
    label,
    unicode,
    confidence,
    isUncertain: confidence < 0.25
  };
}

// ============================================================================
// RENDERING & INTERAKSI HASIL
// ============================================================================

function redrawMainCanvas() {
  if (!sourceImage) return;
  const w = mainCanvas.width;
  const h = mainCanvas.height;

  const cmpChk = $('#toggleCompare');
  const cmpDivider = $('#compareDivider');
  const cmpRange = $('#compareSplitRange');
  const enhC = window.__aksaraLastEnhancedCanvas;
  const compareActive = Boolean(cmpChk && cmpChk.checked && enhC);

  if (cmpDivider) cmpDivider.hidden = !compareActive;
  if (cmpRange) cmpRange.hidden = !compareActive;

  if ($('#showBin')?.checked && labStages.binary) {
    const imgData = mainCtx.createImageData(w, h);
    const bin = labStages.binary;
    for (let i = 0, p = 0; i < bin.length; i++, p += 4) {
      const v = bin[i] ? 0 : 255;
      imgData.data[p] = v;
      imgData.data[p + 1] = v;
      imgData.data[p + 2] = v;
      imgData.data[p + 3] = 255;
    }
    mainCtx.putImageData(imgData, 0, 0);
  } else if (compareActive) {
    const pct = clampPercent(parseFloat(cmpRange?.value || '50'));
    const splitX = Math.round((pct / 100) * w);
    if (cmpDivider) cmpDivider.style.left = `${pct}%`;

    mainCtx.clearRect(0, 0, w, h);
    // Kiri: citra asli sebelum enhancer
    mainCtx.save();
    mainCtx.beginPath();
    mainCtx.rect(0, 0, splitX, h);
    mainCtx.clip();
    mainCtx.drawImage(sourceImage, 0, 0, w, h);
    mainCtx.restore();

    // Kanan: citra setelah praproses Worker
    mainCtx.save();
    mainCtx.beginPath();
    mainCtx.rect(splitX, 0, Math.max(0, w - splitX), h);
    mainCtx.clip();
    mainCtx.drawImage(enhC, 0, 0, w, h);
    mainCtx.restore();
  } else {
    mainCtx.drawImage(sourceImage, 0, 0, w, h);
  }

  // Overlay Kotak Deteksi Karakter (Berwarna per Kategori)
  if ($('#showBoxes')?.checked && currentDetections.length) {
    mainCtx.lineWidth = Math.max(1.5, w / 600);
    mainCtx.font = `bold ${Math.max(11, Math.round(w / 80))}px 'Plus Jakarta Sans', sans-serif`;

    currentDetections.forEach((det) => {
      const [x0, y0, x1, y1] = det.box;
      const bw = x1 - x0;
      const bh = y1 - y0;

      let strokeCol = '#c8963e';
      let tagBg = '#1f2a44';
      if (det.isUncertain) {
        strokeCol = '#d1563b';
        tagBg = '#d1563b';
      } else if (det.category === 'pasangan') {
        strokeCol = '#5c8df6';
        tagBg = '#284587';
      } else if (det.category === 'sandhangan') {
        strokeCol = '#e88c30';
        tagBg = '#7c430d';
      } else if (det.category === 'pada') {
        strokeCol = '#3fa97a';
        tagBg = '#1e5e41';
      }

      mainCtx.strokeStyle = strokeCol;
      mainCtx.strokeRect(x0, y0, bw, bh);

      // Label mini pas lebar teks di atas kotak
      const tw = Math.ceil(mainCtx.measureText(det.label).width) + 8;
      mainCtx.fillStyle = tagBg;
      mainCtx.fillRect(x0, Math.max(0, y0 - 17), tw, 17);
      mainCtx.fillStyle = '#f4ead5';
      mainCtx.fillText(det.label, x0 + 4, Math.max(12, y0 - 4));
    });
  }
}

function clampPercent(v) {
  if (!Number.isFinite(v)) return 50;
  return Math.max(5, Math.min(95, v));
}

function updateEmptyRecoveryNotice(hasDetections) {
  const alertEl = $('#scanEmptyAlert');
  if (!alertEl) return;
  if (hasDetections) {
    alertEl.hidden = true;
    return;
  }
  const rep = window.__aksaraLastEnhanceReport;
  const blurLv = rep?.blurBefore?.level || '';
  const msgEl = $('#scanEmptyMsg');
  if (msgEl) {
    if (blurLv === 'berat' || blurLv === 'sedang') {
      msgEl.textContent = 'Foto terlalu buram atau rendah kontras sehingga aksara belum terbaca. Coba foto ulang lebih dekat dengan cahaya merata, aktifkan Tajamkan Foto Blur, atau turunkan ambang ke 0.18.';
    } else {
      msgEl.textContent = 'Tidak ada aksara Jawa yang terdeteksi pada gambar ini. Pastikan gambar memuat teks aksara Jawa yang jelas, atau coba turunkan ambang keyakinan.';
    }
  }
  alertEl.hidden = false;
}

function renderOutputs() {
  if (!currentDetections.length) {
    resetResults();
    updateEmptyRecoveryNotice(false);
    return;
  }
  updateEmptyRecoveryNotice(true);

  let javaneseText = '';
  let latinText = '';
  let sentenceLatin = '—';
  let sentenceIndonesian = '—';

  if (activeEngine === 'yolo' && yoloReconstructed) {
    javaneseText = yoloReconstructed.unicodeText;
    latinText = yoloReconstructed.latinText;
    sentenceLatin = yoloReconstructed.sentenceLatin || latinText;
    sentenceIndonesian = yoloReconstructed.sentenceIndonesian || '—';
  } else {
    javaneseText = currentDetections.map(d => d.unicode).join(' ');
    latinText = currentDetections.map(d => d.label).join(' ');
    if (window.AksaraLexicon) {
      const tr = window.AksaraLexicon.translateJavaneseTokens(
        currentDetections.map(d => ({ isPunct: false, latin: d.label }))
      );
      sentenceLatin = tr.sentenceLatin;
      sentenceIndonesian = tr.sentenceIndonesian;
    } else {
      sentenceLatin = latinText;
    }
  }

  const avgConf = Math.round(
    (currentDetections.reduce((a, b) => a + b.confidence, 0) / currentDetections.length) * 100
  );

  $('#outJavanese').textContent = javaneseText;
  $('#outLatin').textContent = latinText;
  if ($('#outSentenceLatin')) $('#outSentenceLatin').textContent = sentenceLatin;
  if ($('#outIndonesian')) $('#outIndonesian').textContent = sentenceIndonesian;
  const meterVal = $('#meterVal');
  if (meterVal) meterVal.textContent = `${avgConf}%`;
  const meterFill = $('#meterFill');
  if (meterFill) meterFill.style.width = `${avgConf}%`;
  const meterWrap = $('#meterWrap');
  if (meterWrap) meterWrap.hidden = false;
  if ($('#glyphCountBadge')) $('#glyphCountBadge').textContent = String(currentDetections.length);

  const s2St = window.__aksaraLastStage2Stats;
  const s2El = $('#stage2Summary');
  if (s2El) {
    if (s2St && s2St.active) {
      s2El.textContent = `Stage-2: ${s2St.verified || 0} diperiksa · ${s2St.rejectedNonAksara || 0} non-aksara ditolak`;
    } else {
      s2El.textContent = activeEngine === 'yolo' ? 'Mode Detektor Tahap-1' : 'Pipeline Klasik SVM';
    }
  }

  // Grid Karakter Glif
  const grid = $('#glyphsGrid');
  grid.replaceChildren();
  const normSize = model ? model.size : 96;

  currentDetections.forEach((det, idx) => {
    const card = document.createElement('div');
    card.className = `glyph-card ${det.isUncertain ? 'uncertain' : ''}`;
    card.title = `Ketuk untuk inspeksi referensi #${idx + 1}`;
    if (det.category === 'pasangan') card.style.borderColor = '#5c8df6';
    else if (det.category === 'sandhangan') card.style.borderColor = '#e88c30';
    else if (det.category === 'pada') card.style.borderColor = '#3fa97a';

    const c = document.createElement('canvas');
    c.width = normSize;
    c.height = normSize;
    const ctx = c.getContext('2d');
    const id = ctx.createImageData(normSize, normSize);
    for (let i = 0, p = 0; i < det.normCanvas.length; i++, p += 4) {
      const v = det.normCanvas[i] ? 0 : 255;
      id.data[p] = v; id.data[p + 1] = v; id.data[p + 2] = v; id.data[p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);

    const unicodeSpan = document.createElement('div');
    unicodeSpan.className = 'glyph-card__unicode';
    unicodeSpan.textContent = det.unicode;

    const labelSpan = document.createElement('div');
    labelSpan.className = 'glyph-card__label';
    labelSpan.textContent = det.label;

    const confSpan = document.createElement('small');
    confSpan.className = 'glyph-card__conf';
    confSpan.textContent = `${Math.round(det.confidence * 100)}%`;

    card.append(c, unicodeSpan, labelSpan, confSpan);
    card.addEventListener('click', () => openCorrectionModal(idx));
    grid.append(card);
  });

  window.dispatchEvent(new CustomEvent('aksara:scancomplete', {
    detail: {
      fileName: lastSourceName || 'Pindaian',
      unicodeText: javaneseText,
      sentenceLatin,
      sentenceIndonesian,
      avgConf,
      engine: activeEngine,
      blurLevel: window.__aksaraLastEnhanceReport?.blurBefore?.level || '—',
      detectionsCount: currentDetections.length,
    }
  }));
}

// Modal Koreksi Manual (20 Nglegena di mode SVM; 52 Kelas di mode YOLOv8)
let modalListenersBound = false;
function setupCorrectionModal() {
  const modal = $('#correctionModal');
  modal.hidden = true;
  modal.classList.add('is-hidden');
  const grid = $('#correctionGrid');
  grid.replaceChildren();

  const items = [];
  if (activeEngine === 'yolo' && yoloMeta) {
    yoloMeta.classes.forEach((clsName, idx) => {
      items.push({
        latin: clsName,
        unicode: yoloMeta.unicode[clsName] || '',
        category: yoloMeta.categories[idx]
      });
    });
  } else {
    AKSARA_DATA.forEach(a => items.push({ latin: a.latin, unicode: a.unicode, category: 'nglegena' }));
  }

  items.forEach(a => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'correction-btn';
    btn.dataset.label = a.latin.toLowerCase();
    if (a.category === 'pasangan') btn.style.borderColor = 'rgba(92,141,246,0.45)';
    else if (a.category === 'sandhangan') btn.style.borderColor = 'rgba(232,140,48,0.45)';
    btn.innerHTML = `<span class="aksara-glyph">${a.unicode}</span><span class="aksara-name">${a.latin}</span>`;
    btn.addEventListener('click', () => {
      if (selectedGlyphIndex >= 0 && selectedGlyphIndex < currentDetections.length) {
        const prev = currentDetections[selectedGlyphIndex].label;
        currentDetections[selectedGlyphIndex].label = a.latin;
        currentDetections[selectedGlyphIndex].unicode = a.unicode;
        currentDetections[selectedGlyphIndex].category = a.category;
        currentDetections[selectedGlyphIndex].confidence = 1.0;
        currentDetections[selectedGlyphIndex].isUncertain = false;
        window.dispatchEvent(new CustomEvent('aksara:correction', { detail: { type: 'manual_glyph', from: prev, to: a.latin, index: selectedGlyphIndex } }));
        if (activeEngine === 'yolo' && window.AksaraOrthography) {
          yoloReconstructed = window.AksaraOrthography.reconstructJavaneseText(currentDetections);
        }
        renderOutputs();
        redrawMainCanvas();
      }
      closeCorrectionModal();
    });
    grid.append(btn);
  });

  if (!modalListenersBound) {
    $('#closeModal').addEventListener('click', closeCorrectionModal);
    modal.addEventListener('click', e => { if (e.target === modal) closeCorrectionModal(); });
    modalListenersBound = true;
  }
}

function closeCorrectionModal() {
  const modal = $('#correctionModal');
  modal.hidden = true;
  modal.classList.add('is-hidden');
  selectedGlyphIndex = -1;
}

function openCorrectionModal(index) {
  selectedGlyphIndex = index;
  const det = currentDetections[index];
  $('#modalTargetLabel').textContent = `Inspektor Referensi · Aksara #${index + 1} (${det.label} / ${det.unicode})`;
  if (window.AksaraV4 && window.AksaraV4.renderInspector) {
    window.AksaraV4.renderInspector(det, index, sourceImage, window.__aksaraLastEnhancedCanvas || null);
  }
  const modal = $('#correctionModal');
  modal.hidden = false;
  modal.classList.remove('is-hidden');
}

// ============================================================================
// LAB PIPELINE INTERAKTIF (8 TAHAPAN + HISTOGRAM)
// ============================================================================

// Overlay bounding box hasil CCL pada citra biner (Lab tahap 5 & 6).
// Nilai array: 0 = kertas, 1 = tinta, 2 = garis kotak pembatas.
function overlaySegmentation(bin, w, h, segments) {
  const out = new Uint8Array(w * h);
  for (let i = 0; i < bin.length; i++) out[i] = bin[i] ? 1 : 0;
  for (const s of segments) {
    const [x0, y0, x1, y1] = s.box;
    for (let x = x0; x <= x1; x++) { out[y0 * w + x] = 2; out[y1 * w + x] = 2; }
    for (let y = y0; y <= y1; y++) { out[y * w + x0] = 2; out[y * w + x1] = 2; }
  }
  return out;
}

// Montase glif ternormalisasi 96x96 berlabel prediksi (Lab tahap 7 & 8).
function cacheSvmMontage(dets, size) {
  if (!dets.length) { labStages.svm = null; return; }
  const shown = dets.slice(0, 12);
  const cols = 4;
  const rows = Math.ceil(shown.length / cols);
  const labelH = 18;
  const cell = size + 8;
  const c = document.createElement('canvas');
  c.width = cols * cell + 8;
  c.height = rows * (cell + labelH) + 8;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, c.width, c.height);
  // Salinan tahap 7: glif pertama pada kanvas normalisasi 96x96
  const first = document.createElement('canvas');
  first.width = size; first.height = size;
  const fctx = first.getContext('2d');
  fctx.fillStyle = '#f2ede2';
  fctx.fillRect(0, 0, size, size);
  fctx.fillStyle = '#000000';
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (shown[0].normCanvas[y * size + x]) fctx.fillRect(x, y, 1, 1);
    }
  }
  labStages.norm = first;
  shown.forEach((d, i) => {
    const gx = 8 + (i % cols) * cell;
    const gy = 8 + Math.floor(i / cols) * (cell + labelH);
    ctx.fillStyle = '#f2ede2';
    ctx.fillRect(gx, gy, size, size);
    ctx.fillStyle = '#000000';
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (d.normCanvas[y * size + x]) ctx.fillRect(gx + x, gy + y, 1, 1);
      }
    }
    ctx.fillStyle = d.isUncertain ? '#bd4327' : '#1e1b18';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(`${d.label} ${(d.confidence * 100).toFixed(0)}%`, gx, gy + size + 13);
  });
  labStages.svm = c;
}

function cacheLabStage(key, arrayData, w, h, type) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');
  const imgData = ctx.createImageData(w, h);

  if (type === 'gray') {
    for (let i = 0, p = 0; i < arrayData.length; i++, p += 4) {
      const v = arrayData[i];
      imgData.data[p] = v; imgData.data[p + 1] = v; imgData.data[p + 2] = v; imgData.data[p + 3] = 255;
    }
  } else if (type === 'bin') {
    for (let i = 0, p = 0; i < arrayData.length; i++, p += 4) {
      const v = arrayData[i] ? 0 : 255;
      imgData.data[p] = v; imgData.data[p + 1] = v; imgData.data[p + 2] = v; imgData.data[p + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  labStages[key] = c;
}

function cacheHistogram(hist, threshold) {
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 100;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#121826';
  ctx.fillRect(0, 0, 256, 100);

  let maxVal = 1;
  for (let i = 0; i < 256; i++) if (hist[i] > maxVal) maxVal = hist[i];

  ctx.strokeStyle = '#c8963e';
  ctx.beginPath();
  for (let x = 0; x < 256; x++) {
    const y = 100 - (hist[x] / maxVal) * 90;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Garis Ambang Threshold
  ctx.strokeStyle = '#d1563b';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(threshold, 0);
  ctx.lineTo(threshold, 100);
  ctx.stroke();

  labStages.histogram = c;
}

function renderLabStage(stageKey) {
  const box = $('#labPreviewBox');
  box.replaceChildren();

  const canvas = labStages[stageKey];
  if (canvas) {
    box.append(canvas);
  } else {
    box.innerHTML = '<p style="padding:40px;color:var(--text-muted);font-size:13px">Tahapan belum diproses.</p>';
  }

  // Tampilkan histogram jika Otsu
  const histBox = $('#labHistogramBox');
  histBox.replaceChildren();
  if (labStages.histogram) {
    histBox.append(labStages.histogram);
  }
}

// Handler Tab Tahapan Lab
$$('.lab-stage-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.lab-stage-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderLabStage(btn.dataset.stage);
  });
});

// Slider Otsu Interaktif
$('#otsuRange').addEventListener('input', e => {
  manualThreshold = parseInt(e.target.value, 10);
  $('#otsuValueDisplay').textContent = manualThreshold + ' (manual)';
});

$('#btnResetThreshold').addEventListener('click', () => {
  manualThreshold = null;
  $('#otsuValueDisplay').textContent = currentThreshold;
  $('#otsuRange').value = currentThreshold;
  if (sourceImage) executePipeline();
});

$('#btnReapplyLab').addEventListener('click', () => {
  manualThreshold = parseInt($('#otsuRange').value, 10);
  executePipeline();
});

// Kamus Aksara (Nglegena + Pasangan + Sandhangan/Pada)
function renderDictionary() {
  const grid = $('#dictionaryGrid');
  if (grid) {
    grid.replaceChildren();
    AKSARA_DATA.forEach((a, i) => {
      const card = document.createElement('div');
      card.className = 'dict-card';
      card.innerHTML = `
        <div class="glif">${a.unicode}</div>
        <div class="nama">${a.name}</div>
        <div class="latin">#${i + 1} · ${a.latin}</div>
      `;
      grid.append(card);
    });
  }

  const pGrid = $('#pasanganDictGrid');
  if (pGrid) {
    pGrid.replaceChildren();
    AKSARA_DATA.forEach(a => {
      const posNote = (a.latin === 'ha' || a.latin === 'sa' || a.latin === 'pa') ? 'posisi kanan' : 'di bawah';
      const card = document.createElement('div');
      card.className = 'dict-card';
      card.innerHTML = `
        <div class="glif">꧀${a.unicode}</div>
        <div class="nama">p_${a.latin}</div>
        <div class="latin">${posNote}</div>
      `;
      pGrid.append(card);
    });
  }

  const sGrid = $('#sandhanganDictGrid');
  if (sGrid) {
    sGrid.replaceChildren();
    const sItems = [
      { g: 'ꦶ', n: 'Wulu', l: 'Vokal -i (atas)' },
      { g: 'ꦸ', n: 'Suku', l: 'Vokal -u (bawah)' },
      { g: 'ꦼ', n: 'Pepet', l: 'Vokal -ê (atas)' },
      { g: 'ꦺ', n: 'Taling', l: 'Vokal -é/è (kiri)' },
      { g: 'ꦴ', n: 'Tarung', l: 'Pasangan -o (kanan)' },
      { g: 'ꦂ', n: 'Layar', l: 'Akhir -r (atas)' },
      { g: 'ꦁ', n: 'Cecak', l: 'Akhir -ng (atas)' },
      { g: 'ꦃ', n: 'Wignyan', l: 'Akhir -h (kanan)' },
      { g: '꧀', n: 'Pangkon', l: 'Pemati vokal' },
      { g: 'ꦿ', n: 'Cakra', l: 'Sisipan -ra' },
      { g: 'ꦽ', n: 'Keret', l: 'Sisipan -rê' },
      { g: 'ꦾ', n: 'Pengkal', l: 'Sisipan -ya' },
      { g: '꧋', n: 'Pada Adeg', l: 'Pembuka kalimat' },
      { g: '꧈', n: 'Pada Lingsa', l: 'Tanda koma (,)' },
      { g: '꧉', n: 'Pada Lungsi', l: 'Tanda titik (.)' },
    ];
    sItems.forEach(it => {
      const card = document.createElement('div');
      card.className = 'dict-card';
      const isPada = it.g === '꧋' || it.g === '꧈' || it.g === '꧉';
      card.innerHTML = `
        <div class="glif">${isPada ? it.g : 'ꦏ' + it.g}</div>
        <div class="nama">${it.n}</div>
        <div class="latin">${it.l}</div>
      `;
      sGrid.append(card);
    });
  }
}

// Utilitas Clipboard & Unduh
// Ekspor hasil deteksi sebagai JSON (PRD: Salin Unicode, Salin Latin, Unduh JSON).
function buildExportText(kind) {
  const unicode = $('#outJavanese')?.textContent || '—';
  const latin = $('#outSentenceLatin')?.textContent || $('#outLatin')?.textContent || '—';
  const indo = $('#outIndonesian')?.textContent || '—';
  if (kind === 'md') {
    return `# Hasil Pemindaian AksaraScan\n\n- **Sumber:** ${lastSourceName || 'Unggahan pengguna'}\n\n## Aksara Jawa (Unicode)\n\n${unicode}\n\n## Bacaan Latin\n\n${latin}\n\n## Arti Bahasa Indonesia\n\n${indo}\n`;
  }
  return `Aksara Jawa (Unicode):\n${unicode}\n\nBacaan Latin:\n${latin}\n\nArti Bahasa Indonesia:\n${indo}\n`;
}

function downloadTextFile(content, filename, mime = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type: mime });
  const link = document.createElement('a');
  link.download = filename;
  link.href = URL.createObjectURL(blob);
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function downloadJson() {
  if (!currentDetections.length) return;
  const isYolo = (activeEngine === 'yolo' && yoloMeta);
  const payload = {
    aplikasi: 'AksaraScan',
    versi: '4.0.0',
    sumber: lastSourceName || null,
    model: isYolo
      ? { algoritma: 'YOLOv8-Nano ONNX + Stage-2', jumlahKelas: yoloMeta.classes.length, mAP50: yoloMeta.metrics?.map50 ?? null }
      : { algoritma: 'SVM Linear OvR', jumlahKelas: model.classes.length, akurasiUji: model.accuracy },
    rataRataKeyakinan: currentDetections.reduce((a, b) => a + b.confidence, 0) / currentDetections.length,
    teksUnicode: $('#outJavanese').textContent,
    kalimatLatin: $('#outSentenceLatin')?.textContent || $('#outLatin').textContent,
    terjemahanIndonesia: $('#outIndonesian')?.textContent || '—',
    transliterasiLatin: $('#outLatin').textContent,
    deteksi: currentDetections.map(d => ({
      index: d.index + 1,
      kategori: d.category || 'nglegena',
      label: d.label,
      unicode: d.unicode,
      confidence: d.confidence,
      ragu: d.isUncertain,
      kotak: d.box
    }))
  };
  downloadTextFile(JSON.stringify(payload, null, 2), `AksaraScan_${Date.now()}.json`, 'application/json');
}

function copyToClipboard(text, msg) {
  if (!text || text === '—') return;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(() => {});
  }
}

async function processImageFileBatch(file) {
  const url = URL.createObjectURL(file);
  const img = await new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error(`Gagal memuat ${file.name}`));
    im.src = url;
  });
  URL.revokeObjectURL(url);

  const maxDim = 1500;
  const scale = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
  const CW = Math.max(1, Math.round(img.naturalWidth * scale));
  const CH = Math.max(1, Math.round(img.naturalHeight * scale));
  const c = document.createElement('canvas');
  c.width = CW; c.height = CH;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, CW, CH);

  // Buat thumbnail ringkas untuk preview antrean
  const thC = document.createElement('canvas');
  const ts = Math.min(1, 140 / Math.max(CW, CH));
  thC.width = Math.max(1, Math.round(CW * ts));
  thC.height = Math.max(1, Math.round(CH * ts));
  thC.getContext('2d').drawImage(c, 0, 0, thC.width, thC.height);
  const thumbDataUrl = thC.toDataURL('image/jpeg', 0.72);

  const imgData = ctx.getImageData(0, 0, CW, CH);
  const data = imgData.data;
  const gray = new Uint8Array(CW * CH);
  const histLum = new Uint32Array(256);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const v = Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]);
    gray[i] = v;
    histLum[v]++;
  }
  let cum = 0, med = 255;
  const half = gray.length >> 1;
  for (let b = 0; b < 256; b++) { cum += histLum[b]; if (cum >= half) { med = b; break; } }
  let yoloIn = c;
  if (med < 112) {
    for (let k = 0; k < gray.length; k++) gray[k] = 255 - gray[k];
    const invC = document.createElement('canvas');
    invC.width = CW; invC.height = CH;
    const ictx = invC.getContext('2d', { willReadFrequently: true });
    const invImg = ictx.createImageData(CW, CH);
    for (let p = 0; p < data.length; p += 4) {
      invImg.data[p] = 255 - data[p];
      invImg.data[p + 1] = 255 - data[p + 1];
      invImg.data[p + 2] = 255 - data[p + 2];
      invImg.data[p + 3] = 255;
    }
    ictx.putImageData(invImg, 0, 0);
    yoloIn = invC;
  }

  const { threshold } = calculateOtsu(gray, CW, CH);
  const bin = adaptiveBinary(gray, CW, CH, threshold);
  clearBorderNoise(bin, CW, CH);
  const medBin = applyMedian3x3(bin, CW, CH);
  const morphBin = applyMorphology(medBin, CW, CH);

  const t0 = performance.now();
  if (yoloSession) {
    const res = await runYoloInferenceCore(CW, CH, 0, 0, CW, CH, null, morphBin, yoloIn, { silent: true });
    const avgConf = res.kept.length ? Math.round((res.kept.reduce((s, d) => s + d.confidence, 0) / res.kept.length) * 100) : 0;
    return {
      fileName: file.name,
      width: CW,
      height: CH,
      thumbDataUrl,
      unicodeText: res.reconstructed ? res.reconstructed.unicodeText : res.kept.map(d => d.unicode).join(' '),
      sentenceLatin: res.reconstructed ? res.reconstructed.sentenceLatin : res.kept.map(d => d.label).join(' '),
      sentenceIndonesian: res.reconstructed ? res.reconstructed.sentenceIndonesian : '—',
      detectionsCount: res.kept.length,
      avgConf,
      blurLevel: res.enhanceReport?.blurBefore?.level || '—',
      latencyMs: Math.round(performance.now() - t0),
    };
  }
  return {
    fileName: file.name,
    width: CW,
    height: CH,
    thumbDataUrl,
    unicodeText: '—',
    sentenceLatin: 'Model YOLOv8 belum aktif',
    sentenceIndonesian: '—',
    detectionsCount: 0,
    avgConf: 0,
    blurLevel: '—',
    latencyMs: Math.round(performance.now() - t0),
  };
}

window.AksaraApp = {
  switchMainView,
  openCorrectionModal,
  closeCorrectionModal,
  redrawMainCanvas,
  renderOutputs,
  processImageFileBatch,
  applyCandidateCorrection(idx, cand) {
    if (idx < 0 || idx >= currentDetections.length || !cand) return;
    currentDetections[idx].label = cand.label;
    currentDetections[idx].unicode = (yoloMeta && yoloMeta.unicode[cand.label]) || UNICODE_MAP[cand.label] || cand.unicode || '';
    currentDetections[idx].category = cand.category || currentDetections[idx].category;
    currentDetections[idx].confidence = Math.max(0.85, cand.prob || 0.85);
    currentDetections[idx].isUncertain = false;
    if (activeEngine === 'yolo' && window.AksaraOrthography) {
      yoloReconstructed = window.AksaraOrthography.reconstructJavaneseText(currentDetections);
    }
    renderOutputs();
    redrawMainCanvas();
  },
  getStateSnapshot() {
    return {
      sourceImage,
      enhancedCanvas: window.__aksaraLastEnhancedCanvas || null,
      detections: currentDetections,
      reconstructed: yoloReconstructed,
      enhanceReport: window.__aksaraLastEnhanceReport || null,
      stage2Stats: window.__aksaraLastStage2Stats || null,
      lastSourceName,
      activeEngine,
    };
  },
};

// Inisialisasi saat DOM siap
document.addEventListener('DOMContentLoaded', initApp);