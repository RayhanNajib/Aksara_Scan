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

  // Setup Tema
  const savedTheme = localStorage.getItem('aksarascan_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  $('#themeToggle').addEventListener('click', toggleTheme);

  // Setup Events
  // Kamera: mobile pakai input capture; desktop pakai MediaDevices getUserMedia.
  const hasCapture = camInput.capture !== undefined && camInput.capture !== '';
  $('#btnCam').addEventListener('click', () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia && !hasCapture) startDesktopCamera();
    else camInput.click();
  });
  $('#btnFile').addEventListener('click', () => fileInput.click());
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

  initSelection();
  initPasteAndDrop();

  // Drag and drop
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', e => { e.preventDefault(); dropzone.classList.remove('dragover'); });
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) handleImageUpload(e.dataTransfer.files[0]);
  });

  // Tombol aksi
  btnRun.addEventListener('click', () => { clearSelection(); executePipeline(); });
  $('#showBoxes').addEventListener('change', redrawMainCanvas);
  $('#showBin').addEventListener('change', redrawMainCanvas);
  $('#btnCopyUnicode').addEventListener('click', () => copyToClipboard($('#outJavanese').textContent, 'Aksara Jawa disalin!'));
  $('#btnCopyLatin').addEventListener('click', () => copyToClipboard($('#outLatin').textContent, 'Teks Latin disalin!'));
  $('#btnDownload').addEventListener('click', downloadJson);

  // Klik langsung pada kanvas untuk mengoreksi karakter
  mainCanvas.addEventListener('click', handleCanvasClick);

  // Lab Tab switcher
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.tab;
      if (target === 'result') {
        $('#tabResult').hidden = false;
        $('#tabLab').hidden = true;
      } else {
        $('#tabResult').hidden = true;
        $('#tabLab').hidden = false;
        renderLabStage('otsu');
      }
    });
  });

  // Demo Card Coba 1-Klik (dukungan click dan touch)
  $$('.demo-card').forEach(card => {
    const trigger = (e) => {
      e.preventDefault();
      loadDemoImage(card.dataset.src);
    };
    card.addEventListener('click', trigger);
  });

  // Setup Kamus Aksara & Modal
  renderDictionary();
  setupCorrectionModal();
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('aksarascan_theme', next);
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

function loadImageElement(img, label) {
  sourceImage = img;
  manualThreshold = null;
  lastSourceName = typeof label === 'string' ? label : String(label || 'gambar');
  const maxDim = 1500;
  const scale = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
  mainCanvas.width = Math.round(img.naturalWidth * scale);
  mainCanvas.height = Math.round(img.naturalHeight * scale);
  mainCtx.drawImage(img, 0, 0, mainCanvas.width, mainCanvas.height);

  dropzone.hidden = true;
  stageWrap.hidden = false;
  clearSelection();
  $('#srcDim').textContent = `${mainCanvas.width}×${mainCanvas.height} px (${label})`;
  btnRun.disabled = false;
  currentDetections = [];
  resetResults();
  // Auto-deteksi: begitu gambar masuk (upload/paste/drop/kamera) langsung scan,
  // persis aplikasi scanner modern. Model belum siap -> tunggu lalu jalan sendiri.
  if (model) {
    setTimeout(() => { if (sourceImage === img) executePipeline(); }, 120);
  } else {
    const wait = setInterval(() => {
      if (model) { clearInterval(wait); if (sourceImage === img) executePipeline(); }
    }, 100);
  }
}

function resetResults() {
  $('#outJavanese').textContent = '—';
  $('#outLatin').textContent = '—';
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

function executePipeline(roi) {
  if (!sourceImage || !model) return;
  btnRun.disabled = true;
  showScanFx(roi ? 'Memindai area pilihan…' : 'Menganalisis…');

  setTimeout(() => {
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

      // Tahap 1: Grayscale (ITU-R BT.601)
      const gray = new Uint8Array(w * h);
      const data = imgData.data;
      for (let y = 0; y < h; y++) {
        const srcRow = ((ry + y) * CW + rx) * 4;
        const dstRow = y * w;
        for (let x = 0; x < w; x++) {
          const i = srcRow + x * 4;
          gray[dstRow + x] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
        }
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
      labStages.binary = roiFullBin(morph, CW, CH, rx, ry, w, h);
      cacheSvmMontage(detections, model.size);

      // Selesai
      updateProgress(8, 'Selesai!');
      setTimeout(() => { progressBox.hidden = true; }, 300);
      hideScanFx();

      redrawMainCanvas();
      renderOutputs();
    } catch (err) {
      console.error(err);
      alert(`Terjadi kesalahan pemrosesan: ${err.message}`);
      progressBox.hidden = true;
      hideScanFx();
    } finally {
      btnRun.disabled = false;
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
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) handleImageUpload(f);
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

  if ($('#showBin').checked && labStages.binary) {
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
  } else {
    mainCtx.drawImage(sourceImage, 0, 0, w, h);
  }

  // Overlay Kotak Deteksi Karakter
  if ($('#showBoxes').checked && currentDetections.length) {
    mainCtx.lineWidth = Math.max(1.5, w / 600);
    mainCtx.font = `bold ${Math.max(12, Math.round(w / 75))}px 'Plus Jakarta Sans', sans-serif`;

    currentDetections.forEach((det, idx) => {
      const [x0, y0, x1, y1] = det.box;
      const bw = x1 - x0;
      const bh = y1 - y0;

      mainCtx.strokeStyle = det.isUncertain ? '#d1563b' : '#c8963e';
      mainCtx.strokeRect(x0, y0, bw, bh);

      // Label mini di atas kotak
      mainCtx.fillStyle = det.isUncertain ? '#d1563b' : '#1f2a44';
      mainCtx.fillRect(x0, Math.max(0, y0 - 18), 34, 18);
      mainCtx.fillStyle = '#f4ead5';
      mainCtx.fillText(det.label, x0 + 4, Math.max(12, y0 - 4));
    });
  }
}

function renderOutputs() {
  if (!currentDetections.length) {
    resetResults();
    return;
  }

  const javaneseText = currentDetections.map(d => d.unicode).join(' ');
  const latinText = currentDetections.map(d => d.label).join(' ');
  const avgConf = Math.round(
    (currentDetections.reduce((a, b) => a + b.confidence, 0) / currentDetections.length) * 100
  );

  $('#outJavanese').textContent = javaneseText;
  $('#outLatin').textContent = latinText;
  const meterVal = $('#meterVal');
  if (meterVal) meterVal.textContent = `${avgConf}%`;
  const meterFill = $('#meterFill');
  if (meterFill) meterFill.style.width = `${avgConf}%`;
  const meterWrap = $('#meterWrap');
  if (meterWrap) meterWrap.hidden = false;

  // Grid Karakter Glif
  const grid = $('#glyphsGrid');
  grid.replaceChildren();

  currentDetections.forEach((det, idx) => {
    const card = document.createElement('div');
    card.className = `glyph-card ${det.isUncertain ? 'uncertain' : ''}`;
    card.title = `Ketuk untuk mengoreksi aksara #${idx + 1}`;

    const c = document.createElement('canvas');
    c.width = model.size;
    c.height = model.size;
    const ctx = c.getContext('2d');
    const id = ctx.createImageData(model.size, model.size);
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
}

// Modal Koreksi Manual 20 Aksara
function setupCorrectionModal() {
  const modal = $('#correctionModal');
  modal.hidden = true;
  modal.classList.add('is-hidden');
  const grid = $('#correctionGrid');
  grid.replaceChildren();

  AKSARA_DATA.forEach(a => {
    const btn = document.createElement('button');
    btn.className = 'correction-btn';
    btn.innerHTML = `<span class="aksara-glyph">${a.unicode}</span><span class="aksara-name">${a.latin}</span>`;
    btn.addEventListener('click', () => {
      if (selectedGlyphIndex >= 0 && selectedGlyphIndex < currentDetections.length) {
        currentDetections[selectedGlyphIndex].label = a.latin;
        currentDetections[selectedGlyphIndex].unicode = a.unicode;
        currentDetections[selectedGlyphIndex].confidence = 1.0;
        currentDetections[selectedGlyphIndex].isUncertain = false;
        renderOutputs();
        redrawMainCanvas();
      }
      closeCorrectionModal();
    });
    grid.append(btn);
  });

  $('#closeModal').addEventListener('click', closeCorrectionModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeCorrectionModal(); });
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
  $('#modalTargetLabel').textContent = `Koreksi Aksara #${index + 1} (Saat ini: ${det.label} / ${det.unicode})`;
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

// Kamus Aksara
function renderDictionary() {
  const grid = $('#dictionaryGrid');
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

// Utilitas Clipboard & Unduh
// Ekspor hasil deteksi sebagai JSON (PRD: Salin Unicode, Salin Latin, Unduh JSON).
function downloadJson() {
  if (!currentDetections.length) return;
  const payload = {
    aplikasi: 'AksaraScan',
    versi: '1.0.0',
    sumber: lastSourceName || null,
    model: { algoritma: 'SVM Linear OvR', jumlahKelas: model.classes.length, akurasiUji: model.accuracy },
    rataRataKeyakinan: currentDetections.reduce((a, b) => a + b.confidence, 0) / currentDetections.length,
    teksUnicode: currentDetections.map(d => d.unicode).join(' '),
    transliterasiLatin: currentDetections.map(d => d.label).join(' '),
    deteksi: currentDetections.map(d => ({
      index: d.index + 1,
      label: d.label,
      unicode: d.unicode,
      confidence: d.confidence,
      ragu: d.isUncertain,
      kotak: d.box
    }))
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.download = `AksaraScan_${Date.now()}.json`;
  link.href = URL.createObjectURL(blob);
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function copyToClipboard(text, msg) {
  if (!text || text === '—') return;
  navigator.clipboard.writeText(text).then(() => {
    alert(msg);
  });
}

// Inisialisasi saat DOM siap
document.addEventListener('DOMContentLoaded', initApp);