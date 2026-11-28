/**
 * AksaraScan v4.0 — Stage-2 Reference Prototype Metric Verifier & Penolak Non-Aksara (Browser WASM)
 *
 * Memuat secara lazy:
 *   - `./aksara_embed.onnx` (0.77 MB, [B, 3, 64, 64] -> { emb: [B, 64], logits: [B, 56] })
 *   - `./reference_prototypes.json` (176 centroid Spherical K-Means + miniatur 28x28)
 *   - `./stage2_calibration.json` (ambang terkalibrasi semata pada split validation)
 *
 * Aman terhadap kegagalan muat: bila salah satu aset gagal/diblokir, mengembalikan
 * status fallback sehingga detektor Tahap-1 tetap berjalan normal tanpa crash.
 */
(function (global) {
  'use strict';

  const NON_AKSARA_CID = 55;
  const DEFAULT_CALIBRATION = {
    trigger_lo: 0.18,
    trigger_hi: 0.62,
    margin_trigger: 0.15,
    vet_all_non_aksara: true,
    temp_s2: 1.25,
    proto_alpha: 14.0,
    w_det: 0.62,
    w_cls: 0.65,
    base_conf_min: 0.25,
    recover_s2_min: 1.01,
    relabel_delta: 0.08,
    relabel_s2_min: 0.45,
    neg_reject_thr_lo: 0.48,
    neg_reject_thr_hi: 0.62,
    neg_hard_thr: 0.88,
    abstain_thr: 0.34,
    abstain_margin: 0.12,
    abstain_proto_min: 0.42,
  };

  const state = {
    session: null,
    bank: null,
    protoMats: null, // Array len 56 of Array of { vec: Float32Array, style: string, thumb: string }
    cal: { ...DEFAULT_CALIBRATION },
    ready: false,
    loading: false,
    fallbackReason: null,
  };

  function softmax(arr, scale = 1.0, offset = 0, len = arr.length) {
    let mx = -Infinity;
    for (let i = 0; i < len; i++) {
      const v = arr[offset + i] * scale;
      if (v > mx) mx = v;
    }
    const out = new Float32Array(len);
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const e = Math.exp(arr[offset + i] * scale - mx);
      out[i] = e;
      sum += e;
    }
    const inv = sum > 0 ? 1.0 / sum : 1.0;
    for (let i = 0; i < len; i++) out[i] *= inv;
    return out;
  }

  function buildPrototypeIndex(bank) {
    const mats = new Array(56);
    for (let c = 0; c < 56; c++) {
      const clsObj = (bank.classes && bank.classes[c]) || {};
      const thumbs = {};
      (clsObj.thumbs || []).forEach((t) => { thumbs[t.style] = t.png || ''; });
      const defaultThumb = (clsObj.thumbs && clsObj.thumbs[0] && clsObj.thumbs[0].png) || '';
      mats[c] = (clsObj.prototypes || []).map((p) => {
        const vec = new Float32Array(p.vec);
        let norm = 0;
        for (let k = 0; k < vec.length; k++) norm += vec[k] * vec[k];
        norm = Math.sqrt(Math.max(1e-12, norm));
        for (let k = 0; k < vec.length; k++) vec[k] /= norm;
        return {
          style: p.style || 'ref',
          vec,
          thumbPng: thumbs[p.style] || defaultThumb,
        };
      });
    }
    return mats;
  }

  async function init(opts = {}) {
    if (state.ready) return { ok: true, cal: state.cal };
    if (state.loading) return { ok: false, fallbackReason: 'Memuat…' };
    if (typeof ort === 'undefined') {
      state.fallbackReason = 'ONNX Runtime tidak tersedia';
      return { ok: false, fallbackReason: state.fallbackReason };
    }
    state.loading = true;
    state.fallbackReason = null;

    const onnxPath = opts.onnxPath || './aksara_embed.onnx';
    const protoPath = opts.prototypesPath || './reference_prototypes.json';
    const calPath = opts.calibrationPath || './stage2_calibration.json';

    try {
      const [protoRes, calRes] = await Promise.all([
        fetch(protoPath),
        fetch(calPath).catch(() => null),
      ]);
      if (!protoRes || !protoRes.ok) {
        throw new Error(`Prototipe referensi gagal dimuat (${protoRes ? protoRes.status : 'network'})`);
      }
      const bank = await protoRes.json();
      if (!bank || !Array.isArray(bank.classes) || bank.classes.length < 56) {
        throw new Error('Skema reference_prototypes.json tidak valid');
      }

      if (calRes && calRes.ok) {
        const calJson = await calRes.json();
        state.cal = { ...DEFAULT_CALIBRATION, ...calJson };
      }

      const sess = await ort.InferenceSession.create(onnxPath, {
        executionProviders: ['wasm'],
      });

      state.bank = bank;
      state.protoMats = buildPrototypeIndex(bank);
      state.session = sess;
      state.ready = true;
      state.loading = false;
      return { ok: true, cal: state.cal, totalPrototypes: bank.total_prototypes || 176 };
    } catch (err) {
      state.ready = false;
      state.loading = false;
      state.fallbackReason = err && err.message ? err.message : String(err);
      console.warn('[Stage-2 Verifier Fallback] Kembali ke mode detektor Tahap-1 saja:', state.fallbackReason);
      return { ok: false, fallbackReason: state.fallbackReason };
    }
  }

  /**
   * Ekstrak tensor [N, 3, 64, 64] aspect-preserving median-border pad
   * identik matematis dengan `letterbox_crop_patch` di train_stage2_embedder.py.
   */
  function extractBatchTensor64(srcCanvas, srcW, srcH, candidateBoxes, padFraction = 0.10) {
    const N = candidateBoxes.length;
    const out = new Float32Array(N * 3 * 64 * 64);
    const tmp = document.createElement('canvas');
    tmp.width = 64;
    tmp.height = 64;
    const tctx = tmp.getContext('2d', { willReadFrequently: true });
    const rawTmp = document.createElement('canvas');
    const rctx = rawTmp.getContext('2d', { willReadFrequently: true });

    const scaleX = (srcCanvas.naturalWidth || srcCanvas.width || srcW) / Math.max(1, srcW);
    const scaleY = (srcCanvas.naturalHeight || srcCanvas.height || srcH) / Math.max(1, srcH);

    for (let i = 0; i < N; i++) {
      const [bx0, by0, bx1, by1] = candidateBoxes[i];
      const bw = Math.max(2, bx1 - bx0);
      const bh = Math.max(2, by1 - by0);
      const sx0 = Math.max(0, Math.round((bx0 - padFraction * bw) * scaleX));
      const sy0 = Math.max(0, Math.round((by0 - padFraction * bh) * scaleY));
      const sx1 = Math.min(srcCanvas.naturalWidth || srcCanvas.width || srcW, Math.round((bx1 + padFraction * bw) * scaleX));
      const sy1 = Math.min(srcCanvas.naturalHeight || srcCanvas.height || srcH, Math.round((by1 + padFraction * bh) * scaleY));
      const cw = Math.max(2, sx1 - sx0);
      const ch = Math.max(2, sy1 - sy0);

      // Ambil warna median tepi dari crop untuk padding yang halus
      rawTmp.width = Math.min(96, cw);
      rawTmp.height = Math.min(96, ch);
      rctx.drawImage(srcCanvas, sx0, sy0, cw, ch, 0, 0, rawTmp.width, rawTmp.height);
      const rPx = rctx.getImageData(0, 0, rawTmp.width, rawTmp.height).data;
      const rw = rawTmp.width, rh = rawTmp.height;
      const bR = [], bG = [], bB = [];
      for (let x = 0; x < rw; x++) {
        let pTop = x * 4, pBot = ((rh - 1) * rw + x) * 4;
        bR.push(rPx[pTop], rPx[pBot]);
        bG.push(rPx[pTop + 1], rPx[pBot + 1]);
        bB.push(rPx[pTop + 2], rPx[pBot + 2]);
      }
      for (let y = 0; y < rh; y++) {
        let pL = (y * rw) * 4, pR = (y * rw + rw - 1) * 4;
        bR.push(rPx[pL], rPx[pR]);
        bG.push(rPx[pL + 1], rPx[pR + 1]);
        bB.push(rPx[pL + 2], rPx[pR + 2]);
      }
      bR.sort((a, b) => a - b);
      bG.sort((a, b) => a - b);
      bB.sort((a, b) => a - b);
      const mid = bR.length >> 1;
      tctx.fillStyle = `rgb(${bR[mid] ?? 235}, ${bG[mid] ?? 235}, ${bB[mid] ?? 235})`;
      tctx.fillRect(0, 0, 64, 64);

      const usable = 60;
      const sc = Math.min(usable / cw, usable / ch);
      const nw = Math.max(2, Math.min(usable, Math.round(cw * sc)));
      const nh = Math.max(2, Math.min(usable, Math.round(ch * sc)));
      const ox = Math.floor((64 - nw) / 2);
      const oy = Math.floor((64 - nh) / 2);
      tctx.drawImage(srcCanvas, sx0, sy0, cw, ch, ox, oy, nw, nh);

      const px = tctx.getImageData(0, 0, 64, 64).data;
      const baseOff = i * 3 * 4096;
      const plane = 4096;
      for (let k = 0; k < plane; k++) {
        const p = k * 4;
        out[baseOff + k] = px[p] / 127.5 - 1.0;
        out[baseOff + plane + k] = px[p + 1] / 127.5 - 1.0;
        out[baseOff + 2 * plane + k] = px[p + 2] / 127.5 - 1.0;
      }
    }
    return new ort.Tensor('float32', out, [N, 3, 64, 64]);
  }

  /**
   * Verifikasi kandidat deteksi Tahap-1 menggunakan aksara_embed.onnx + reference_prototypes.json.
   * Menolak distraktor Latin/tembok (kelas #55) & memperbarui topK + referenceMatches[].
   */
  async function verifyDetections(candidates, srcCanvas, srcW, srcH, yoloMeta) {
    if (!state.ready || !candidates || candidates.length === 0) {
      return {
        detections: candidates || [],
        stats: {
          active: false,
          fallbackReason: state.fallbackReason,
          verified: 0,
          relabeled: 0,
          rejectedNonAksara: 0,
          abstained: 0,
          stage2Ms: 0,
        },
      };
    }

    const t0 = performance.now();
    const cfg = state.cal;
    const nc = yoloMeta.classes.length; // 55
    const toVerify = [];
    for (let i = 0; i < candidates.length; i++) {
      const d = candidates[i];
      const top2 = d.topK && d.topK.length >= 2 ? d.topK[1].prob : 0;
      const margin = d.confidence - top2;
      const isAmbig = (d.confidence >= cfg.trigger_lo && d.confidence <= cfg.trigger_hi) || (margin < cfg.margin_trigger);
      if (isAmbig || cfg.vet_all_non_aksara) {
        toVerify.push({ origIdx: i, isAmbig, box: d.box });
      }
    }

    if (toVerify.length === 0) {
      return {
        detections: candidates.filter((d) => d.confidence >= cfg.base_conf_min),
        stats: { active: true, verified: 0, relabeled: 0, rejectedNonAksara: 0, abstained: 0, stage2Ms: 0 },
      };
    }

    const tensor = extractBatchTensor64(srcCanvas, srcW, srcH, toVerify.map((x) => x.box));
    const inName = state.session.inputNames[0];
    const res = await state.session.run({ [inName]: tensor });
    const embArr = res[state.session.outputNames[0]].data; // [M, 64]
    const logArr = res[state.session.outputNames[1]].data; // [M, 56]

    const idxMap = new Map();
    toVerify.forEach((item, row) => idxMap.set(item.origIdx, { row, isAmbig: item.isAmbig }));

    const keptOut = [];
    let relabeled = 0, rejectedNonAksara = 0, abstained = 0;

    for (let i = 0; i < candidates.length; i++) {
      const d = candidates[i];
      if (!idxMap.has(i)) {
        if (d.confidence >= cfg.base_conf_min) keptOut.push(d);
        continue;
      }

      const { row, isAmbig } = idxMap.get(i);
      const embOff = row * 64;
      const logOff = row * 56;
      const pS2 = softmax(logArr, 1.0 / cfg.temp_s2, logOff, 56);

      // Hitung kemiripan kosinus maksimum ke tiap kelas & simpan kecocokan prototipe terbaik
      const protoMaxSim = new Float32Array(56);
      const bestProtoMeta = new Array(56);
      for (let c = 0; c < 56; c++) {
        let bestS = -1.0;
        let bestP = null;
        for (const pr of (state.protoMats[c] || [])) {
          let dot = 0;
          for (let k = 0; k < 64; k++) dot += embArr[embOff + k] * pr.vec[k];
          if (dot > bestS) {
            bestS = dot;
            bestP = pr;
          }
        }
        protoMaxSim[c] = bestS;
        bestProtoMeta[c] = bestP;
      }
      const pProto = softmax(protoMaxSim, cfg.proto_alpha, 0, 56);

      let maxS2Aks = 0, maxPrAks = 0;
      for (let c = 0; c < nc; c++) {
        if (pS2[c] > maxS2Aks) maxS2Aks = pS2[c];
        if (pProto[c] > maxPrAks) maxPrAks = pProto[c];
      }
      const negBinCls = pS2[NON_AKSARA_CID] / Math.max(1e-8, pS2[NON_AKSARA_CID] + maxS2Aks);
      const negBinPr = pProto[NON_AKSARA_CID] / Math.max(1e-8, pProto[NON_AKSARA_CID] + maxPrAks);
      const s2NegProb = cfg.w_cls * negBinCls + (1.0 - cfg.w_cls) * negBinPr;

      const negThr = d.confidence >= 0.55 ? cfg.neg_reject_thr_hi : cfg.neg_reject_thr_lo;
      if (s2NegProb >= negThr || negBinCls >= cfg.neg_hard_thr) {
        rejectedNonAksara++;
        continue;
      }

      // Vektor skor detektor penuh (disediakan oleh executeYoloPipeline bila tersedia)
      const detVec = d.detVec || new Float32Array(nc);
      if (!d.detVec) {
        (d.topK || []).forEach((tk) => { if (tk.classIdx >= 0 && tk.classIdx < nc) detVec[tk.classIdx] = tk.prob; });
        detVec[d.classIdx] = d.confidence;
      }

      const fused = new Float32Array(nc);
      let bestC = 0, bestFused = -1, secFused = -1;
      for (let c = 0; c < nc; c++) {
        const s2Comb = cfg.w_cls * pS2[c] + (1.0 - cfg.w_cls) * pProto[c];
        const fv = cfg.w_det * detVec[c] + (1.0 - cfg.w_det) * s2Comb;
        fused[c] = fv;
        if (fv > bestFused) {
          secFused = bestFused;
          bestFused = fv;
          bestC = c;
        } else if (fv > secFused) {
          secFused = fv;
        }
      }

      const s2CombBest = cfg.w_cls * pS2[bestC] + (1.0 - cfg.w_cls) * pProto[bestC];
      if (d.confidence < cfg.base_conf_min) {
        if (Math.max(s2CombBest, pS2[bestC]) < cfg.recover_s2_min) {
          continue;
        }
      }

      const d2 = { ...d };
      delete d2.detVec;

      // Perbarui daftar Top-3 & Reference Matches untuk Stage-2 & Inspektor Referensi Fase 4
      const ranked = [];
      for (let c = 0; c < nc; c++) {
        ranked.push({
          classIdx: c,
          label: yoloMeta.classes[c],
          category: yoloMeta.categories[c],
          prob: Number(fused[c].toFixed(4)),
          detProb: Number(detVec[c].toFixed(4)),
          s2Prob: Number(pS2[c].toFixed(4)),
          protoSim: Number(protoMaxSim[c].toFixed(4)),
        });
      }
      ranked.sort((a, b) => b.prob - a.prob);
      d2.topK = ranked.slice(0, 3);
      d2.referenceMatches = ranked.slice(0, 3).map((item) => {
        const pm = bestProtoMeta[item.classIdx];
        const simPct = Math.max(0, Math.min(100, Math.round(item.protoSim * 100)));
        return {
          classIdx: item.classIdx,
          label: item.label,
          style: pm ? pm.style : 'ref',
          similarity: item.protoSim,
          similarityPct: simPct,
          thumbPng: pm ? pm.thumbPng : '',
        };
      });
      d2.nonAksaraProb = Number(s2NegProb.toFixed(4));

      if (isAmbig) {
        const oldScore = fused[d.classIdx];
        if (bestC !== d.classIdx && (bestFused - oldScore) >= cfg.relabel_delta && pS2[bestC] >= cfg.relabel_s2_min) {
          d2.classIdx = bestC;
          d2.label = yoloMeta.classes[bestC];
          d2.category = yoloMeta.categories[bestC];
          d2.unicode = yoloMeta.unicode[d2.label] || '';
          relabeled++;
        }
        d2.confidence = Number(bestFused.toFixed(4));
        const margin = bestFused - Math.max(0, secFused);
        const curSim = protoMaxSim[d2.classIdx];
        d2.protoSim = Number(curSim.toFixed(4));
        if (bestFused < cfg.abstain_thr || (margin < cfg.abstain_margin && curSim < cfg.abstain_proto_min)) {
          d2.isUncertain = true;
          abstained++;
        } else {
          d2.isUncertain = false;
        }
      } else {
        d2.protoSim = Number(protoMaxSim[d2.classIdx].toFixed(4));
      }

      keptOut.push(d2);
    }

    const stage2Ms = Math.round(performance.now() - t0);
    return {
      detections: keptOut,
      stats: {
        active: true,
        verified: toVerify.length,
        relabeled,
        rejectedNonAksara,
        abstained,
        stage2Ms,
      },
    };
  }

  function getStatus() {
    return {
      ready: state.ready,
      loading: state.loading,
      fallbackReason: state.fallbackReason,
      totalPrototypes: state.bank ? (state.bank.total_prototypes || 0) : 0,
      cal: state.cal,
    };
  }

  function getBank() {
    return state.bank;
  }

  global.AksaraStage2 = {
    init,
    verifyDetections,
    getStatus,
    getBank,
  };
})(typeof window !== 'undefined' ? window : globalThis);
