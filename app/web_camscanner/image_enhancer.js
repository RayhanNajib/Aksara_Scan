/**
 * AksaraScan v4.0 — Engine Praproses Adaptif, Pemisah Tinta Warna CIELAB,
 * Penekan Garis Buku/Dinding (Tanaka-inspired), Estimasi Blur, & Dekonvolusi Ringan.
 *
 * Berjalan 100% di sisi klien (Main Thread, Web Worker Offscreen, atau Node.js unit test)
 * tanpa menyalin aset pihak ketiga atau melanggar privasi jaringan.
 */
(function (global) {
  'use strict';

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function toLumaAndLab(rgba, N) {
    const luma = new Float32Array(N);
    const aCh = new Float32Array(N);
    const bCh = new Float32Array(N);
    for (let i = 0, p = 0; i < N; i++, p += 4) {
      const r = rgba[p];
      const g = rgba[p + 1];
      const b = rgba[p + 2];
      luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      // Kanal oponen kromatik cepat setara CIE (a*: merah-hijau, b*: kuning-biru)
      aCh[i] = (r - g) * 0.5;
      bCh[i] = ((r + g) * 0.5 - b) * 0.5;
    }
    return { luma, aCh, bCh };
  }

  /**
   * Estimasi latar belakang kertas yang mulus (resolusi 1/4 + dilasi 3x3 + box-blur berulang)
   * untuk Luma, kanal a*, dan kanal b*.
   */
  function estimateBackgroundMaps(luma, aCh, bCh, W, H, ds = 4) {
    const SW = Math.max(4, Math.ceil(W / ds));
    const SH = Math.max(4, Math.ceil(H / ds));
    const sL = new Float32Array(SW * SH);
    const sA = new Float32Array(SW * SH);
    const sB = new Float32Array(SW * SH);

    for (let sy = 0; sy < SH; sy++) {
      const y0 = sy * ds, y1 = Math.min(H, y0 + ds);
      for (let sx = 0; sx < SW; sx++) {
        const x0 = sx * ds, x1 = Math.min(W, x0 + ds);
        let bestL = -1, bestA = 0, bestB = 0;
        for (let yy = y0; yy < y1; yy++) {
          const row = yy * W;
          for (let xx = x0; xx < x1; xx++) {
            const idx = row + xx;
            if (luma[idx] > bestL) {
              bestL = luma[idx];
              bestA = aCh[idx];
              bestB = bCh[idx];
            }
          }
        }
        const si = sy * SW + sx;
        sL[si] = bestL;
        sA[si] = bestA;
        sB[si] = bestB;
      }
    }

    // Dilasi 3x3 pada Luma untuk menyingkirkan goresan tebal dari estimasi kertas
    const dL = new Float32Array(SW * SH);
    const dA = new Float32Array(SW * SH);
    const dB = new Float32Array(SW * SH);
    for (let sy = 0; sy < SH; sy++) {
      for (let sx = 0; sx < SW; sx++) {
        let bestL = -1, bestA = 0, bestB = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = clamp(sy + dy, 0, SH - 1);
          for (let dx = -1; dx <= 1; dx++) {
            const xx = clamp(sx + dx, 0, SW - 1);
            const idx = yy * SW + xx;
            if (sL[idx] > bestL) {
              bestL = sL[idx];
              bestA = sA[idx];
              bestB = sB[idx];
            }
          }
        }
        const si = sy * SW + sx;
        dL[si] = bestL;
        dA[si] = bestA;
        dB[si] = bestB;
      }
    }

    const rad = Math.max(2, Math.min(12, Math.round(Math.min(SW, SH) / 10)));
    function boxBlur2D(arr) {
      const tmp = new Float32Array(SW * SH);
      const out = new Float32Array(SW * SH);
      for (let y = 0; y < SH; y++) {
        const row = y * SW;
        for (let x = 0; x < SW; x++) {
          let sum = 0, cnt = 0;
          const x0 = Math.max(0, x - rad), x1 = Math.min(SW - 1, x + rad);
          for (let xx = x0; xx <= x1; xx++) { sum += arr[row + xx]; cnt++; }
          tmp[row + x] = sum / cnt;
        }
      }
      for (let y = 0; y < SH; y++) {
        const y0 = Math.max(0, y - rad), y1 = Math.min(SH - 1, y + rad);
        for (let x = 0; x < SW; x++) {
          let sum = 0, cnt = 0;
          for (let yy = y0; yy <= y1; yy++) { sum += tmp[yy * SW + x]; cnt++; }
          out[y * SW + x] = sum / cnt;
        }
      }
      return out;
    }

    const bgL = boxBlur2D(dL);
    const bgA = boxBlur2D(dA);
    const bgB = boxBlur2D(dB);
    let minBg = 255, maxBg = 0;
    for (let i = 0; i < bgL.length; i++) {
      if (bgL[i] < minBg) minBg = bgL[i];
      if (bgL[i] > maxBg) maxBg = bgL[i];
    }
    return { SW, SH, ds, bgL, bgA, bgB, minBg, maxBg, shadowRange: maxBg - minBg };
  }

  /**
   * Estimator blur objektif berbasis:
   *  1) Varians Laplacian & kemiringan tepi ternormalisasi kontras (normEdgeSlope)
   *  2) Lebar transisi tepi 20%->80% (edgeSpreadPx)
   *  3) Skor tinta kromatik (coloredInkRatio) & clutter garis lurus (lineClutterScore)
   *
   * @param {ImageData|{data:Uint8ClampedArray|Uint8Array,width:number,height:number}} imageData
   */
  function estimateBlur(imageData) {
    const W = imageData.width;
    const H = imageData.height;
    const rgba = imageData.data;
    const N = W * H;
    const { luma, aCh, bCh } = toLumaAndLab(rgba, N);

    // Histogram luminans + kromatiks
    const hist = new Uint32Array(256);
    let medA = 0, medB = 0;
    let coloredPx = 0;
    for (let i = 0; i < N; i++) {
      hist[clamp(Math.round(luma[i]), 0, 255)]++;
      medA += aCh[i];
      medB += bCh[i];
    }
    medA /= Math.max(1, N);
    medB /= Math.max(1, N);

    let cum = 0;
    let p002Ink = 0, p02Ink = 0, p05Ink = 0, p50 = 128, p90Bg = 240, p99InkDark = 255;
    const c002 = Math.max(8, Math.floor(N * 0.002));
    const c02 = Math.floor(N * 0.02), c05 = Math.floor(N * 0.05), c50 = Math.floor(N * 0.50), c90 = Math.floor(N * 0.90), c998 = Math.min(N - 8, Math.floor(N * 0.998));
    let got002 = false, got02 = false, got05 = false, got50 = false, got90 = false;
    for (let b = 0; b < 256; b++) {
      cum += hist[b];
      if (!got002 && cum >= c002) { p002Ink = b; got002 = true; }
      if (!got02 && cum >= c02) { p02Ink = b; got02 = true; }
      if (!got05 && cum >= c05) { p05Ink = b; got05 = true; }
      if (!got50 && cum >= c50) { p50 = b; got50 = true; }
      if (!got90 && cum >= c90) { p90Bg = b; got90 = true; }
      if (cum >= c998) { p99InkDark = b; break; }
    }

    const isDarkMode = p50 < 112;
    // Evaluasi pada orientasi terang (tinta gelap di kertas cerah) agar estimasi tepi konsisten
    const work = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      work[i] = isDarkMode ? 255 - luma[i] : luma[i];
      const dc = Math.hypot(aCh[i] - medA, bCh[i] - medB);
      if (dc > 16) coloredPx++;
    }
    const bgRef = isDarkMode ? (255 - p05Ink) : p90Bg;
    const inkRef = isDarkMode ? (255 - p99InkDark) : p002Ink;
    const contrastSpan = Math.max(35, bgRef - inkRef);
    const minGradThr = Math.max(2.0, contrastSpan * 0.015);
    const minLocalSpan = contrastSpan * 0.28;

    // Hitung Laplacian & lebar transisi tepi kontigu (Marziliano-inspired, dwi-sumbu X/Y)
    // agar blur Gaussian maupun Motion blur terarah terukur monoton tanpa tersaring gradien lemah.
    let lapAbsSum = 0, lapSqSum = 0, edgeCnt = 0;
    let gradMaxSum = 0;
    let gradSumX = 0, cntX = 0, gradSumY = 0, cntY = 0;
    const spreads = [];
    const spreadsX = [];
    const spreadsY = [];

    for (let y = 2; y < H - 2; y += 2) {
      const row = y * W;
      for (let x = 2; x < W - 2; x += 2) {
        const idx = row + x;
        const c = work[idx];
        const gx = Math.abs(work[idx + 1] - work[idx - 1]) * 0.5;
        const gy = Math.abs(work[idx + W] - work[idx - W]) * 0.5;
        const gMax = gx > gy ? gx : gy;
        if (gMax < minGradThr) continue;

        const isHoriz = gx >= gy;
        let loV = c, hiV = c;
        const maxRad = 8;
        if (isHoriz) {
          const xStart = Math.max(0, x - maxRad), xEnd = Math.min(W - 1, x + maxRad);
          for (let xx = xStart; xx <= xEnd; xx++) {
            const v = work[row + xx];
            if (v < loV) loV = v;
            if (v > hiV) hiV = v;
          }
        } else {
          const yStart = Math.max(0, y - maxRad), yEnd = Math.min(H - 1, y + maxRad);
          for (let yy = yStart; yy <= yEnd; yy++) {
            const v = work[yy * W + x];
            if (v < loV) loV = v;
            if (v > hiV) hiV = v;
          }
        }

        const span = hiV - loV;
        if (span < minLocalSpan) continue;
        // Harus berada pada zona lereng aktif tepi (menghindari datar latar / inti tebal)
        if (c < loV + span * 0.08 || c > hiV - span * 0.08) continue;

        const lap = Math.abs(4 * c - work[idx - 1] - work[idx + 1] - work[idx - W] - work[idx + W]) / contrastSpan;
        lapAbsSum += lap;
        lapSqSum += lap * lap;
        const nSlope = gMax / contrastSpan;
        gradMaxSum += nSlope;
        edgeCnt++;
        if (isHoriz) { gradSumX += gx / contrastSpan; cntX++; }
        else { gradSumY += gy / contrastSpan; cntY++; }

        if (spreads.length < 3000) {
          const tLo = loV + span * 0.16;
          const tHi = loV + span * 0.84;
          let inTrans = 1;
          // Jalan keluar kiri/atas & kanan/bawah secara kontigu sepanjang lereng tunggal
          for (let s = -1; s >= -maxRad; s--) {
            const xx = isHoriz ? x + s : x;
            const yy = isHoriz ? y : y + s;
            if (xx < 0 || yy < 0) break;
            const v = work[yy * W + xx];
            if (v >= tLo && v <= tHi) inTrans++;
            else break;
          }
          for (let s = 1; s <= maxRad; s++) {
            const xx = isHoriz ? x + s : x;
            const yy = isHoriz ? y : y + s;
            if (xx >= W || yy >= H) break;
            const v = work[yy * W + xx];
            if (v >= tLo && v <= tHi) inTrans++;
            else break;
          }
          const sp = Math.max(1.0, inTrans + 0.5);
          spreads.push(sp);
          if (isHoriz && spreadsX.length < 1500) spreadsX.push(sp);
          else if (!isHoriz && spreadsY.length < 1500) spreadsY.push(sp);
        }
      }
    }

    const rawMeanSlope = edgeCnt > 10 ? gradMaxSum / edgeCnt : 0.42;
    const slopeX = cntX >= 12 ? gradSumX / cntX : rawMeanSlope;
    const slopeY = cntY >= 12 ? gradSumY / cntY : rawMeanSlope;
    const maxAxisSlope = (cntX >= 12 && cntY >= 12) ? Math.max(slopeX, slopeY) : rawMeanSlope;
    const minAxisSlope = (cntX >= 12 && cntY >= 12) ? Math.min(slopeX, slopeY) : rawMeanSlope;
    const normEdgeSlope = 0.55 * rawMeanSlope + 0.45 * minAxisSlope;
    const axisSymmetry = Number((minAxisSlope / Math.max(0.01, maxAxisSlope)).toFixed(3));

    const meanNormLap = edgeCnt > 10 ? lapAbsSum / edgeCnt : 0.7;
    const normLapVar = edgeCnt > 10 ? Math.max(0, lapSqSum / edgeCnt - meanNormLap * meanNormLap) : 0.3;
    spreads.sort((a, b) => a - b);
    spreadsX.sort((a, b) => a - b);
    spreadsY.sort((a, b) => a - b);
    const medAll = spreads.length > 0 ? spreads[Math.floor(spreads.length * 0.5)] : 1.5;
    const medX = spreadsX.length >= 12 ? spreadsX[Math.floor(spreadsX.length * 0.5)] : medAll;
    const medY = spreadsY.length >= 12 ? spreadsY[Math.floor(spreadsY.length * 0.5)] : medAll;
    const edgeSpreadPx = (spreadsX.length >= 12 && spreadsY.length >= 12)
      ? 0.5 * medAll + 0.5 * Math.max(medX, medY)
      : medAll;

    // Deteksi indikasi garis lurus tipis kontinyu memanjang >=55% lebar tanpa salah mendeteksi kata tulisan tangan atau bayangan lebar
    let lineClusters = 0;
    let inCluster = false;
    const lineStepY = Math.max(1, Math.floor(H / 240));
    const darkThr = bgRef - Math.max(14, contrastSpan * 0.10);
    const dyOff = Math.min(6, Math.max(3, Math.floor(H / 35)));
    for (let y = dyOff; y < H - dyOff; y += lineStepY) {
      const row = y * W;
      const rowUp = (y - dyOff) * W;
      const rowDn = (y + dyOff) * W;
      let run = 0, maxRun = 0;
      for (let x = 0; x < W; x++) {
        const px = work[row + x];
        const rim = Math.min(work[rowUp + x], work[rowDn + x]);
        if (px < darkThr && (rim - px) >= 12) {
          run++;
          if (run > maxRun) maxRun = run;
        } else {
          run = 0;
        }
      }
      if (maxRun >= W * 0.55) {
        if (!inCluster) { lineClusters++; inCluster = true; }
      } else {
        inCluster = false;
      }
    }
    const lineClutterScore = Number(Math.min(1, lineClusters * 0.12).toFixed(4));

    // Estimasi variasi pencahayaan kasar pada latar
    const { minBg, maxBg, shadowRange } = estimateBackgroundMaps(work, aCh, bCh, W, H, 4);

    // Skor ketajaman komposit [0..1] terkalibrasi pada blur sintetis & nyata
    // normEdgeSlope: ~0.46 (tajam), ~0.27 (ringan), ~0.16 (sedang), <=0.10 (berat)
    // edgeSpreadPx:  ~1.5 (tajam), ~2.5 (ringan), ~3.6 (sedang), >=4.8 (berat)
    const slopeNorm = clamp((normEdgeSlope - 0.08) / 0.34, 0, 1);
    const spreadNorm = clamp(1.0 - (edgeSpreadPx - 1.5) / 3.8, 0, 1);
    const sharpnessScore = clamp(0.62 * slopeNorm + 0.38 * spreadNorm, 0, 1);

    let level = 'tajam';
    if (sharpnessScore < 0.24 || (normEdgeSlope < 0.145 && edgeSpreadPx >= 4.0)) {
      level = 'berat';
    } else if (sharpnessScore < 0.48 || normEdgeSlope < 0.235 || edgeSpreadPx >= 3.0) {
      level = 'sedang';
    } else if (sharpnessScore < 0.72 || normEdgeSlope < 0.345 || edgeSpreadPx >= 2.15) {
      level = 'ringan';
    }

    return {
      level,
      sharpnessScore: Number(sharpnessScore.toFixed(4)),
      normEdgeSlope: Number(normEdgeSlope.toFixed(4)),
      normLapVar: Number(normLapVar.toFixed(4)),
      edgeSpreadPx: Number(edgeSpreadPx.toFixed(2)),
      contrastSpan: Math.round(contrastSpan),
      bgMean: Math.round(p90Bg),
      p02Ink: Math.round(p02Ink),
      minBg: Math.round(minBg),
      maxBg: Math.round(maxBg),
      shadowRange: Math.round(shadowRange),
      isDarkMode,
      coloredInkRatio: Number((coloredPx / Math.max(1, N)).toFixed(4)),
      lineClutterScore,
      axisSymmetry,
    };
  }

  /**
   * Penekan garis buku/dinding memanjang berbasis kontinuitas arah (terinspirasi Tanaka et al.):
   * hanya mengangkat piksel yang menjadi bagian run lurus >= minRun dan BUKAN inti goresan lengkung tinggi.
   */
  function suppressLongStraightLines(luma, bgL, SW, SH, ds, W, H, strength = 0.78, includeVert = false) {
    const N = W * H;
    const delta = new Float32Array(N);
    for (let y = 0; y < H; y++) {
      const syf = Math.min(SH - 1.001, y / ds);
      const sy0 = Math.floor(syf), fy = syf - sy0, sy1 = Math.min(SH - 1, sy0 + 1);
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const sxf = Math.min(SW - 1.001, x / ds);
        const sx0 = Math.floor(sxf), fx = sxf - sx0, sx1 = Math.min(SW - 1, sx0 + 1);
        const bgVal =
          (bgL[sy0 * SW + sx0] * (1 - fx) + bgL[sy0 * SW + sx1] * fx) * (1 - fy) +
          (bgL[sy1 * SW + sx0] * (1 - fx) + bgL[sy1 * SW + sx1] * fx) * fy;
        delta[row + x] = bgVal - luma[row + x];
      }
    }

    // Erosi-Dilasi 1D Horizontal (pembukaan morfologis berorientasi garis buku/dinding penuh >=22% lebar)
    const hLen = Math.max(39, Math.round(W * 0.22)) | 1;
    const hHalf = hLen >> 1;
    const hErode = new Float32Array(N);
    const hOpen = new Float32Array(N);
    for (let y = 0; y < H; y++) {
      const row = y * W;
      for (let x = 0; x < W; x++) {
        let mn = 255;
        const x0 = Math.max(0, x - hHalf), x1 = Math.min(W - 1, x + hHalf);
        for (let xx = x0; xx <= x1; xx++) {
          const v = delta[row + xx];
          if (v < mn) mn = v;
        }
        hErode[row + x] = mn > 0 ? mn : 0;
      }
      for (let x = 0; x < W; x++) {
        let mx = 0;
        const x0 = Math.max(0, x - hHalf), x1 = Math.min(W - 1, x + hHalf);
        for (let xx = x0; xx <= x1; xx++) {
          const v = hErode[row + xx];
          if (v > mx) mx = v;
        }
        hOpen[row + x] = mx;
      }
    }

    // Pembukaan morfologis Vertikal hanya bila diminta eksplisit untuk garis tepi/retak dinding vertikal penuh
    const vOpen = new Float32Array(N);
    if (includeVert) {
      const vLen = Math.max(45, Math.round(H * 0.25)) | 1;
      const vHalf = vLen >> 1;
      const vErode = new Float32Array(N);
      for (let x = 0; x < W; x++) {
        for (let y = 0; y < H; y++) {
          let mn = 255;
          const y0 = Math.max(0, y - vHalf), y1 = Math.min(H - 1, y + vHalf);
          for (let yy = y0; yy <= y1; yy++) {
            const v = delta[yy * W + x];
            if (v < mn) mn = v;
          }
          vErode[y * W + x] = mn > 0 ? mn : 0;
        }
        for (let y = 0; y < H; y++) {
          let mx = 0;
          const y0 = Math.max(0, y - vHalf), y1 = Math.min(H - 1, y + vHalf);
          for (let yy = y0; yy <= y1; yy++) {
            const v = vErode[yy * W + x];
            if (v > mx) mx = v;
          }
          vOpen[y * W + x] = mx;
        }
      }
    }

    let suppressedCount = 0;
    const out = new Float32Array(luma);
    for (let y = 3; y < H - 3; y++) {
      const row = y * W;
      for (let x = 3; x < W - 3; x++) {
        const idx = row + x;
        const linePart = Math.max(hOpen[idx], vOpen[idx]);
        if (linePart < 6 || linePart > 72) continue;
        // Lindungi persimpangan huruf hingga radius +-3 piksel agar tulisan di dekat/atas garis buku tak terpotong!
        const nonLineH = hOpen[idx] >= vOpen[idx]
          ? Math.max(
              delta[idx - W] - hOpen[idx - W],
              delta[idx + W] - hOpen[idx + W],
              delta[idx - 2 * W] - hOpen[idx - 2 * W],
              delta[idx + 2 * W] - hOpen[idx + 2 * W],
              delta[idx - 3 * W] - hOpen[idx - 3 * W],
              delta[idx + 3 * W] - hOpen[idx + 3 * W]
            )
          : Math.max(
              delta[idx - 1] - vOpen[idx - 1],
              delta[idx + 1] - vOpen[idx + 1],
              delta[idx - 2] - vOpen[idx - 2],
              delta[idx + 2] - vOpen[idx + 2],
              delta[idx - 3] - vOpen[idx - 3],
              delta[idx + 3] - vOpen[idx + 3]
            );
        if (nonLineH > 12) continue;
        const dOwn = delta[idx];
        if (dOwn - linePart > 20) continue;
        const lift = Math.min(dOwn, linePart * strength);
        if (lift > 3) {
          out[idx] = Math.min(255, luma[idx] + lift);
          suppressedCount++;
        }
      }
    }
    return { luma: out, suppressedCount };
  }

  /**
   * CLAHE (Contrast Limited Adaptive Histogram Equalization) bilinear antar ubin pada Luma [0..255].
   */
  function applyClaheLuma(luma, W, H, gridX = 6, gridY = 6, clipLimit = 1.8, blend = 0.65) {
    const N = W * H;
    const gx = Math.max(2, Math.min(10, gridX));
    const gy = Math.max(2, Math.min(10, gridY));
    const luts = new Array(gy * gx);

    for (let ty = 0; ty < gy; ty++) {
      const y0 = Math.floor((ty * H) / gy);
      const y1 = Math.max(y0 + 1, Math.floor(((ty + 1) * H) / gy));
      for (let tx = 0; tx < gx; tx++) {
        const x0 = Math.floor((tx * W) / gx);
        const x1 = Math.max(x0 + 1, Math.floor(((tx + 1) * W) / gx));
        const hist = new Uint32Array(256);
        let area = 0;
        for (let y = y0; y < y1; y++) {
          const row = y * W;
          for (let x = x0; x < x1; x++) {
            hist[clamp(Math.round(luma[row + x]), 0, 255)]++;
            area++;
          }
        }
        const clip = Math.max(2, Math.round((clipLimit * area) / 256));
        let excess = 0;
        for (let b = 0; b < 256; b++) {
          if (hist[b] > clip) {
            excess += hist[b] - clip;
            hist[b] = clip;
          }
        }
        const incr = excess / 256;
        const lut = new Float32Array(256);
        let cdf = 0;
        for (let b = 0; b < 256; b++) {
          cdf += hist[b] + incr;
          lut[b] = clamp((cdf * 255.0) / Math.max(1, area), 0, 255);
        }
        luts[ty * gx + tx] = lut;
      }
    }

    const tileW = W / gx;
    const tileH = H / gy;
    const out = new Float32Array(N);
    for (let y = 0; y < H; y++) {
      const fyCenter = y / tileH - 0.5;
      const ty0 = clamp(Math.floor(fyCenter), 0, gy - 1);
      const ty1 = clamp(ty0 + 1, 0, gy - 1);
      const wy = clamp(fyCenter - ty0, 0, 1);
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const fxCenter = x / tileW - 0.5;
        const tx0 = clamp(Math.floor(fxCenter), 0, gx - 1);
        const tx1 = clamp(tx0 + 1, 0, gx - 1);
        const wx = clamp(fxCenter - tx0, 0, 1);

        const b = clamp(Math.round(luma[row + x]), 0, 255);
        const v00 = luts[ty0 * gx + tx0][b];
        const v10 = luts[ty0 * gx + tx1][b];
        const v01 = luts[ty1 * gx + tx0][b];
        const v11 = luts[ty1 * gx + tx1][b];
        const eq = (v00 * (1 - wx) + v10 * wx) * (1 - wy) + (v01 * (1 - wx) + v11 * wx) * wy;
        out[row + x] = luma[row + x] * (1 - blend) + eq * blend;
      }
    }
    return out;
  }

  function gaussianBlurSeparable(src, W, H, radius) {
    const N = W * H;
    const tmp = new Float32Array(N);
    const out = new Float32Array(N);
    if (radius <= 1) {
      // [1, 2, 1] / 4
      for (let y = 0; y < H; y++) {
        const r = y * W;
        for (let x = 0; x < W; x++) {
          const xm = x > 0 ? x - 1 : 0, xp = x < W - 1 ? x + 1 : W - 1;
          tmp[r + x] = (src[r + xm] + 2 * src[r + x] + src[r + xp]) * 0.25;
        }
      }
      for (let y = 0; y < H; y++) {
        const ym = y > 0 ? y - 1 : 0, yp = y < H - 1 ? y + 1 : H - 1;
        for (let x = 0; x < W; x++) {
          out[y * W + x] = (tmp[ym * W + x] + 2 * tmp[y * W + x] + tmp[yp * W + x]) * 0.25;
        }
      }
      return out;
    }
    // [1, 4, 6, 4, 1] / 16 (radius 2)
    for (let y = 0; y < H; y++) {
      const r = y * W;
      for (let x = 0; x < W; x++) {
        const x_2 = Math.max(0, x - 2), x_1 = Math.max(0, x - 1);
        const x1 = Math.min(W - 1, x + 1), x2 = Math.min(W - 1, x + 2);
        tmp[r + x] = (src[r + x_2] + 4 * src[r + x_1] + 6 * src[r + x] + 4 * src[r + x1] + src[r + x2]) / 16.0;
      }
    }
    for (let y = 0; y < H; y++) {
      const y_2 = Math.max(0, y - 2), y_1 = Math.max(0, y - 1);
      const y1 = Math.min(H - 1, y + 1), y2 = Math.min(H - 1, y + 2);
      for (let x = 0; x < W; x++) {
        out[y * W + x] = (tmp[y_2 * W + x] + 4 * tmp[y_1 * W + x] + 6 * tmp[y * W + x] + 4 * tmp[y1 * W + x] + tmp[y2 * W + x]) / 16.0;
      }
    }
    return out;
  }

  /**
   * Multi-Scale Unsharp Mask + opsional Richardson-Lucy 3 iterasi dengan pengaman halo/ringing
   * dan penjaga kontinuitas goresan tipis (inti gelap tidak diizinkan terpecah menjadi terang).
   */
  function applySharpenAndRL(luma, W, H, amount, useRL, rlIterations = 3, opts = {}) {
    const N = W * H;
    const inkProtectThr = opts.inkProtectThr !== undefined ? opts.inkProtectThr : 195;
    const noiseCoring = opts.noiseCoring !== undefined ? opts.noiseCoring : 5.0;
    let work = new Float32Array(luma);

    if (useRL && rlIterations > 0) {
      const iters = Math.min(4, Math.max(1, rlIterations));
      const obs = new Float32Array(N);
      for (let i = 0; i < N; i++) obs[i] = Math.max(1.0, work[i]);
      let est = new Float32Array(obs);

      for (let it = 0; it < iters; it++) {
        const conv = gaussianBlurSeparable(est, W, H, 1);
        const ratio = new Float32Array(N);
        for (let i = 0; i < N; i++) {
          ratio[i] = obs[i] / Math.max(1.0, conv[i]);
        }
        const corr = gaussianBlurSeparable(ratio, W, H, 1);
        for (let i = 0; i < N; i++) {
          const rawDelta = Math.abs(obs[i] - conv[i]);
          // Jangan perkuat noise latar datar (pixel di bawah ambang detail/coring)
          const edgeMask = rawDelta <= noiseCoring ? 0 : clamp((rawDelta - noiseCoring) / (noiseCoring * 1.5), 0, 1);
          const nextV = work[i] + edgeMask * (est[i] * corr[i] - work[i]);
          // Batasi ringing dan jaga inti tinta (termasuk tinta buram) agar tidak terpecah menjadi terang
          const deltaMax = 22.0;
          const upCap = work[i] < inkProtectThr ? 3.0 : 8.0;
          const clamped = clamp(nextV, work[i] - deltaMax, work[i] + upCap);
          est[i] = clamp(clamped, 0, 255);
        }
      }
      work = est;
    }

    if (amount <= 0.02) return work;

    const g1 = gaussianBlurSeparable(work, W, H, 1);
    const g2 = gaussianBlurSeparable(work, W, H, 2);
    const out = new Float32Array(N);

    for (let y = 0; y < H; y++) {
      const ym = Math.max(0, y - 1), yp = Math.min(H - 1, y + 1);
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const xm = Math.max(0, x - 1), xp = Math.min(W - 1, x + 1);
        const idx = row + x;
        const c = work[idx];

        // Local 3x3 bounds sebagai pembatas halo anti-ringing
        let locMin = c, locMax = c;
        for (let yy = ym; yy <= yp; yy++) {
          const rr = yy * W;
          for (let xx = xm; xx <= xp; xx++) {
            const v = work[rr + xx];
            if (v < locMin) locMin = v;
            if (v > locMax) locMax = v;
          }
        }

        const dFine = c - g1[idx];
        const dCoarse = c - g2[idx];
        const act = Math.max(Math.abs(dFine), Math.abs(dCoarse));
        // Noise coring: redam penajaman di area kertas mulus agar bintik JPEG tidak terangkat
        const gainMask = act <= noiseCoring ? 0 : clamp((act - noiseCoring) / (noiseCoring * 1.4), 0, 1);
        const rawDetail = 0.65 * dFine + 0.45 * dCoarse;
        // Asimetri pelindung tinta tipis: perkuat penggelapan inti tinta penuh (1.0x), redam pemucatan tepi (0.35x)
        const dirScale = rawDetail < 0 ? 1.0 : 0.35;
        let sharp = c + amount * gainMask * dirScale * rawDetail;

        // Pengaman total-variation nol-overshoot: jaga intensitas berada di [locMin, locMax] agar 0 halo/double-edge
        sharp = clamp(sharp, locMin, locMax);
        if (luma[idx] < inkProtectThr) {
          sharp = Math.min(sharp, luma[idx] + 2);
        }
        out[idx] = clamp(sharp, 0, 255);
      }
    }
    return out;
  }

  /**
   * Rotasi 90/180/270 murni pada buffer RGBA.
   */
  function rotateRgba(rgba, W, H, deg) {
    const d = ((Math.round(deg / 90) * 90) % 360 + 360) % 360;
    if (d === 0) return { data: new Uint8ClampedArray(rgba), width: W, height: H };
    const nW = (d === 90 || d === 270) ? H : W;
    const nH = (d === 90 || d === 270) ? W : H;
    const out = new Uint8ClampedArray(nW * nH * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const src = (y * W + x) * 4;
        let nx = x, ny = y;
        if (d === 90) { nx = H - 1 - y; ny = x; }
        else if (d === 180) { nx = W - 1 - x; ny = H - 1 - y; }
        else if (d === 270) { nx = y; ny = W - 1 - x; }
        const dst = (ny * nW + nx) * 4;
        out[dst] = rgba[src];
        out[dst + 1] = rgba[src + 1];
        out[dst + 2] = rgba[src + 2];
        out[dst + 3] = rgba[src + 3];
      }
    }
    return { data: out, width: nW, height: nH };
  }

  /**
   * Deskew sudut kecil (-7..+7 derajat) berbasis varians proyeksi horizontal untuk naskah miring.
   */
  function autoDeskewRgba(rgba, W, H) {
    const stepY = Math.max(1, Math.floor(H / 160));
    const stepX = Math.max(1, Math.floor(W / 240));
    const sw = Math.ceil(W / stepX), sh = Math.ceil(H / stepY);
    const small = new Float32Array(sw * sh);
    let meanLum = 0;
    for (let sy = 0; sy < sh; sy++) {
      const y = sy * stepY;
      for (let sx = 0; sx < sw; sx++) {
        const p = (y * W + sx * stepX) * 4;
        const v = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
        small[sy * sw + sx] = v;
        meanLum += v;
      }
    }
    meanLum /= Math.max(1, sw * sh);

    let bestAngle = 0, bestVar = -1;
    for (let ang = -6.0; ang <= 6.0; ang += 0.75) {
      const tanA = Math.tan((ang * Math.PI) / 180);
      const bins = new Float32Array(sh);
      const cx = sw * 0.5;
      for (let sy = 0; sy < sh; sy++) {
        for (let sx = 0; sx < sw; sx++) {
          const darkness = Math.max(0, meanLum - small[sy * sw + sx]);
          if (darkness < 12) continue;
          const ry = Math.round(sy + (sx - cx) * tanA * (stepX / stepY));
          if (ry >= 0 && ry < sh) bins[ry] += darkness;
        }
      }
      let s1 = 0, s2 = 0;
      for (let i = 0; i < sh; i++) { s1 += bins[i]; s2 += bins[i] * bins[i]; }
      const v = s2 / sh - (s1 / sh) ** 2;
      if (v > bestVar) { bestVar = v; bestAngle = ang; }
    }

    if (Math.abs(bestAngle) < 1.2) {
      return { data: rgba, width: W, height: H, deskewDeg: 0 };
    }

    const rad = (-bestAngle * Math.PI) / 180;
    const cosA = Math.cos(rad), sinA = Math.sin(rad);
    const cx = W * 0.5, cy = H * 0.5;
    const out = new Uint8ClampedArray(rgba.length);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const sx = Math.round(cosA * (x - cx) + sinA * (y - cy) + cx);
        const sy = Math.round(-sinA * (x - cx) + cosA * (y - cy) + cy);
        const cxp = clamp(sx, 0, W - 1), cyp = clamp(sy, 0, H - 1);
        const sIdx = (cyp * W + cxp) * 4;
        const dIdx = (y * W + x) * 4;
        out[dIdx] = rgba[sIdx];
        out[dIdx + 1] = rgba[sIdx + 1];
        out[dIdx + 2] = rgba[sIdx + 2];
        out[dIdx + 3] = 255;
      }
    }
    return { data: out, width: W, height: H, deskewDeg: Number(bestAngle.toFixed(2)) };
  }

  /**
   * Pipeline adaptif non-destruktif penuh: selalu bekerja pada salinan baru.
   *
   * @param {ImageData|{data:Uint8ClampedArray|Uint8Array,width:number,height:number}} imageData
   * @param {Object} [params]
   */
  function enhance(imageData, params = {}) {
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const rotDeg = params.rotateDeg || 0;
    let base = rotateRgba(imageData.data, imageData.width, imageData.height, rotDeg);
    let deskewDeg = 0;
    if (params.autoDeskew) {
      const ds = autoDeskewRgba(base.data, base.width, base.height);
      base = { data: ds.data, width: ds.width, height: ds.height };
      deskewDeg = ds.deskewDeg;
    }

    const W = base.width, H = base.height, N = W * H;
    const srcRgba = base.data;
    if (params.skipEnhance) {
      const blurBefore = estimateBlur({ data: srcRgba, width: W, height: H });
      const elapsedMs = Math.round((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0);
      return {
        imageData: { data: new Uint8ClampedArray(srcRgba), width: W, height: H },
        report: {
          mode: 'skipped',
          blurBefore,
          blurAfter: blurBefore,
          deskewDeg,
          rotateDeg: rotDeg,
          suppressedLinePixels: 0,
          usedRL: false,
          usedClahe: false,
          sharpAmount: 0,
          elapsedMs,
        },
      };
    }

    const blurBefore = estimateBlur({ data: srcRgba, width: W, height: H });
    const outRgba = new Uint8ClampedArray(srcRgba);

    // Polaritas awal: pastikan teks gelap di kertas cerah selama praproses
    if (blurBefore.isDarkMode) {
      for (let p = 0; p < outRgba.length; p += 4) {
        outRgba[p] = 255 - outRgba[p];
        outRgba[p + 1] = 255 - outRgba[p + 1];
        outRgba[p + 2] = 255 - outRgba[p + 2];
      }
    }

    const { luma, aCh, bCh } = toLumaAndLab(outRgba, N);
    const bgMaps = estimateBackgroundMaps(luma, aCh, bCh, W, H, 4);
    const { SW, SH, ds, bgL, bgA, bgB } = bgMaps;

    // 1. Cek apakah citra sudah tajam, terang merata, kontras baik, dan tanpa garis mengganggu
    const isCleanSharp = (
      blurBefore.level === 'tajam' &&
      blurBefore.shadowRange <= 32 &&
      blurBefore.bgMean >= 190 &&
      blurBefore.contrastSpan >= 115 &&
      blurBefore.lineClutterScore < 0.08 &&
      blurBefore.coloredInkRatio < 0.04 &&
      !blurBefore.isDarkMode &&
      !params.force
    );

    if (isCleanSharp && !(params.sharpen > 0) && !(params.contrast > 0) && params.removeLines !== true) {
      const elapsedMs = Math.round((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0);
      return {
        imageData: { data: outRgba, width: W, height: H },
        report: {
          mode: 'passthrough_sharp',
          blurBefore,
          blurAfter: blurBefore,
          deskewDeg,
          rotateDeg: rotDeg,
          suppressedLinePixels: 0,
          usedRL: false,
          usedClahe: false,
          sharpAmount: 0,
          elapsedMs,
        },
      };
    }

    // 2. Illumination Flattening + Pemisahan Tinta Warna CIELAB (menangkap pulpen biru/merah/hijau)
    const needShadowFlat = params.flattenShadow !== undefined
      ? Boolean(params.flattenShadow)
      : (bgMaps.shadowRange > 28 || bgMaps.minBg < 192 || blurBefore.isDarkMode || Boolean(params.force));
    const chromaBoost = params.chromaBoost !== undefined
      ? params.chromaBoost
      : (blurBefore.coloredInkRatio > 0.015 ? 0.55 : 0.25);

    const flatLuma = new Float32Array(N);
    for (let y = 0; y < H; y++) {
      const syf = Math.min(SH - 1.001, y / ds);
      const sy0 = Math.floor(syf), fy = syf - sy0, sy1 = Math.min(SH - 1, sy0 + 1);
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const sxf = Math.min(SW - 1.001, x / ds);
        const sx0 = Math.floor(sxf), fx = sxf - sx0, sx1 = Math.min(SW - 1, sx0 + 1);
        const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
        const idx00 = sy0 * SW + sx0, idx10 = sy0 * SW + sx1, idx01 = sy1 * SW + sx0, idx11 = sy1 * SW + sx1;
        const localBgL = bgL[idx00] * w00 + bgL[idx10] * w10 + bgL[idx01] * w01 + bgL[idx11] * w11;
        const localBgA = bgA[idx00] * w00 + bgA[idx10] * w10 + bgA[idx01] * w01 + bgA[idx11] * w11;
        const localBgB = bgB[idx00] * w00 + bgB[idx10] * w10 + bgB[idx01] * w01 + bgB[idx11] * w11;

        const idx = row + x;
        let v = needShadowFlat ? clamp((luma[idx] * 246.0) / Math.max(42.0, localBgL), 0, 255) : luma[idx];
        // Jarak kromatik oponen terhadap kertas lokal memperkuat tinta warna tanpa menggelapkan kertas bernoda
        const dC = Math.hypot(aCh[idx] - localBgA, bCh[idx] - localBgB);
        if (dC > 12.0 && chromaBoost > 0) {
          const pull = Math.min(95.0, (dC - 12.0) * chromaBoost * 1.45);
          v = Math.max(8.0, v - pull);
        }
        flatLuma[idx] = v;
      }
    }

    // 3. Penekan garis buku / retakan tembok non-destruktif (hanya jalankan vertikal bila diminta eksplisit)
    let workLuma = flatLuma;
    let suppressedLinePixels = 0;
    const shouldSuppressLines = params.removeLines === true || (params.removeLines !== false && blurBefore.lineClutterScore >= 0.08);
    if (shouldSuppressLines) {
      const sup = suppressLongStraightLines(
        flatLuma, bgL, SW, SH, ds, W, H,
        params.lineSuppressStrength ?? 0.78,
        Boolean(params.removeVertLines)
      );
      workLuma = sup.luma;
      suppressedLinePixels = sup.suppressedCount;
    }

    // 4. CLAHE adaptif hanya untuk kontras rendah / pencahayaan redup nyata (bukan latar kertas cerah seragam)
    let usedClahe = false;
    const isDimOrLowContrast = (blurBefore.bgMean < 185) || (blurBefore.contrastSpan < 95 && blurBefore.bgMean < 215);
    const autoClaheClip = params.claheClip !== undefined
      ? params.claheClip
      : (isDimOrLowContrast ? (blurBefore.level === 'berat' ? 1.8 : 1.45) : 0);
    if (autoClaheClip > 0.2) {
      const claheBlend = blurBefore.level === 'berat' ? 0.45 : 0.32;
      workLuma = applyClaheLuma(workLuma, W, H, params.claheGrid || 6, params.claheGrid || 6, autoClaheClip, params.claheBlend ?? claheBlend);
      usedClahe = true;
    }

    // 5. Penajaman Sadar-Blur + Richardson-Lucy opsional (dengan anti-ringing & penjaga kontinuitas)
    let autoSharp = 0;
    if (params.sharpen !== undefined) {
      autoSharp = clamp(params.sharpen, 0, 2.2);
    } else if (params.autoSharpen !== false) {
      if (blurBefore.level === 'berat') autoSharp = 0.72;
      else if (blurBefore.level === 'sedang') autoSharp = 0.28;
      else if (blurBefore.level === 'ringan') autoSharp = 0.16;
      else autoSharp = 0.0;
      // Redam penajaman isotropik saat blur sangat direksional (motion blur satu sumbu) agar tak muncul tepi ganda
      if (blurBefore.axisSymmetry !== undefined && blurBefore.axisSymmetry < 0.72) {
        autoSharp *= 0.45;
      }
    }
    const useRL = params.useRL !== undefined
      ? Boolean(params.useRL)
      : (blurBefore.level === 'berat' && (blurBefore.axisSymmetry === undefined || blurBefore.axisSymmetry >= 0.72));

    if (autoSharp > 0.02 || useRL) {
      const inkProtectThr = Math.max(115, blurBefore.bgMean - Math.max(24, blurBefore.contrastSpan * 0.22));
      const noiseCoring = Math.max(6.8, blurBefore.contrastSpan * 0.045);
      workLuma = applySharpenAndRL(workLuma, W, H, autoSharp, useRL, params.rlIterations || 3, {
        inkProtectThr,
        noiseCoring,
      });
    }

    // Opsional kontrol kontras pengguna tanpa kurva pow(1.32) yang memutus goresan tipis
    const userContrast = clamp(params.contrast || 0, -0.5, 1.0);
    if (Math.abs(userContrast) > 0.01) {
      const pivot = 165;
      const mul = 1.0 + userContrast;
      for (let i = 0; i < N; i++) {
        workLuma[i] = clamp(pivot + (workLuma[i] - pivot) * mul, 0, 255);
      }
    }

    // Rekonstruksi RGBA: bila flattening aktif gunakan luminans bersih agar bayangan warna ikut rata,
    // namun pertahankan rona ringan bila tanpa flattening
    for (let i = 0, p = 0; i < N; i++, p += 4) {
      if (needShadowFlat) {
        const v = clamp(Math.round(workLuma[i]), 0, 255);
        outRgba[p] = v;
        outRgba[p + 1] = v;
        outRgba[p + 2] = v;
      } else {
        const ratio = workLuma[i] / Math.max(8, luma[i]);
        outRgba[p] = clamp(Math.round(outRgba[p] * ratio), 0, 255);
        outRgba[p + 1] = clamp(Math.round(outRgba[p + 1] * ratio), 0, 255);
        outRgba[p + 2] = clamp(Math.round(outRgba[p + 2] * ratio), 0, 255);
      }
    }

    const blurAfter = estimateBlur({ data: outRgba, width: W, height: H });
    const elapsedMs = Math.round((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0);

    return {
      imageData: { data: outRgba, width: W, height: H },
      report: {
        mode: 'enhanced',
        blurBefore,
        blurAfter,
        deskewDeg,
        rotateDeg: rotDeg,
        suppressedLinePixels,
        usedRL: useRL,
        usedClahe,
        sharpAmount: Number(autoSharp.toFixed(2)),
        elapsedMs,
      },
    };
  }

  const api = {
    estimateBlur,
    enhance,
    rotateRgba,
    autoDeskewRgba,
  };

  global.AksaraEnhancer = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
