/**
 * Rekonstruktor Ortografi Spasial Aksara Jawa (AksaraScan v2.0)
 * Mengubah kumpulan bounding box deteksi atomik (Nglegena, Pasangan, Sandhangan)
 * menjadi deretan Unicode Aksara Jawa yang valid (U+A980..U+A9DF) dan transliterasi Latin.
 */

const JAVANESE_UNICODE = {
  // 20 Nglegena
  ha: '\uA9B2', na: '\uA9A4', ca: '\uA995', ra: '\uA9AB', ka: '\uA98F',
  da: '\uA9A2', ta: '\uA9A0', sa: '\uA9B1', wa: '\uA9AE', la: '\uA9AD',
  pa: '\uA9A5', dha: '\uA99D', ja: '\uA997', ya: '\uA9AA', nya: '\uA99A',
  ma: '\uA9A9', ga: '\uA992', ba: '\uA9A7', tha: '\uA99B', nga: '\uA994',
  // Sandhangan & Pangkon
  wulu: '\uA9B6', suku: '\uA9B8', pepet: '\uA9BC', taling: '\uA9BA', tarung: '\uA9B4',
  layar: '\uA982', cecak: '\uA981', wignyan: '\uA983', pangkon: '\uA9C0',
  cakra: '\uA9BF', keret: '\uA9BD', pengkal: '\uA9BE'
};

function boxMetrics(box) {
  const [x0, y0, x1, y1] = box;
  return {
    x0, y0, x1, y1,
    w: Math.max(1, x1 - x0),
    h: Math.max(1, y1 - y0),
    cx: (x0 + x1) / 2,
    cy: (y0 + y1) / 2
  };
}

function horizOverlap(a, b) {
  return Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0));
}

function vertOverlap(a, b) {
  return Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
}

function stripTrailingA(syl) {
  return syl.endsWith('a') ? syl.slice(0, -1) : syl;
}

/**
 * Susun ulang deteksi YOLO menjadi gugus suku kata & baris teks utuh.
 * @param {Array} detections - [{label, category, box:[x0,y0,x1,y1], confidence}, ...]
 * @returns {{unicodeText: string, latinText: string, clusters: Array}}
 */
function reconstructJavaneseText(detections) {
  if (!detections || !detections.length) {
    return { unicodeText: '—', latinText: '—', clusters: [] };
  }

  // Pisahkan Nglegena (jangkar utama) dan modifier (Pasangan & Sandhangan)
  const bases = [];
  const modifiers = [];

  detections.forEach((d, idx) => {
    const item = { ...d, m: boxMetrics(d.box), origIdx: idx };
    if (d.category === 'nglegena') {
      bases.push({ base: item, pasangan: null, sandhangans: [], box: [...d.box] });
    } else {
      modifiers.push(item);
    }
  });

  // Fallback jika model hanya mendeteksi sandhangan/pasangan tanpa base yang tertangkap
  if (!bases.length) {
    const sorted = [...modifiers].sort((a, b) => a.m.cy - b.m.cy || a.m.x0 - b.m.x0);
    return {
      unicodeText: sorted.map(s => JAVANESE_UNICODE[s.label] || s.unicode || '').join(' '),
      latinText: sorted.map(s => s.label).join(' '),
      clusters: sorted.map(s => ({
        label: s.label,
        unicode: s.unicode || JAVANESE_UNICODE[s.label] || '',
        confidence: s.confidence,
        box: s.box
      }))
    };
  }

  // Kelompokkan base ke dalam baris teks
  bases.sort((a, b) => a.base.m.cy - b.base.m.cy);
  const avgBaseH = bases.reduce((s, b) => s + b.base.m.h, 0) / bases.length;
  const lines = [];

  for (const cl of bases) {
    let placed = false;
    for (const line of lines) {
      const avgY = line.reduce((s, it) => s + it.base.m.cy, 0) / line.length;
      if (Math.abs(cl.base.m.cy - avgY) < Math.max(24, avgBaseH * 0.9)) {
        line.push(cl);
        placed = true;
        break;
      }
    }
    if (!placed) lines.push([cl]);
  }

  // Urutkan baris atas->bawah dan huruf kiri->kanan
  lines.sort((l1, l2) => {
    const y1 = l1.reduce((s, it) => s + it.base.m.cy, 0) / l1.length;
    const y2 = l2.reduce((s, it) => s + it.base.m.cy, 0) / l2.length;
    return y1 - y2;
  });
  lines.forEach(line => line.sort((a, b) => a.base.m.x0 - b.base.m.x0));

  const flatClusters = lines.flat();

  // Pasangkan setiap Pasangan terlebih dahulu ke base terdekat (di bawah atau sebelah kanan)
  const remainingMods = [];
  for (const mod of modifiers) {
    if (mod.category === 'pasangan') {
      let bestHost = null;
      let bestScore = Infinity;
      const isRightPasangan = (mod.label === 'p_ha' || mod.label === 'p_sa' || mod.label === 'p_pa');

      for (const cl of flatClusters) {
        const bm = cl.base.m;
        const dx = mod.m.cx - bm.cx;
        const dy = mod.m.cy - bm.cy;

        if (isRightPasangan) {
          // Berada di sebelah kanan huruf induk dengan ketinggian sejajar
          if (mod.m.x0 >= bm.x0 && mod.m.x0 - bm.x1 < bm.w * 1.4 && Math.abs(dy) < bm.h * 0.9) {
            const dist = Math.hypot(mod.m.x0 - bm.x1, dy);
            if (dist < bestScore) { bestScore = dist; bestHost = cl; }
          }
        } else {
          // Berada di bawah huruf induk
          const hOv = horizOverlap(mod.m, bm);
          if (dy > 0 && dy < bm.h * 2.6 && (hOv > 0 || Math.abs(dx) < bm.w * 0.85)) {
            const dist = Math.hypot(dx * 1.5, dy);
            if (dist < bestScore) { bestScore = dist; bestHost = cl; }
          }
        }
      }
      if (bestHost && !bestHost.pasangan) {
        bestHost.pasangan = mod;
      } else {
        remainingMods.push(mod);
      }
    } else {
      remainingMods.push(mod);
    }
  }

  // Pasangkan seluruh Sandhangan (wulu, suku, pepet, taling, tarung, layar, cecak, wignyan, pangkon, cakra, keret, pengkal)
  for (const mod of remainingMods) {
    if (mod.category !== 'sandhangan') continue;
    let bestHost = null;
    let bestScore = Infinity;

    for (const cl of flatClusters) {
      // Gunakan gabungan area base (+ pasangan jika ada) sebagai referensi jangkar
      const bm = cl.base.m;
      const fullAnchor = cl.pasangan
        ? boxMetrics([
            Math.min(bm.x0, cl.pasangan.m.x0),
            Math.min(bm.y0, cl.pasangan.m.y0),
            Math.max(bm.x1, cl.pasangan.m.x1),
            Math.max(bm.y1, cl.pasangan.m.y1)
          ])
        : bm;

      const label = mod.label;
      if (label === 'taling') {
        // Taling selalu di KIRI huruf induk
        const gapLeft = bm.x0 - mod.m.x1;
        if (mod.m.cx < bm.cx && gapLeft < bm.w * 1.3 && gapLeft > -bm.w * 0.5 && vertOverlap(mod.m, fullAnchor) > 0) {
          const score = Math.abs(gapLeft) + Math.abs(mod.m.cy - bm.cy) * 0.4;
          if (score < bestScore) { bestScore = score; bestHost = cl; }
        }
      } else if (label === 'tarung' || label === 'wignyan' || label === 'pangkon' || label === 'pengkal') {
        // Di KANAN huruf induk/pasangan
        const gapRight = mod.m.x0 - fullAnchor.x1;
        if (mod.m.cx > fullAnchor.x0 && gapRight < bm.w * 1.4 && vertOverlap(mod.m, fullAnchor) >= -bm.h * 0.3) {
          const score = Math.max(0, gapRight) + Math.abs(mod.m.cy - fullAnchor.cy) * 0.5;
          if (score < bestScore) { bestScore = score; bestHost = cl; }
        }
      } else if (label === 'wulu' || label === 'pepet' || label === 'layar' || label === 'cecak') {
        // Di ATAS huruf induk
        const dy = bm.cy - mod.m.cy;
        const hOv = horizOverlap(mod.m, bm);
        if (dy > 0 && dy < bm.h * 2.5 && (hOv > 0 || Math.abs(mod.m.cx - bm.cx) < bm.w * 0.9)) {
          const score = Math.hypot(mod.m.cx - bm.cx, dy * 0.7);
          if (score < bestScore) { bestScore = score; bestHost = cl; }
        }
      } else {
        // suku, cakra, keret -> di BAWAH / melingkari
        const dy = mod.m.cy - bm.y0;
        const hOv = horizOverlap(mod.m, fullAnchor);
        if (dy > 0 && dy < fullAnchor.h * 2.4 && (hOv > 0 || Math.abs(mod.m.cx - fullAnchor.cx) < fullAnchor.w * 1.1)) {
          const score = Math.hypot(mod.m.cx - fullAnchor.cx, Math.abs(mod.m.y0 - fullAnchor.y1) * 0.8);
          if (score < bestScore) { bestScore = score; bestHost = cl; }
        }
      }
    }

    if (bestHost) {
      bestHost.sandhangans.push(mod);
    }
  }

  // Konversi setiap gugus menjadi Unicode & Latin yang akurat
  const lineStringsUni = [];
  const lineStringsLat = [];
  const outClusters = [];

  for (const line of lines) {
    const uniWords = [];
    const latWords = [];

    for (const cl of line) {
      const baseName = cl.base.label;
      let uni = JAVANESE_UNICODE[baseName] || '';
      let lat = baseName;

      // Gabung bounding box menyeluruh untuk gugus
      let [ux0, uy0, ux1, uy1] = cl.base.box;
      let confSum = cl.base.confidence;
      let countParts = 1;

      if (cl.pasangan) {
        const pRoot = cl.pasangan.label.replace(/^p_/, '');
        uni += '\uA9C0' + (JAVANESE_UNICODE[pRoot] || '');
        lat = stripTrailingA(baseName) + pRoot;
        ux0 = Math.min(ux0, cl.pasangan.box[0]);
        uy0 = Math.min(uy0, cl.pasangan.box[1]);
        ux1 = Math.max(ux1, cl.pasangan.box[2]);
        uy1 = Math.max(uy1, cl.pasangan.box[3]);
        confSum += cl.pasangan.confidence;
        countParts++;
      }

      const modNames = new Set(cl.sandhangans.map(s => s.label));
      for (const s of cl.sandhangans) {
        ux0 = Math.min(ux0, s.box[0]);
        uy0 = Math.min(uy0, s.box[1]);
        ux1 = Math.max(ux1, s.box[2]);
        uy1 = Math.max(uy1, s.box[3]);
        confSum += s.confidence;
        countParts++;
      }

      // Aturan urutan fonetik Jawa:
      // 1. Wyanjana (-r-, -re-, -y-)
      if (modNames.has('cakra')) {
        uni += JAVANESE_UNICODE.cakra;
        lat = stripTrailingA(lat) + 'ra';
      } else if (modNames.has('keret')) {
        uni += JAVANESE_UNICODE.keret;
        lat = stripTrailingA(lat) + 're';
      } else if (modNames.has('pengkal')) {
        uni += JAVANESE_UNICODE.pengkal;
        lat = stripTrailingA(lat) + 'ya';
      }

      // 2. Swara (vokal) & Pangkon (pemati vokal)
      if (modNames.has('pangkon')) {
        uni += JAVANESE_UNICODE.pangkon;
        lat = stripTrailingA(lat);
      } else if (modNames.has('taling') && modNames.has('tarung')) {
        uni += JAVANESE_UNICODE.taling + JAVANESE_UNICODE.tarung;
        lat = stripTrailingA(lat) + 'o';
      } else if (modNames.has('taling')) {
        uni += JAVANESE_UNICODE.taling;
        lat = stripTrailingA(lat) + 'è';
      } else if (modNames.has('wulu')) {
        uni += JAVANESE_UNICODE.wulu;
        lat = stripTrailingA(lat) + 'i';
      } else if (modNames.has('suku')) {
        uni += JAVANESE_UNICODE.suku;
        lat = stripTrailingA(lat) + 'u';
      } else if (modNames.has('pepet')) {
        uni += JAVANESE_UNICODE.pepet;
        lat = stripTrailingA(lat) + 'ê';
      }

      // 3. Panyigeg Wanda (-r, -ng, -h)
      if (modNames.has('layar')) {
        uni += JAVANESE_UNICODE.layar;
        lat += 'r';
      }
      if (modNames.has('cecak')) {
        uni += JAVANESE_UNICODE.cecak;
        lat += 'ng';
      }
      if (modNames.has('wignyan')) {
        uni += JAVANESE_UNICODE.wignyan;
        lat += 'h';
      }

      uniWords.push(uni);
      latWords.push(lat);

      outClusters.push({
        label: lat,
        unicode: uni,
        baseLabel: baseName,
        pasanganLabel: cl.pasangan ? cl.pasangan.label : null,
        sandhangans: [...modNames],
        confidence: confSum / countParts,
        isUncertain: (confSum / countParts) < 0.40,
        box: [ux0, uy0, ux1, uy1]
      });
    }

    lineStringsUni.push(uniWords.join(' '));
    lineStringsLat.push(latWords.join(' '));
  }

  return {
    unicodeText: lineStringsUni.join('\n'),
    latinText: lineStringsLat.join('\n'),
    clusters: outClusters
  };
}

window.AksaraOrthography = {
  reconstructJavaneseText,
  JAVANESE_UNICODE
};
