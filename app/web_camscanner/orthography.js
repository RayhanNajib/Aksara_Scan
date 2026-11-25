/**
 * Rekonstruktor Ortografi Spasial Aksara Jawa (AksaraScan v2.5)
 * Mengubah kumpulan bounding box deteksi atomik (Nglegena, Pasangan, Sandhangan, Pada)
 * menjadi deretan Unicode Aksara Jawa (U+A980..U+A9DF), suku kata Latin,
 * kalimat Latin tergabung (DP word segmentation), dan terjemahan Bahasa Indonesia.
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
  cakra: '\uA9BF', keret: '\uA9BD', pengkal: '\uA9BE',
  // Tanda Baca Pada
  pada_adeg: '\uA9CB', pada_lingsa: '\uA9C8', pada_lungsi: '\uA9C9'
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
 * Susun ulang deteksi YOLO menjadi gugus suku kata, tanda baca, kalimat Latin, & terjemahan ID.
 */
function reconstructJavaneseText(detections) {
  if (!detections || !detections.length) {
    return { unicodeText: '—', latinText: '—', sentenceLatin: '—', sentenceIndonesian: '—', clusters: [] };
  }

  const anchors = [];
  const modifiers = [];

  detections.forEach((d, idx) => {
    const item = { ...d, m: boxMetrics(d.box), origIdx: idx };
    if (d.category === 'nglegena' || d.category === 'pada') {
      anchors.push({
        isPada: d.category === 'pada',
        base: item,
        pasangan: null,
        sandhangans: [],
        box: [...d.box]
      });
    } else {
      modifiers.push(item);
    }
  });

  if (!anchors.length) {
    const sorted = [...modifiers].sort((a, b) => a.m.cy - b.m.cy || a.m.x0 - b.m.x0);
    return {
      unicodeText: sorted.map(s => JAVANESE_UNICODE[s.label] || s.unicode || '').join(' '),
      latinText: sorted.map(s => s.label).join(' '),
      sentenceLatin: sorted.map(s => s.label).join(' '),
      sentenceIndonesian: '—',
      clusters: sorted.map(s => ({
        label: s.label,
        unicode: s.unicode || JAVANESE_UNICODE[s.label] || '',
        confidence: s.confidence,
        box: s.box
      }))
    };
  }

  // Kelompokkan jangkar (nglegena & pada) ke baris teks
  const baseOnly = anchors.filter(a => !a.isPada);
  const refHList = baseOnly.length ? baseOnly : anchors;
  const avgBaseH = refHList.reduce((s, b) => s + b.base.m.h, 0) / refHList.length;

  anchors.sort((a, b) => a.base.m.cy - b.base.m.cy);
  const lines = [];

  for (const cl of anchors) {
    let placed = false;
    for (const line of lines) {
      const refItems = line.filter(it => !it.isPada);
      const pool = refItems.length ? refItems : line;
      const avgY = pool.reduce((s, it) => s + it.base.m.cy, 0) / pool.length;
      if (Math.abs(cl.base.m.cy - avgY) < Math.max(24, avgBaseH * 1.05)) {
        line.push(cl);
        placed = true;
        break;
      }
    }
    if (!placed) lines.push([cl]);
  }

  lines.sort((l1, l2) => {
    const y1 = l1.reduce((s, it) => s + it.base.m.cy, 0) / l1.length;
    const y2 = l2.reduce((s, it) => s + it.base.m.cy, 0) / l2.length;
    return y1 - y2;
  });
  lines.forEach(line => line.sort((a, b) => a.base.m.x0 - b.base.m.x0));

  const syllableClusters = lines.flat().filter(c => !c.isPada);

  // Pasangkan setiap Pasangan terlebih dahulu ke nglegena terdekat
  const remainingMods = [];
  for (const mod of modifiers) {
    if (mod.category === 'pasangan') {
      let bestHost = null;
      let bestScore = Infinity;
      const isRightPasangan = (mod.label === 'p_ha' || mod.label === 'p_sa' || mod.label === 'p_pa');

      for (const cl of syllableClusters) {
        const bm = cl.base.m;
        const dx = mod.m.cx - bm.cx;
        const dy = mod.m.cy - bm.cy;

        if (isRightPasangan) {
          if (mod.m.x0 >= bm.x0 && mod.m.x0 - bm.x1 < bm.w * 1.4 && Math.abs(dy) < bm.h * 0.9) {
            const dist = Math.hypot(mod.m.x0 - bm.x1, dy);
            if (dist < bestScore) { bestScore = dist; bestHost = cl; }
          }
        } else {
          const hOv = horizOverlap(mod.m, bm);
          if (dy > 0 && dy < bm.h * 2.8 && (hOv > 0 || Math.abs(dx) < bm.w * 0.9)) {
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

  // Pasangkan seluruh Sandhangan
  for (const mod of remainingMods) {
    if (mod.category !== 'sandhangan') continue;
    let bestHost = null;
    let bestScore = Infinity;

    for (const cl of syllableClusters) {
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
        const gapLeft = bm.x0 - mod.m.x1;
        if (mod.m.cx < bm.cx && gapLeft < bm.w * 1.3 && gapLeft > -bm.w * 0.5 && vertOverlap(mod.m, fullAnchor) > 0) {
          const score = Math.abs(gapLeft) + Math.abs(mod.m.cy - bm.cy) * 0.4;
          if (score < bestScore) { bestScore = score; bestHost = cl; }
        }
      } else if (label === 'tarung' || label === 'wignyan' || label === 'pangkon' || label === 'pengkal') {
        const gapRight = mod.m.x0 - fullAnchor.x1;
        if (mod.m.cx > fullAnchor.x0 && gapRight < bm.w * 1.4 && vertOverlap(mod.m, fullAnchor) >= -bm.h * 0.35) {
          const score = Math.max(0, gapRight) + Math.abs(mod.m.cy - fullAnchor.cy) * 0.5;
          if (score < bestScore) { bestScore = score; bestHost = cl; }
        }
      } else if (label === 'wulu' || label === 'pepet' || label === 'layar' || label === 'cecak') {
        const dy = bm.cy - mod.m.cy;
        const hOv = horizOverlap(mod.m, bm);
        if (dy > 0 && dy < bm.h * 2.6 && (hOv > 0 || Math.abs(mod.m.cx - bm.cx) < bm.w * 0.95)) {
          const score = Math.hypot(mod.m.cx - bm.cx, dy * 0.7);
          if (score < bestScore) { bestScore = score; bestHost = cl; }
        }
      } else {
        // suku, cakra, keret
        const dy = mod.m.cy - bm.y0;
        const hOv = horizOverlap(mod.m, fullAnchor);
        if (dy > 0 && dy < fullAnchor.h * 2.8 && (hOv > 0 || Math.abs(mod.m.cx - fullAnchor.cx) < fullAnchor.w * 1.15)) {
          const score = Math.hypot(mod.m.cx - fullAnchor.cx, Math.abs(mod.m.y0 - fullAnchor.y1) * 0.8);
          if (score < bestScore) { bestScore = score; bestHost = cl; }
        }
      }
    }

    if (bestHost) {
      bestHost.sandhangans.push(mod);
    }
  }

  const lineStringsUni = [];
  const lineStringsLat = [];
  const outClusters = [];
  const phraseTokens = [];

  for (const line of lines) {
    const uniWords = [];
    const latWords = [];

    for (const cl of line) {
      if (cl.isPada) {
        const pName = cl.base.label;
        const uniP = JAVANESE_UNICODE[pName] || '';
        const punctChar = pName === 'pada_lingsa' ? ',' : (pName === 'pada_lungsi' ? '.' : '');
        uniWords.push(uniP);
        if (punctChar) latWords.push(punctChar);
        phraseTokens.push({ isPunct: true, punctChar });
        outClusters.push({
          label: pName,
          unicode: uniP,
          confidence: cl.base.confidence,
          isUncertain: cl.base.confidence < 0.40,
          box: cl.base.box
        });
        continue;
      }

      const baseName = cl.base.label;
      let uni = JAVANESE_UNICODE[baseName] || '';
      let lat = baseName;

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

      // Wyanjana
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

      // Swara / Pangkon
      if (modNames.has('pangkon')) {
        uni += JAVANESE_UNICODE.pangkon;
        lat = stripTrailingA(lat);
      } else if (modNames.has('taling') && modNames.has('tarung')) {
        uni += JAVANESE_UNICODE.taling + JAVANESE_UNICODE.tarung;
        lat = stripTrailingA(lat) + 'o';
      } else if (modNames.has('taling')) {
        uni += JAVANESE_UNICODE.taling;
        lat = stripTrailingA(lat) + 'é';
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

      // Panyigeg Wanda
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
      phraseTokens.push({ isPunct: false, latin: lat });

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

    lineStringsUni.push(uniWords.join(''));
    lineStringsLat.push(latWords.join(' '));
  }

  let sentenceLatin = lineStringsLat.join(' ');
  let sentenceIndonesian = '—';
  if (window.AksaraLexicon) {
    const tr = window.AksaraLexicon.translateJavaneseTokens(phraseTokens);
    sentenceLatin = tr.sentenceLatin;
    sentenceIndonesian = tr.sentenceIndonesian;
  }

  return {
    unicodeText: lineStringsUni.join('\n'),
    latinText: lineStringsLat.join('\n'),
    sentenceLatin,
    sentenceIndonesian,
    clusters: outClusters
  };
}

window.AksaraOrthography = {
  reconstructJavaneseText,
  JAVANESE_UNICODE
};
