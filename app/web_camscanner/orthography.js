/**
 * Rekonstruktor Ortografi Spasial Aksara Jawa, Validator Blok Unicode (U+A980..U+A9DF),
 * dan Dekoder Beam Konservatif v3.0 — AksaraScan (Fase 4b-4d).
 *
 * Mempertahankan geometri pasangan/sandhangan teruji v2.5 (termasuk `isRightPasangan`
 * untuk p_ha/p_sa/p_pa & kotak gabungan `fullAnchor`), ditambah validator kanonik
 * U+A980..U+A9DF dan Dekoder Beam terkendali yang HANYA memperbaiki kata OOV menuju
 * kata kamus/korpus Jawa sah tanpa menghalusinasi `cecak` atau afiks palsu.
 */
(function (global) {
  'use strict';

  const U = {
    CECAK: '\uA981',
    LAYAR: '\uA982',
    WIGNYAN: '\uA983',
    CEC_TELU: '\uA9B3',
    TARUNG: '\uA9B4',
    WULU: '\uA9B6',
    SUKU: '\uA9B8',
    TALING: '\uA9BA',
    PEPET: '\uA9BC',
    KERET: '\uA9BD',
    PENGKAL: '\uA9BE',
    CAKRA: '\uA9BF',
    PANGKON: '\uA9C0',
    PADA_LINGSA: '\uA9C8',
    PADA_LUNGSI: '\uA9C9',
    PADA_ADEG: '\uA9CB',
  };

  const JAVANESE_UNICODE = {
    ha: '\uA9B2', na: '\uA9A4', ca: '\uA995', ra: '\uA9AB', ka: '\uA98F',
    da: '\uA9A2', ta: '\uA9A0', sa: '\uA9B1', wa: '\uA9AE', la: '\uA9AD',
    pa: '\uA9A5', dha: '\uA99D', ja: '\uA997', ya: '\uA9AA', nya: '\uA99A',
    ma: '\uA9A9', ga: '\uA992', ba: '\uA9A7', tha: '\uA99B', nga: '\uA994',
    p_ha: '\uA9C0\uA9B2', p_na: '\uA9C0\uA9A4', p_ca: '\uA9C0\uA995', p_ra: '\uA9C0\uA9AB', p_ka: '\uA9C0\uA98F',
    p_da: '\uA9C0\uA9A2', p_ta: '\uA9C0\uA9A0', p_sa: '\uA9C0\uA9B1', p_wa: '\uA9C0\uA9AE', p_la: '\uA9C0\uA9AD',
    p_pa: '\uA9C0\uA9A5', p_dha: '\uA9C0\uA99D', p_ja: '\uA9C0\uA997', p_ya: '\uA9C0\uA9AA', p_nya: '\uA9C0\uA99A',
    p_ma: '\uA9C0\uA9A9', p_ga: '\uA9C0\uA992', p_ba: '\uA9C0\uA9A7', p_tha: '\uA9C0\uA99B', p_nga: '\uA9C0\uA994',
    wulu: U.WULU, suku: U.SUKU, pepet: U.PEPET, taling: U.TALING, tarung: U.TARUNG,
    layar: U.LAYAR, cecak: U.CECAK, wignyan: U.WIGNYAN, pangkon: U.PANGKON,
    cakra: U.CAKRA, keret: U.KERET, pengkal: U.PENGKAL,
    pada_adeg: U.PADA_ADEG, pada_lingsa: U.PADA_LINGSA, pada_lungsi: U.PADA_LUNGSI,
  };

  const BASE_CHARS = {
    '\uA9B2': 'ha', '\uA9A4': 'na', '\uA995': 'ca', '\uA9AB': 'ra', '\uA98F': 'ka',
    '\uA9A2': 'da', '\uA9A0': 'ta', '\uA9B1': 'sa', '\uA9AE': 'wa', '\uA9AD': 'la',
    '\uA9A5': 'pa', '\uA99D': 'dha', '\uA997': 'ja', '\uA9AA': 'ya', '\uA99A': 'nya',
    '\uA9A9': 'ma', '\uA992': 'ga', '\uA9A7': 'ba', '\uA99B': 'tha', '\uA994': 'nga',
    '\uA991': 'ka', '\uA993': 'ga', '\uA996': 'ca', '\uA998': 'ja',
    '\uA99F': 'da', '\uA9A1': 'ta', '\uA9A3': 'dha', '\uA9A6': 'pa', '\uA9A8': 'ba', '\uA9AF': 'sa',
  };
  const CHAR_TO_CONS = {};
  Object.keys(BASE_CHARS).forEach((ch) => {
    const nm = BASE_CHARS[ch];
    CHAR_TO_CONS[ch] = nm === 'ha' ? 'h' : nm.slice(0, -1);
  });

  const PANYIGEG_SET = new Set([U.CECAK, U.LAYAR, U.WIGNYAN]);
  const SWARA_SET = new Set([U.WULU, U.SUKU, U.PEPET, U.TALING, U.TARUNG]);
  const WYANJANA_SET = new Set([U.CAKRA, U.KERET, U.PENGKAL]);
  const PADA_SET = new Set([U.PADA_ADEG, U.PADA_LINGSA, U.PADA_LUNGSI, '\uA9C7', '\uA9CA', '\uA9CC', '\uA9CF']);
  const PA_CEREK = '\uA989';
  const NGA_LELET = '\uA98A';
  const SWARA_INDEP = new Set(['\uA984', '\uA986', '\uA988', '\uA98C', '\uA98E']);

  // Hanya pasangan konfusi detektor terukur dari evaluasi Fase 0 (pepet/wulu/cecak/taling)
  const CONFUSABLE_MODS = {
    pepet: ['taling', 'cecak', 'wulu'],
    taling: ['pepet'],
    wulu: ['cecak', 'pepet'],
    cecak: ['pepet', 'wulu'],
  };

  function boxMetrics(box) {
    const [x0, y0, x1, y1] = box;
    return {
      x0, y0, x1, y1,
      w: Math.max(1, x1 - x0),
      h: Math.max(1, y1 - y0),
      cx: (x0 + x1) / 2,
      cy: (y0 + y1) / 2,
    };
  }

  function horizOverlap(a, b) {
    return Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0));
  }

  function vertOverlap(a, b) {
    return Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
  }

  /**
   * Validator kanonik blok Unicode Jawa U+A980..U+A9DF (port 1:1 tools/synth_v2/orthography_rules.py).
   */
  function validateJavaneseUnicode(text) {
    if (!text || !String(text).trim()) {
      return { valid: false, errors: ['teks kosong'], clusters: [] };
    }
    const s = String(text).replace(/\s+/g, '');
    const errors = [];
    const clusters = [];
    let i = 0;
    const n = s.length;

    while (i < n) {
      const ch = s[i];
      if (PADA_SET.has(ch)) {
        clusters.push(ch);
        i += 1;
        continue;
      }
      if (SWARA_INDEP.has(ch) || (ch >= '\uA9D0' && ch <= '\uA9D9')) {
        clusters.push(ch);
        i += 1;
        continue;
      }
      if (ch === PA_CEREK || ch === NGA_LELET) {
        let cl = ch;
        i += 1;
        if (i < n && SWARA_SET.has(s[i])) {
          errors.push(`Pa cerek/Nga lelet tidak boleh memakai sandhangan vokal di indeks ${i}`);
        }
        while (i < n && PANYIGEG_SET.has(s[i])) {
          cl += s[i];
          i += 1;
        }
        clusters.push(cl);
        continue;
      }
      if (!BASE_CHARS[ch]) {
        errors.push(`Karakter tak dikenal / posisi yatim U+${ch.charCodeAt(0).toString(16).toUpperCase()} di indeks ${i}`);
        i += 1;
        continue;
      }

      let cl = ch;
      i += 1;
      if (i < n && s[i] === U.CEC_TELU) {
        cl += s[i];
        i += 1;
      }

      let pasanganCount = 0;
      let endedWithPangkon = false;
      while (i < n && s[i] === U.PANGKON) {
        cl += U.PANGKON;
        i += 1;
        if (i < n && BASE_CHARS[s[i]]) {
          pasanganCount += 1;
          if (pasanganCount > 2) errors.push(`Tumpukan pasangan melebihi 2 tingkat di indeks ${i}`);
          cl += s[i];
          i += 1;
          if (i < n && s[i] === U.CEC_TELU) {
            cl += s[i];
            i += 1;
          }
        } else {
          endedWithPangkon = true;
          break;
        }
      }

      if (endedWithPangkon) {
        if (i < n && (WYANJANA_SET.has(s[i]) || SWARA_SET.has(s[i]) || PANYIGEG_SET.has(s[i]))) {
          errors.push(`Kombinasi terlarang setelah pangkon akhir di indeks ${i}`);
        }
        clusters.push(cl);
        continue;
      }

      let wyanjanaSeen = null;
      if (i < n && WYANJANA_SET.has(s[i])) {
        wyanjanaSeen = s[i];
        cl += wyanjanaSeen;
        i += 1;
        if (i < n && WYANJANA_SET.has(s[i])) {
          errors.push(`Duplikasi sandhangan wyanjana di indeks ${i}`);
        }
      }

      const swaraSeen = [];
      while (i < n && SWARA_SET.has(s[i])) {
        swaraSeen.push(s[i]);
        cl += s[i];
        i += 1;
      }
      if (swaraSeen.length > 0) {
        if (swaraSeen.length > 2 || (swaraSeen.length === 2 && (swaraSeen[0] !== U.TALING || swaraSeen[1] !== U.TARUNG))) {
          errors.push(`Kombinasi sandhangan swara tidak sah pada kluster ${cl}`);
        }
        if (wyanjanaSeen === U.KERET) {
          errors.push('Keret tidak boleh digabung sandhangan swara');
        }
        if (wyanjanaSeen === U.CAKRA && swaraSeen.includes(U.PEPET)) {
          errors.push('Cakra + Pepet tidak sah; gunakan Keret');
        }
      }

      const panyigegSeen = [];
      while (i < n && PANYIGEG_SET.has(s[i])) {
        panyigegSeen.push(s[i]);
        cl += s[i];
        i += 1;
      }
      if (panyigegSeen.length > 1) {
        errors.push(`Lebih dari satu panyigeg wanda pada kluster ${cl}`);
      }
      if (i < n && (SWARA_SET.has(s[i]) || WYANJANA_SET.has(s[i]) || s[i] === U.PANGKON)) {
        errors.push(`Urutan Unicode terbalik setelah panyigeg di indeks ${i}`);
      }

      clusters.push(cl);
    }

    return { valid: errors.length === 0, errors, clusters };
  }

  /**
   * Membangun rangkaian string Unicode kanonik untuk satu kluster basis + pasangan + modifier.
   */
  function buildClusterUnicode(baseLabel, pasanganLabels, modLabels) {
    if (baseLabel.startsWith('pada_')) {
      return JAVANESE_UNICODE[baseLabel] || '';
    }
    const baseCh = JAVANESE_UNICODE[baseLabel] || '';
    if (!baseCh) return '';

    let out = baseCh;
    const mods = new Set(modLabels || []);

    for (const pLbl of (pasanganLabels || [])) {
      if (pLbl && pLbl.startsWith('p_')) {
        const ch = JAVANESE_UNICODE[pLbl.slice(2)];
        if (ch) out += U.PANGKON + ch;
      }
    }

    if (mods.has('pangkon')) {
      return out + U.PANGKON;
    }

    if (mods.has('cakra')) out += U.CAKRA;
    else if (mods.has('keret')) out += U.KERET;
    else if (mods.has('pengkal')) out += U.PENGKAL;

    if (mods.has('taling') && mods.has('tarung')) out += U.TALING + U.TARUNG;
    else if (mods.has('taling')) out += U.TALING;
    else if (mods.has('wulu')) out += U.WULU;
    else if (mods.has('suku')) out += U.SUKU;
    else if (mods.has('pepet')) out += U.PEPET;

    if (mods.has('cecak')) out += U.CECAK;
    else if (mods.has('layar')) out += U.LAYAR;
    else if (mods.has('wignyan')) out += U.WIGNYAN;

    return out;
  }

  function clusterUnicodeToLatin(clusterStr, isClauseStart) {
    if (!clusterStr) return '';
    if (clusterStr === U.PADA_ADEG) return '';
    if (clusterStr === U.PADA_LINGSA) return ', ';
    if (clusterStr === U.PADA_LUNGSI) return '. ';
    if (PADA_SET.has(clusterStr)) return ' ';

    let idx = 0;
    const n = clusterStr.length;
    const ch0 = clusterStr[idx];
    if (ch0 === PA_CEREK || ch0 === NGA_LELET) {
      let out = ch0 === PA_CEREK ? 'rê' : 'lê';
      idx += 1;
      while (idx < n && PANYIGEG_SET.has(clusterStr[idx])) {
        out += clusterStr[idx] === U.CECAK ? 'ng' : clusterStr[idx] === U.LAYAR ? 'r' : 'h';
        idx += 1;
      }
      return out;
    }

    const baseCons = CHAR_TO_CONS[ch0];
    if (!baseCons) return '';
    idx += 1;
    if (idx < n && clusterStr[idx] === U.CEC_TELU) idx += 1;

    let onset = baseCons;
    let killed = false;
    while (idx < n && clusterStr[idx] === U.PANGKON) {
      idx += 1;
      if (idx < n && CHAR_TO_CONS[clusterStr[idx]]) {
        onset += CHAR_TO_CONS[clusterStr[idx]];
        idx += 1;
        if (idx < n && clusterStr[idx] === U.CEC_TELU) idx += 1;
      } else {
        killed = true;
        break;
      }
    }
    if (killed) return onset;

    let med = '';
    if (idx < n && WYANJANA_SET.has(clusterStr[idx])) {
      const wy = clusterStr[idx];
      idx += 1;
      if (wy === U.CAKRA) med = 'r';
      else if (wy === U.KERET) {
        let tail = '';
        while (idx < n && PANYIGEG_SET.has(clusterStr[idx])) {
          tail += clusterStr[idx] === U.CECAK ? 'ng' : clusterStr[idx] === U.LAYAR ? 'r' : 'h';
          idx += 1;
        }
        return onset + 'rê' + tail;
      } else if (wy === U.PENGKAL) med = 'y';
    }

    const sw = [];
    while (idx < n && SWARA_SET.has(clusterStr[idx])) {
      sw.push(clusterStr[idx]);
      idx += 1;
    }

    let vowel = 'a';
    if (sw.length === 2 && sw[0] === U.TALING && sw[1] === U.TARUNG) vowel = 'o';
    else if (sw.length === 1) {
      const s0 = sw[0];
      if (s0 === U.WULU) vowel = 'i';
      else if (s0 === U.SUKU) vowel = 'u';
      else if (s0 === U.PEPET) vowel = 'ê';
      else if (s0 === U.TALING) vowel = 'é';
      else if (s0 === U.TARUNG) vowel = 'o';
    }

    let coda = '';
    while (idx < n && PANYIGEG_SET.has(clusterStr[idx])) {
      const py = clusterStr[idx];
      coda += py === U.CECAK ? 'ng' : py === U.LAYAR ? 'r' : 'h';
      idx += 1;
    }

    let fullOnset = onset + med;
    if (fullOnset === 'h' && isClauseStart) fullOnset = '';
    return fullOnset + vowel + coda;
  }

  function decodeClustersToLatin(clusterStrings) {
    const pieces = [];
    let isStart = true;
    for (let i = 0; i < clusterStrings.length; i++) {
      const cl = clusterStrings[i];
      if (PADA_SET.has(cl)) {
        const p = clusterUnicodeToLatin(cl, true);
        if (p) pieces.push(p);
        isStart = true;
        continue;
      }
      pieces.push(clusterUnicodeToLatin(cl, isStart));
      isStart = false;
    }

    const rawJoined = pieces.join('');
    const lex = global.AksaraLexicon;
    if (!lex || !lex.segmentClauseWords) return rawJoined.trim();

    let segmented = lex.segmentClauseWords(rawJoined);
    if (lex.isKnownJavaneseWord) {
      segmented = segmented
        .split(/\s+/)
        .map((tok) => {
          const m = tok.match(/^([.,;:!?]*)(.*?)([.,;:!?]*)$/);
          if (!m) return tok;
          const [, lead, core, trail] = m;
          const low = core.toLowerCase();
          if (low.startsWith('h') && low.length >= 2 && !lex.isKnownJavaneseWord(low)) {
            const dropH = low.slice(1);
            if (lex.isKnownJavaneseWord(dropH)) return lead + dropH + trail;
          }
          return tok;
        })
        .join(' ');
    }
    return segmented.replace(/\s+([,.;:!?])/g, '$1').trim();
  }

  /**
   * Penyusun baris & kluster geometri teruji v2.5 (`isRightPasangan`, `fullAnchor`, `avgBaseH * 1.05`).
   */
  function buildLinesFromDetections(detections) {
    const anchors = [];
    const modifiers = [];

    (detections || []).forEach((d, idx) => {
      if (!d || !Array.isArray(d.box) || d.box.length !== 4 || !d.label) return;
      const item = { ...d, m: boxMetrics(d.box), origIdx: idx };
      if (d.category === 'nglegena' || d.category === 'pada') {
        anchors.push({
          isPada: d.category === 'pada',
          base: item,
          pasangan: null,
          sandhangans: [],
          box: [...d.box],
        });
      } else {
        modifiers.push(item);
      }
    });

    if (!anchors.length) return { lines: [], orphanModifiers: modifiers };

    const baseOnly = anchors.filter((a) => !a.isPada);
    const refHList = baseOnly.length ? baseOnly : anchors;
    const avgBaseH = refHList.reduce((s, b) => s + b.base.m.h, 0) / refHList.length;

    anchors.sort((a, b) => a.base.m.cy - b.base.m.cy);
    const lines = [];

    for (const cl of anchors) {
      let placed = false;
      for (const line of lines) {
        const refItems = line.filter((it) => !it.isPada);
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
    lines.forEach((line) => line.sort((a, b) => a.base.m.x0 - b.base.m.x0));

    const syllableClusters = lines.flat().filter((c) => !c.isPada);
    const remainingMods = [];

    // 1. Pasangkan Pasangan terlebih dahulu (memperhitungkan p_ha, p_sa, p_pa di samping kanan!)
    for (const mod of modifiers) {
      if (mod.category === 'pasangan') {
        let bestHost = null;
        let bestScore = Infinity;
        const isRightPasangan = mod.label === 'p_ha' || mod.label === 'p_sa' || mod.label === 'p_pa';

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
        if (bestHost && !bestHost.pasangan) bestHost.pasangan = mod;
        else remainingMods.push(mod);
      } else {
        remainingMods.push(mod);
      }
    }

    // 2. Pasangkan seluruh Sandhangan & Pangkon relatif terhadap fullAnchor
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
              Math.max(bm.y1, cl.pasangan.m.y1),
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
      if (bestHost) bestHost.sandhangans.push(mod);
    }

    return { lines, orphanModifiers: [] };
  }

  /**
   * Penyempurnaan Dekoder Beam Konservatif per Baris:
   * - Pertama menyusun hasil mentah top-1 100% setia pada detektor.
   * - Memetakan kluster yang termasuk dalam kata OOV saja, lalu mencoba hipotesis
   *   terarah (segitiga konfusi pepet<->taling<->cecak<->wulu + sisip cecak jika kata penuh
   *   menjadi entri kamus kuat >= 5 huruf), DAN menerima koreksi hanya bila:
   *     1) Valid secara ortografi U+A980..U+A9DF
   *     2) Rasio kata kamus sah MENINGKAT dan skor LM kata bersih naik melebihi ambang konservatif.
   */
  function refineLineWithConservativeBeam(lineClusters) {
    const rawClStrings = lineClusters.map((cl) => {
      if (cl.isPada) return JAVANESE_UNICODE[cl.base.label] || '';
      const pas = cl.pasangan ? [cl.pasangan.label] : [];
      const mods = cl.sandhangans.map((s) => s.label);
      return buildClusterUnicode(cl.base.label, pas, mods);
    });

    const rawUnicode = rawClStrings.join('');
    const rawLatin = decodeClustersToLatin(rawClStrings);

    const lex = global.AksaraLexicon;
    if (!lex || !lex.scoreWordSequence || !lex.isKnownJavaneseWord) {
      return { rawUnicode, unicode: rawUnicode, latin: rawLatin, rawLatin, beamUsed: false, wordMeta: [] };
    }

    const rawEval = lex.scoreWordSequence(rawLatin);
    const rawOovCount = (rawEval.wordMeta || []).filter((w) => !w.known && w.word.length >= 3).length;
    const meanBaseConf = lineClusters.reduce((s, cl) => s + (cl.base.confidence || 0), 0) / Math.max(1, lineClusters.length);
    // Hanya jalankan Beam perbaikan jika sebagian besar konteks baris valid (>= 45% kata dikenal)
    // DAN keyakinan dasar detektor memadai — mencegah halusinasi LM pada coretan tak terbaca!
    if (rawOovCount === 0 || (rawEval.knownRatio || 0) < 0.45 || meanBaseConf < 0.76) {
      return {
        rawUnicode,
        unicode: rawUnicode,
        latin: rawEval.segmented || rawLatin,
        rawLatin,
        beamUsed: false,
        wordMeta: rawEval.wordMeta || [],
      };
    }

    // Identifikasi kluster mana yang menghasilkan potongan OOV pendek / awal kata bermasalah
    // Uji kandidat perbaikan lokal per jendela 1-2 kluster (jauh lebih aman daripada beam global bebas!)
    let curStrings = [...rawClStrings];
    let curLatin = rawEval.segmented || rawLatin;
    let curScore = rawEval.score;

    for (let i = 0; i < lineClusters.length; i++) {
      const cl = lineClusters[i];
      if (cl.isPada) continue;
      const pas = cl.pasangan ? [cl.pasangan.label] : [];
      const mods = cl.sandhangans.map((s) => s.label);

      // Kumpulkan variasi lokal yang ketat untuk kluster i (termasuk pasangan i..i+1 jika pepet/cecak)
      const localVariants = [];
      mods.forEach((ml, mi) => {
        for (const alt of (CONFUSABLE_MODS[ml] || [])) {
          const m2 = [...mods];
          m2[mi] = alt;
          localVariants.push({ clIdx: i, mods: m2, cost: 1.5, tag: `swap:${ml}->${alt}` });
        }
        // Cek juga penghapusan modifier hanya jika keyakinan detektor sangat rendah (< 0.48)
        const conf = cl.sandhangans[mi].confidence || 1.0;
        if (conf < 0.48) {
          const mDrop = mods.filter((_, k) => k !== mi);
          localVariants.push({ clIdx: i, mods: mDrop, cost: 2.2, tag: `drop:${ml}` });
        }
      });

      // Cek variasi topK kelas basis jika keyakinan < 0.72
      if ((cl.base.confidence || 1.0) < 0.72 && Array.isArray(cl.base.topK)) {
        for (const tk of cl.base.topK.slice(1, 3)) {
          if (tk.prob >= 0.16 && tk.category === 'nglegena' && tk.label !== cl.base.label) {
            localVariants.push({ clIdx: i, baseOverride: tk.label, mods, cost: 1.8, tag: `base:${tk.label}` });
          }
        }
      }

      // Hipotesis sisip `cecak` terarah hanya untuk perbaikan OOV panjang (mis. weda -> wedang)
      const hasPanyigeg = mods.some((m) => m === 'cecak' || m === 'layar' || m === 'wignyan');
      if (!hasPanyigeg && !mods.includes('pangkon')) {
        localVariants.push({ clIdx: i, mods: [...mods, 'cecak'], cost: 3.1, tag: 'ins:cecak' });
      }

      // Evaluasi perbaikan tunggal maupun pasangan bersebelahan (i dan i+1, mis. `taling` di i + `cecak` di i+1 -> wédang)
      const nextCl = i + 1 < lineClusters.length && !lineClusters[i + 1].isPada ? lineClusters[i + 1] : null;
      const pairPlans = localVariants.map((v) => [{ ...v }]);

      if (nextCl) {
        const nPas = nextCl.pasangan ? [nextCl.pasangan.label] : [];
        const nMods = nextCl.sandhangans.map((s) => s.label);
        const nHasPanyigeg = nMods.some((m) => m === 'cecak' || m === 'layar' || m === 'wignyan');
        const nextVars = [];
        nMods.forEach((ml, mi) => {
          for (const alt of (CONFUSABLE_MODS[ml] || [])) {
            const m2 = [...nMods];
            m2[mi] = alt;
            nextVars.push({ clIdx: i + 1, mods: m2, pas: nPas, cost: 1.5 });
          }
        });
        if (!nHasPanyigeg && !nMods.includes('pangkon')) {
          nextVars.push({ clIdx: i + 1, mods: [...nMods, 'cecak'], pas: nPas, cost: 3.1 });
        }
        for (const v1 of localVariants) {
          for (const v2 of nextVars) {
            pairPlans.push([v1, v2]);
          }
        }
      }

      let bestPlan = null;
      let bestNetGain = 0;

      for (const plan of pairPlans) {
        const trialStrings = [...curStrings];
        let planCost = 0;
        let allValid = true;

        for (const step of plan) {
          const targetCl = lineClusters[step.clIdx];
          const bLbl = step.baseOverride || targetCl.base.label;
          const pLbl = targetCl.pasangan ? [targetCl.pasangan.label] : [];
          const u = buildClusterUnicode(bLbl, pLbl, step.mods);
          const v = validateJavaneseUnicode(u);
          if (!v.valid) {
            allValid = false;
            break;
          }
          trialStrings[step.clIdx] = u;
          planCost += step.cost;
        }
        if (!allValid) continue;

        const trialLat = decodeClustersToLatin(trialStrings);
        const trialEv = lex.scoreWordSequence(trialLat);
        const trialKnownCount = (trialEv.wordMeta || []).filter((w) => w.known).length;
        const curKnownCount = (lex.scoreWordSequence(curLatin).wordMeta || []).filter((w) => w.known).length;

        // Syarat wajib konservatif:
        // 1) Setiap kata baru yang muncul harus kata kamus kuat >= 5 huruf bermakna
        // 2) Jumlah kata dikenal meningkat tanpa meninggalkan pecahan OOV
        // 3) Keuntungan bersih LM - biaya edit melebihi ambang 2.8 (atau 100% seluruh kata baris dikenal)
        const curMeta = lex.scoreWordSequence(curLatin).wordMeta || [];
        const gainedKnownWords = (trialEv.wordMeta || []).filter((w) => {
          return w.known && !curMeta.some((cw) => cw.known && cw.word === w.word);
        });
        const substantiveGain = gainedKnownWords.length > 0 && gainedKnownWords.every((w) => w.word.length >= 5 && w.meaning);
        const curPhrases = (lex.scoreWordSequence(curLatin).phraseHits || 0);
        const phraseImproved = (trialEv.phraseHits || 0) > curPhrases;
        if (!substantiveGain || trialKnownCount <= curKnownCount) continue;
        if ((trialEv.knownRatio || 0) < 0.90 && !phraseImproved) continue;

        const netGain = (trialEv.score - curScore) - planCost;
        if (netGain > bestNetGain + 2.0) {
          bestNetGain = netGain;
          bestPlan = { trialStrings, trialEv, trialLat };
        }
      }

      if (bestPlan) {
        curStrings = bestPlan.trialStrings;
        curLatin = bestPlan.trialEv.segmented || bestPlan.trialLat;
        curScore = bestPlan.trialEv.score;
      }
    }

    const finalEv = lex.scoreWordSequence(curLatin);
    return {
      rawUnicode,
      unicode: curStrings.join(''),
      latin: finalEv.segmented || curLatin,
      rawLatin,
      beamUsed: curStrings.join('') !== rawUnicode,
      wordMeta: finalEv.wordMeta || [],
    };
  }

  function reconstructJavaneseText(detections, options) {
    const opts = options && typeof options === 'object' ? options : {};
    const { lines, orphanModifiers } = buildLinesFromDetections(detections || []);

    if (lines.length === 0) {
      if (orphanModifiers.length > 0) {
        const sorted = [...orphanModifiers].sort((a, b) => a.m.cy - b.m.cy || a.m.x0 - b.m.x0);
        const uText = sorted.map((s) => JAVANESE_UNICODE[s.label] || s.unicode || '').join(' ');
        const lText = sorted.map((s) => s.label).join(' ');
        const resEmpty = {
          rawUnicodeText: uText,
          unicodeText: uText,
          latinText: lText,
          sentenceLatin: lText,
          sentenceIndonesian: '—',
          tokens: lText.split(/\s+/),
          clusters: [],
          wordDetails: [],
          orthoValid: false,
          orthoErrors: ['tidak ada aksara dasar'],
          beamUsed: false,
        };
        global.__aksaraLastOrthoResult = resEmpty;
        return resEmpty;
      }
      const blank = {
        rawUnicodeText: '',
        unicodeText: '—',
        latinText: '—',
        sentenceLatin: '—',
        sentenceIndonesian: '—',
        tokens: [],
        clusters: [],
        wordDetails: [],
        orthoValid: false,
        orthoErrors: [],
        beamUsed: false,
      };
      global.__aksaraLastOrthoResult = blank;
      return blank;
    }

    const rawLines = [];
    const uniLines = [];
    const latLines = [];
    const allWordMeta = [];
    let anyBeamUsed = false;

    for (const ln of lines) {
      const dec = opts.disableBeam
        ? {
            rawUnicode: ln.map((cl) => (cl.isPada ? (JAVANESE_UNICODE[cl.base.label] || '') : buildClusterUnicode(cl.base.label, cl.pasangan ? [cl.pasangan.label] : [], cl.sandhangans.map((s) => s.label)))).join(''),
            unicode: '',
            latin: '',
            beamUsed: false,
            wordMeta: [],
          }
        : refineLineWithConservativeBeam(ln);

      if (opts.disableBeam) {
        dec.unicode = dec.rawUnicode;
        dec.latin = decodeClustersToLatin(
          ln.map((cl) => (cl.isPada ? (JAVANESE_UNICODE[cl.base.label] || '') : buildClusterUnicode(cl.base.label, cl.pasangan ? [cl.pasangan.label] : [], cl.sandhangans.map((s) => s.label))))
        );
      }

      rawLines.push(dec.rawUnicode);
      uniLines.push(dec.unicode);
      latLines.push(dec.latin);
      if (dec.beamUsed) anyBeamUsed = true;
      (dec.wordMeta || []).forEach((wm) => allWordMeta.push(wm));
    }

    const unicodeText = uniLines.join('\n').trim() || '—';
    const rawUnicodeText = rawLines.join('\n').trim();
    const sentenceLatin = latLines
      .join(' ')
      .replace(/\s+/g, ' ')
      .replace(/\s+([,.;:!?])/g, '$1')
      .trim() || '—';

    const lex = global.AksaraLexicon;
    const sentenceIndonesian = lex && lex.translateJavaneseTokens && sentenceLatin !== '—'
      ? lex.translateJavaneseTokens(sentenceLatin)
      : sentenceLatin;

    const val = validateJavaneseUnicode(unicodeText);
    const result = {
      rawUnicodeText,
      unicodeText,
      latinText: sentenceLatin,
      sentenceLatin,
      sentenceIndonesian,
      tokens: sentenceLatin !== '—' ? sentenceLatin.split(/\s+/) : [],
      clusters: lines.flat().map((cl) => ({
        label: cl.base.label,
        unicode: cl.isPada
          ? (JAVANESE_UNICODE[cl.base.label] || '')
          : buildClusterUnicode(cl.base.label, cl.pasangan ? [cl.pasangan.label] : [], cl.sandhangans.map((s) => s.label)),
        confidence: cl.base.confidence,
        box: cl.base.box,
      })),
      wordDetails: allWordMeta,
      orthoValid: val.valid,
      orthoErrors: val.errors,
      beamUsed: anyBeamUsed,
    };

    global.__aksaraLastOrthoResult = result;
    if (typeof global.dispatchEvent === 'function' && typeof global.CustomEvent === 'function') {
      try {
        global.dispatchEvent(new global.CustomEvent('aksara:reconstructed', { detail: result }));
      } catch (_e) { /* ignore non-DOM */ }
    }
    return result;
  }

  const api = {
    JAVANESE_UNICODE,
    validateJavaneseUnicode,
    buildClusterUnicode,
    clusterUnicodeToLatin,
    decodeClustersToLatin,
    reconstructJavaneseText,
  };

  global.AksaraOrthography = api;
  if (typeof module !== 'undefined' && module.exports) {
    Object.assign(module.exports, api);
  }
})(typeof window !== 'undefined' ? window : globalThis);
