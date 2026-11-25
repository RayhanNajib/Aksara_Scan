/**
 * Mesin Bahasa Jawa Hibrid v3.0 — AksaraScan (Fase 4e & 4f)
 * Menggabungkan kamus kurasi (`javanese_lexicon.js`), korpus statistik 866k kata (`AKSARA_LM_DATA`),
 * morfologi afiks produktif Jawa (ng-/ny-/m-/n-/di-/ka-/ke-/dak-/kok-/tak-/sa-, -an/-ake/-aken/-e/-ku/-mu/-ne/-i/-en),
 * segmentasi kata DP, pemodelan char-trigram + unigram untuk Beam Decoder, dan penerjemah berjenjang.
 *
 * 100% berjalan luring di peramban; MT daring bersifat opsional opt-in (mati secara baku).
 */
(function (global) {
  'use strict';

  const baseLex = global.AksaraLexicon || {};
  const BASE_DICT = baseLex.JAVANESE_DICT || {};
  const lmData = global.AKSARA_LM_DATA || {};

  const LM_UNIGRAM = lmData.unigramLog10 || {};
  const LM_TRIGRAM = lmData.trigramLog10 || {};
  const LM_DICT = lmData.dict || {};
  const LM_PHRASES = lmData.phrases || {};

  // Kamus inti tambahan untuk peribahasa, sandhangan, dan kata klasik umum
  const EXTRA_IDIOMS = {
    'wedang': 'minuman hangat / seduhan',
    'wédang': 'minuman hangat / seduhan',
    'bubuk': 'bubuk / serbuk',
    'tanpa': 'tanpa',
    'gula': 'gula',
    'aja': 'jangan',
    'ngamuk': 'mengamuk',
    'mundhak': 'semakin / bertambah',
    'tuwa': 'tua',
    'nglegena': 'huruf dasar',
    'sandhangan': 'tanda vokal/konsonan',
    'pasangan': 'aksara pasangan',
    'aksara': 'aksara / huruf',
    'jawa': 'Jawa',
    'sega': 'nasi',
    'mangan': 'makan',
    'krama': 'halus / krama',
    'kothi': 'kera / monyet',
    'sapa': 'siapa',
    'nandur': 'menanam',
    'ngundhuh': 'memetik hasil',
    'becik': 'baik',
    'ketitik': 'terlihat / tampak',
    'ala': 'buruk',
    'ketara': 'kentara / nyata',
    'wong': 'orang',
    'urip': 'hidup',
    'ngombe': 'minum',
    'turu': 'tidur',
    'lunga': 'pergi',
    'teka': 'datang',
    'mulih': 'pulang',
    'sinau': 'belajar',
    'nulis': 'menulis',
    'maca': 'membaca',
    'tuku': 'membeli',
    'golek': 'mencari',
    'omah': 'rumah',
    'banyu': 'air',
    'geni': 'api',
    'angin': 'angin',
    'lemah': 'tanah',
  };

  const EXTRA_PHRASES = {
    'wedang bubuk': 'kopi/minuman bubuk hangat',
    'tanpa gula': 'tanpa gula',
    'aja ngamuk': 'jangan mengamuk',
    'mundhak tuwa': 'semakin cepat tua',
    'mangan sega': 'makan nasi',
    'aksara jawa': 'aksara Jawa',
    'becik ketitik': 'kebaikan pasti tampak',
    'ala ketara': 'keburukan pasti kentara',
  };

  // Gabungkan kamus tanpa menimpa entri manual kurasi di BASE_DICT
  const MERGED_DICT = Object.assign({}, LM_DICT, EXTRA_IDIOMS, BASE_DICT);
  const MERGED_PHRASES = Object.assign({}, LM_PHRASES, EXTRA_PHRASES);

  function stripJavaneseAccents(str) {
    return String(str || '')
      .toLowerCase()
      .replace(/[éè]/g, 'e')
      .replace(/ê/g, 'e');
  }

  function foldAO(bare) {
    // Alofoni vokal a/o pada dialek Jawa (mis. sapa/sopo, tuwa/tuwo, sega/sego)
    if (bare.endsWith('o') && bare.length >= 3) return bare.slice(0, -1) + 'a';
    if (bare.endsWith('a') && bare.length >= 3) return bare.slice(0, -1) + 'o';
    return null;
  }

  function directLookup(rawWord) {
    const low = String(rawWord || '').toLowerCase().trim();
    if (!low) return null;
    const bare = stripJavaneseAccents(low);
    const alt = foldAO(bare);

    const dictVal = MERGED_DICT[low] || MERGED_DICT[bare] || (alt && MERGED_DICT[alt]) || null;
    const uniLog =
      LM_UNIGRAM[low] !== undefined ? LM_UNIGRAM[low]
      : LM_UNIGRAM[bare] !== undefined ? LM_UNIGRAM[bare]
      : alt && LM_UNIGRAM[alt] !== undefined ? LM_UNIGRAM[alt] - 0.15
      : null;

    if (dictVal !== null || uniLog !== null) {
      return {
        word: low,
        bare,
        known: true,
        exact: LM_UNIGRAM[low] !== undefined || Boolean(MERGED_DICT[low]),
        dictMeaning: dictVal,
        unigramLog10: uniLog !== null ? uniLog : -4.4,
      };
    }
    return null;
  }

  const PREFIXES = [
    { p: 'dak', role: 'ku-' },
    { p: 'kok', role: 'kau-' },
    { p: 'tak', role: 'ku-' },
    { p: 'pra', role: 'pra-' },
    { p: 'ng', role: 'me-', nasals: ['', 'k', 'g'] },
    { p: 'ny', role: 'me-', nasals: ['c', 's'] },
    { p: 'm', role: 'me-', nasals: ['p', 'b', 'w'] },
    { p: 'n', role: 'me-', nasals: ['t', 'd', 'dh', 'th'] },
    { p: 'di', role: 'di-' },
    { p: 'ka', role: 'ter-/ke-' },
    { p: 'ke', role: 'ter-/ke-' },
    { p: 'sa', role: 'se-' },
    { p: 'pa', role: 'pe-' },
    { p: 'pe', role: 'pe-' },
    { p: 'pi', role: 'pe-' },
  ];

  const SUFFIXES = [
    { s: 'aken', role: '-kan' },
    { s: 'ake', role: '-kan' },
    { s: 'ana', role: '-ilah' },
    { s: 'na', role: '-kan' },
    { s: 'an', role: '-an' },
    { s: 'ku', role: 'ku' },
    { s: 'mu', role: 'mu' },
    { s: 'ne', role: '-nya' },
    { s: 'en', role: '-lah' },
    { s: 'a', role: '(andaikan/-lah)' },
    { s: 'e', role: '-nya' },
    { s: 'i', role: '-i' },
  ];

  function stemAndLookup(rawWord) {
    const exact = directLookup(rawWord);
    if (exact) return { ...exact, stem: exact.word, viaAffix: false, meaning: exact.dictMeaning };

    const low = String(rawWord || '').toLowerCase().trim();
    const bare = stripJavaneseAccents(low);
    if (bare.length < 4) return null;

    // 1. Coba lepas sufiks lalu prefiks secara kombinasional
    const sufCands = [{ body: bare, suf: null }];
    for (const sf of SUFFIXES) {
      if (bare.endsWith(sf.s) && bare.length - sf.s.length >= 3) {
        sufCands.push({ body: bare.slice(0, -sf.s.length), suf: sf });
      }
    }

    for (const sc of sufCands) {
      const hitBody = directLookup(sc.body);
      if (hitBody && sc.suf) {
        const baseM = hitBody.dictMeaning || sc.body;
        const m =
          sc.suf.s === 'ku' || sc.suf.s === 'mu' ? `${baseM}${sc.suf.role}`
          : sc.suf.s === 'ne' || sc.suf.s === 'e' ? `${baseM}-nya`
          : `${baseM} (${sc.suf.role})`;
        return {
          word: low,
          bare,
          known: true,
          exact: false,
          viaAffix: true,
          stem: sc.body,
          unigramLog10: hitBody.unigramLog10 - 0.55,
          meaning: m,
        };
      }

      for (const pf of PREFIXES) {
        if (!sc.body.startsWith(pf.p) || sc.body.length - pf.p.length < 3) continue;
        const rest = sc.body.slice(pf.p.length);
        const rootCands = pf.nasals ? pf.nasals.map((ins) => ins + rest) : [rest];
        for (const rc of rootCands) {
          const hitRoot = directLookup(rc);
          if (hitRoot) {
            const baseM = hitRoot.dictMeaning || rc;
            let combined = baseM;
            if (pf.role === 'di-') combined = `di-${baseM}`;
            else if (pf.role === 'me-') combined = `me-${baseM}`;
            else if (pf.role === 'ter-/ke-') combined = `ter-${baseM}`;
            else if (pf.role === 'ku-') combined = `ku-${baseM}`;
            else if (pf.role === 'kau-') combined = `kau-${baseM}`;
            else if (pf.role === 'se-') combined = `se-${baseM}`;
            if (sc.suf) combined += sc.suf.s === 'ne' || sc.suf.s === 'e' ? '-nya' : ` (${sc.suf.role})`;
            return {
              word: low,
              bare,
              known: true,
              exact: false,
              viaAffix: true,
              stem: rc,
              unigramLog10: hitRoot.unigramLog10 - 0.75,
              meaning: combined,
            };
          }
        }
      }
    }
    return null;
  }

  function getCharTrigramScore(word) {
    const clean = String(word || '').toLowerCase().replace(/[^a-zàáéèêëíîóôúû]/g, '');
    if (!clean) return -5.0;
    const padded = '^' + clean + '$';
    let sum = 0;
    let cnt = 0;
    for (let i = 0; i < padded.length - 1; i++) {
      // Gunakan konteks 2-char jika tersedia, sesuai format build_lm_and_lexicon.py (`'^w'`, `'wé'`, `'éd'`)
      const ctx = i === 0 ? '^' + padded[1] : padded.slice(i - 1, i + 1);
      const next = padded[i + 1];
      if (i === 0) {
        const tbl = LM_TRIGRAM['^' + clean[0]];
        if (tbl && clean.length > 1 && tbl[clean[1]] !== undefined) {
          sum += tbl[clean[1]];
        } else {
          sum += -2.8;
        }
        cnt += 1;
        continue;
      }
      const tbl = LM_TRIGRAM[ctx];
      if (tbl && tbl[next] !== undefined) {
        sum += tbl[next];
      } else {
        // Fallback tanpa aksen jika vokal beraksen tidak muncul di trigram
        const bareCtx = stripJavaneseAccents(ctx);
        const bareTbl = LM_TRIGRAM[bareCtx];
        const bareNext = stripJavaneseAccents(next);
        if (bareTbl && bareTbl[bareNext] !== undefined) {
          sum += bareTbl[bareNext] - 0.35;
        } else {
          sum += -4.6;
        }
      }
      cnt += 1;
    }
    return cnt > 0 ? sum / cnt : -4.8;
  }

  // Kata fungsi 1-2 huruf yang sah berdiri sendiri dalam segmentasi DP
  const SHORT_FUNCTION_WORDS = new Set([
    'di', 'ke', 'ka', 'sa', 'ing', 'lan', 'ya', 'yo', 'ta', 'po', 'kok', 'tak', 'dak',
    'nek', 'yen', 'wes', 'wis', 'ora', 'aja', 'ojo', 'ono', 'ana', 'iku', 'iki', 'kui',
    'aku', 'kowe', 'sing', 'wong', 'mu', 'ku', 'ne', 'si', 'bu', 'pak', 'mas', 'mbak',
  ]);

  function wordFrequencyScore(rawWord) {
    const clean = String(rawWord || '').toLowerCase().trim();
    const bare = stripJavaneseAccents(clean);
    const info = stemAndLookup(clean);
    const tri = getCharTrigramScore(clean);

    // Panjang minimal agar potongan 2-huruf acak (spt. 'ng', 'am') tidak mengalahkan kata panjang ('ngamuk')
    if (bare.length <= 2 && !SHORT_FUNCTION_WORDS.has(bare)) {
      return { known: false, score: -3.2 * Math.max(1, bare.length) + tri * 0.25, tri, info: null };
    }
    if (!info) {
      // Penalti proporsional panjang supaya pecahan OOV tidak mendapat untung
      return { known: false, score: -1.35 - 0.75 * bare.length + tri * 0.3, tri, info: null };
    }

    // Bonus proporsional panjang huruf + penalti batas kata -1.45 mencegah pemecahan kata sah ('bubuk' -> 'bu buk')
    const perCharVal = info.exact ? 0.92 : info.viaAffix ? 0.72 : 0.85;
    const lenReward = bare.length * perCharVal;
    const exactAccentBonus = MERGED_DICT[clean] ? 1.85 : 0.0;
    const freqPart = Math.max(-5.5, info.unigramLog10) * 0.22;
    const wordBoundaryCost = -1.45;
    return {
      known: true,
      score: lenReward + exactAccentBonus + freqPart + tri * 0.25 + wordBoundaryCost,
      tri,
      info,
    };
  }

  /**
   * Segmentasi klausa tanpa spasi atau ber-spasi tidak rapi memakai Pemrograman Dinamis (DP)
   * berdasarkan skor unigram + kamus + char-trigram + afiks Jawa.
   */
  function segmentClauseWords(clauseStr) {
    const raw = String(clauseStr || '').trim();
    if (!raw) return '';

    // Jika sudah memiliki spasi yang jelas antar kata valid, evaluasi tiap potongan tetap melalui DP jika ada gabung panjang
    const chunks = raw.split(/\s+/);
    const outWords = [];

    for (const chunk of chunks) {
      // Pisahkan tanda baca pinggir (, . ; : ! ?)
      const m = chunk.match(/^([.,;:!?\"'()]*)(.*?)([.,;:!?\"'()]*)$/);
      const leadPunc = m ? m[1] : '';
      const body = m ? m[2] : chunk;
      const trailPunc = m ? m[3] : '';

      if (!body) {
        if (leadPunc || trailPunc) outWords.push(leadPunc + trailPunc);
        continue;
      }

      const lower = body.toLowerCase();
      // Cek cepat jika utuh sudah dikenal & pendek (<= 11 huruf)
      if (lower.length <= 11 && stemAndLookup(lower)) {
        outWords.push(leadPunc + body + trailPunc);
        continue;
      }

      // DP pemecahan kata hingga panjang segmen 14 huruf
      const n = lower.length;
      const dp = new Array(n + 1).fill(-1e9);
      const prev = new Array(n + 1).fill(-1);
      dp[0] = 0;

      for (let i = 1; i <= n; i++) {
        for (let len = 1; len <= Math.min(14, i); len++) {
          const j = i - len;
          const piece = lower.slice(j, i);
          if (len === 1 && !['a', 'i', 'u', 'e', 'o'].includes(piece)) continue;
          const ev = wordFrequencyScore(piece);
          // Penalti kecil tiap batas kata tambahan sudah termasuk di wordFrequencyScore (-1.45)
          const segScore = ev.score - (j > 0 ? 0.15 : 0);
          if (dp[j] + segScore > dp[i]) {
            dp[i] = dp[j] + segScore;
            prev[i] = j;
          }
        }
      }

      if (prev[n] < 0) {
        outWords.push(chunk);
        continue;
      }
      const parts = [];
      let cur = n;
      while (cur > 0) {
        const p = prev[cur];
        parts.push(body.slice(p, cur));
        cur = p;
      }
      parts.reverse();
      parts[0] = leadPunc + parts[0];
      parts[parts.length - 1] = parts[parts.length - 1] + trailPunc;
      outWords.push(...parts);
    }

    return outWords
      .join(' ')
      .replace(/\s+([,.;:!?])/g, '$1');
  }

  /**
   * Penilai urutan kalimat Latin untuk Beam Decoder (`orthography.js`).
   */
  function scoreWordSequence(latinText) {
    const segmented = segmentClauseWords(latinText);
    const words = segmented
      .toLowerCase()
      .split(/\s+/)
      .map((w) => w.replace(/^[.,;:!?\"'()]+|[.,;:!?\"'()]+$/g, ''))
      .filter(Boolean);

    if (words.length === 0) {
      return { score: -10, segmented, words: [], knownRatio: 0, phraseHits: 0, wordMeta: [] };
    }

    let total = 0;
    let knownCount = 0;
    const wordMeta = [];

    for (const w of words) {
      const ev = wordFrequencyScore(w);
      total += ev.score;
      if (ev.known) knownCount += 1;
      wordMeta.push({
        word: w,
        known: ev.known,
        exact: ev.info ? ev.info.exact : false,
        score: Number(ev.score.toFixed(3)),
        meaning: ev.info ? ev.info.meaning : null,
      });
    }

    // Cek kecocokan frasa 2-kata dan 3-kata
    let phraseHits = 0;
    for (let i = 0; i < words.length - 1; i++) {
      const bg = stripJavaneseAccents(words[i]) + ' ' + stripJavaneseAccents(words[i + 1]);
      if (MERGED_PHRASES[bg]) {
        total += 2.2;
        phraseHits += 1;
      }
    }

    return {
      score: total,
      segmented,
      words,
      knownRatio: knownCount / words.length,
      phraseHits,
      wordMeta,
    };
  }

  /**
   * Penerjemah berjenjang Bahasa Jawa -> Bahasa Indonesia (100% Luring).
   * Menangani kalimat panjang, frasa, reduplikasi, serta imbuhan produktif.
   */
  function translateJavaneseTokens(tokensOrSentence, opts) {
    let words = [];
    if (Array.isArray(tokensOrSentence)) {
      const joined = tokensOrSentence
        .map((t) => (typeof t === 'string' ? t : t && t.latin ? t.latin : ''))
        .join('');
      const seg = segmentClauseWords(joined);
      words = seg.split(/\s+/).filter(Boolean);
    } else {
      const seg = segmentClauseWords(String(tokensOrSentence || ''));
      words = seg.split(/\s+/).filter(Boolean);
    }

    const outParts = [];
    let matched = 0;
    let totalContent = 0;

    for (let i = 0; i < words.length; i++) {
      const rawTok = words[i];
      const m = rawTok.match(/^([.,;:!?\"'()]*)(.*?)([.,;:!?\"'()]*)$/);
      const lead = m ? m[1] : '';
      const core = m ? m[2] : rawTok;
      const trail = m ? m[3] : '';
      if (!core) {
        outParts.push(rawTok);
        continue;
      }

      totalContent += 1;
      const coreLower = core.toLowerCase();
      const bareCore = stripJavaneseAccents(coreLower);

      // 1. Cek frasa 2 kata
      if (i + 1 < words.length) {
        const nextCore = words[i + 1].replace(/^[.,;:!?\"'()]+|[.,;:!?\"'()]+$/g, '').toLowerCase();
        const nextTrailMatch = words[i + 1].match(/[.,;:!?\"'()]+$/);
        const nextTrail = nextTrailMatch ? nextTrailMatch[0] : '';
        const pairKey = bareCore + ' ' + stripJavaneseAccents(nextCore);
        if (MERGED_PHRASES[pairKey]) {
          outParts.push(lead + MERGED_PHRASES[pairKey] + nextTrail);
          matched += 2;
          totalContent += 1;
          i += 1;
          continue;
        }
      }

      // 2. Cek kamus langsung / imbuhan
      const info = stemAndLookup(coreLower);
      if (info && info.meaning) {
        const firstGloss = String(info.meaning).split('/')[0].trim();
        outParts.push(lead + firstGloss + trail);
        matched += 1;
      } else if (info && info.known) {
        // Kata dikenal di korpus Jawa namun belum ada glosa ID (mis. kata serapan/nama umum)
        outParts.push(lead + coreLower + trail);
        matched += 0.6;
      } else {
        outParts.push(rawTok);
      }
    }

    const sentence = outParts
      .join(' ')
      .replace(/\s+([,.;:!?])/g, '$1')
      .replace(/(^\w|[.!?]\s+\w)/g, (c) => c.toUpperCase());

    if (opts && opts.detailed) {
      return {
        indonesian: sentence,
        coverageRatio: totalContent > 0 ? matched / totalContent : 0,
      };
    }
    return sentence;
  }

  /**
   * Hook MT daring opsional opt-in (HANYA aktif bila pengguna menyalakan toggle secara sadar).
   */
  async function optionalOnlineTranslate(text, isEnabled) {
    if (!isEnabled || !text || !text.trim()) return null;
    try {
      const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=jw|id`;
      const resp = await fetch(url);
      if (!resp.ok) return null;
      const data = await resp.json();
      const tr = data && data.responseData && data.responseData.translatedText;
      return tr || null;
    } catch (_err) {
      return null;
    }
  }

  global.AksaraLexicon = Object.assign(baseLex, {
    JAVANESE_DICT: MERGED_DICT,
    PHRASES: MERGED_PHRASES,
    stripJavaneseAccents,
    isKnownJavaneseWord: (w) => Boolean(stemAndLookup(w)),
    getCharTrigramScore,
    wordFrequencyScore,
    segmentClauseWords,
    scoreWordSequence,
    translateJavaneseTokens,
    optionalOnlineTranslate,
  });
})(typeof window !== 'undefined' ? window : globalThis);
