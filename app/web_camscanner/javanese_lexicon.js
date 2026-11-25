/**
 * Segmenter Kata Dinamis & Penerjemah Kalimat Jawa -> Indonesia (AksaraScan v2.5)
 * Memecah deretan suku kata Aksara Jawa tanpa spasi (scriptio continua)
 * menjadi kata-kata Bahasa Jawa yang valid menggunakan Dynamic Programming,
 * lalu menerjemahkannya ke dalam bentuk kalimat Bahasa Indonesia yang utuh.
 */

const JAVANESE_DICT = {
  // Pronomina & Orang
  aku: 'saya', kula: 'saya', kawula: 'saya', dalem: 'saya',
  kowe: 'kamu', sampeyan: 'Anda', panjenengan: 'Anda', sliramu: 'kamu',
  dheweke: 'dia', piyambakipun: 'beliau', kita: 'kita', kabeh: 'semua',
  wong: 'orang', tiyang: 'orang', bocah: 'anak', lare: 'anak',
  bapak: 'ayah', rama: 'ayah', ibu: 'ibu', biyung: 'ibu', simbah: 'kakek/nenek',
  eyang: 'kakek/nenek', kakang: 'kakak', mas: 'kakak', mbakyu: 'kakak perempuan',
  adhi: 'adik', adhi: 'adik', rayi: 'adik', kanca: 'teman', rencang: 'teman',
  guru: 'guru', murid: 'murid', siswa: 'siswa', ratu: 'raja', prabu: 'raja',
  tani: 'petani', dhokter: 'dokter',

  // Kata Kerja Umum (Ngoko & Krama)
  mangan: 'makan', nedha: 'makan', dhahar: 'makan',
  ngombe: 'minum', ngunjuk: 'minum', nginum: 'minum',
  turu: 'tidur', tilem: 'tidur', sare: 'tidur',
  tangi: 'bangun', wungu: 'bangun',
  mlaku: 'berjalan', mlampah: 'berjalan', mlayu: 'berlari', mlajeng: 'berlari',
  lunga: 'pergi', kesah: 'pergi', tindak: 'pergi',
  teka: 'datang', dhateng: 'datang', rawuh: 'datang',
  mulih: 'pulang', mantuk: 'pulang', kondur: 'pulang',
  sinau: 'belajar', maca: 'membaca', maos: 'membaca', nulis: 'menulis', nyerat: 'menulis',
  nggambar: 'menggambar', nyanyi: 'menyanyi', nembang: 'bernyanyi',
  tuku: 'membeli', tumbas: 'membeli', mundhut: 'membeli',
  adol: 'menjual', sade: 'menjual',
  gawe: 'membuat', damel: 'membuat', makarya: 'bekerja', nyambut: 'bekerja',
  golek: 'mencari', pados: 'mencari', nemu: 'menemukan',
  deleng: 'melihat', mirsani: 'melihat', nonton: 'menonton',
  rungu: 'mendengar', mireng: 'mendengar',
  takon: 'bertanya', nyuwun: 'meminta', jaluk: 'minta', weneh: 'memberi', menehi: 'memberi',
  ngamuk: 'marah-marah', nesu: 'marah', duka: 'marah',
  guyu: 'tertawa', ngguyu: 'tertawa', mesem: 'tersenyum', nangis: 'menangis',
  lungguh: 'duduk', lenggah: 'duduk', ngadeg: 'berdiri', jumeneng: 'berdiri',
  tandang: 'mengerjakan', tandur: 'menanam', panen: 'panen',
  masak: 'memasak', adus: 'mandi', siram: 'mandi',
  tulung: 'tolong', rewang: 'membantu', jaga: 'menjaga', mikul: 'memikul', mendhem: 'mengubur',
  urip: 'hidup', gesang: 'hidup', mati: 'mati', seda: 'wafat',
  tresna: 'cinta', sayang: 'sayang', eling: 'ingat', lali: 'lupa', supe: 'lupa',
  ngerti: 'mengerti', mangertos: 'mengerti', ngira: 'mengira', pikir: 'pikir',
  dadi: 'menjadi', dados: 'menjadi', ana: 'ada', wonten: 'ada',
  duwe: 'punya', gadhah: 'punya', kagungan: 'memiliki',
  kudu: 'harus', kedah: 'harus', oleh: 'boleh', angsal: 'mendapat', entuk: 'mendapat',
  bisa: 'bisa', saged: 'bisa', gelem: 'mau', purun: 'mau', kersa: 'berkenan',
  wedang: 'minuman seduh', bubuk: 'bubuk', gula: 'gula', uyah: 'garam',

  // Kata Tugas, Partikel, Preposisi, Negasi
  aja: 'jangan', sampun: 'jangan/sudah', ora: 'tidak', mboten: 'tidak', dudu: 'bukan',
  tanpa: 'tanpa', lan: 'dan', sarta: 'serta', karo: 'dengan', kalih: 'dengan',
  nganggo: 'memakai', ngagem: 'memakai',
  ing: 'di', nang: 'di', menyang: 'ke', dhateng: 'ke', saka: 'dari', saking: 'dari',
  kanggo: 'untuk', kagem: 'untuk', marang: 'kepada',
  yen: 'jika', menawa: 'apabila', nalika: 'ketika', sadurunge: 'sebelum',
  sawise: 'setelah', banjur: 'lalu', lajeng: 'lalu', nanging: 'tetapi', ananging: 'namun',
  amarga: 'karena', awit: 'sebab', mula: 'oleh karena itu', supaya: 'agar',
  mundhak: 'nanti makin', mengko: 'nanti', saiki: 'sekarang', sakniki: 'sekarang',
  wingi: 'kemarin', sesuk: 'besok', benjang: 'besok', mau: 'tadi',
  wis: 'sudah', sampun: 'sudah', durung: 'belum', dereng: 'belum',
  isih: 'masih', taksih: 'masih', lagi: 'sedang', arep: 'akan', badhe: 'akan',
  sing: 'yang', ingkang: 'yang', kang: 'yang',
  iki: 'ini', punika: 'ini/itu', iku: 'itu', kuwi: 'itu', kae: 'itu sana',
  apa: 'apa', punapa: 'apa', sapa: 'siapa', sinten: 'siapa',
  pira: 'berapa', pinten: 'berapa', endi: 'mana', pundi: 'mana',
  kapan: 'kapan', kepriye: 'bagaimana', kadospundi: 'bagaimana', geneya: 'mengapa',

  // Kata Sifat & Keterangan
  tuwa: 'tua', sepuh: 'tua', enom: 'muda', anem: 'muda',
  apik: 'baik', sae: 'baik', becik: 'baik', ala: 'buruk', awon: 'buruk',
  bener: 'benar', leres: 'benar', salah: 'salah', luput: 'keliru',
  gedhe: 'besar', ageng: 'besar', cilik: 'kecil', alit: 'kecil',
  dawa: 'panjang', panjang: 'panjang', cendhak: 'pendek',
  dhuwur: 'tinggi', inggil: 'tinggi', cendhek: 'rendah', jero: 'dalam', cethek: 'dangkal',
  panas: 'panas', adhem: 'dingin', asrep: 'dingin', anget: 'hangat',
  legi: 'manis', pait: 'pahit', asin: 'asin', kecut: 'asam', pedhes: 'pedas',
  seneng: 'senang', bungah: 'gembira', susah: 'sedih', sedhih: 'sedih',
  pinter: 'pintar', wasis: 'pandai', bodho: 'bodoh', sregep: 'rajin', kesed: 'malas',
  resik: 'bersih', reged: 'kotor', padhang: 'terang', peteng: 'gelap',
  sehat: 'sehat', waras: 'sehat', lara: 'sakit', gerah: 'sakit',
  sabar: 'sabar', ikhlas: 'ikhlas', legawa: 'lapang dada', narima: 'menerima',
  cepet: 'cepat', enggal: 'segera', alon: 'pelan', rindhik: 'lambat',
  akeh: 'banyak', kathah: 'banyak', sithik: 'sedikit', sekedhik: 'sedikit',
  anyar: 'baru', enggal: 'baru', lawas: 'lama', dangu: 'lama',
  rame: 'ramai', sepi: 'sepi', ayem: 'tenteram', tentrem: 'damai',

  // Kata Benda, Makanan, Alam, Waktu, Budaya
  sega: 'nasi', sekul: 'nasi', banyu: 'air', toya: 'air',
  kopi: 'kopi', teh: 'teh', susu: 'susu', roti: 'roti', goreng: 'goreng',
  tempe: 'tempe', tahu: 'tahu', iwak: 'ikan/lauk', ulam: 'ikan/lauk',
  endhog: 'telur', tigan: 'telur', daging: 'daging', sayur: 'sayur', jangan: 'sayur',
  godhong: 'daun', ron: 'daun', kembang: 'bunga', sekar: 'bunga',
  woh: 'buah', wit: 'pohon', suket: 'rumput', pari: 'padi', beras: 'beras', uwos: 'beras',
  omah: 'rumah', griya: 'rumah', dalem: 'rumah', sekolah: 'sekolah', pawiyatan: 'sekolah',
  kelas: 'kelas', buku: 'buku', serat: 'surat/tulisan', pulpen: 'pulpen', potlot: 'pensil',
  meja: 'meja', kursi: 'kursi', lawang: 'pintu', konten: 'pintu', jendela: 'jendela',
  dalan: 'jalan', margi: 'jalan', desa: 'desa', dhusun: 'dusun',
  kutha: 'kota', kitha: 'kota', kotha: 'kota', negara: 'negara', praja: 'negara',
  pasar: 'pasar', peken: 'pasar', warung: 'warung', toko: 'toko',
  aksara: 'aksara', jawa: 'Jawa', hanacaraka: 'Hanacaraka', carakan: 'Carakan',
  basa: 'bahasa', tembung: 'kata', ukara: 'kalimat', cerita: 'cerita', dongeng: 'dongeng',
  krama: 'krama/sopan', ngoko: 'ngoko', madya: 'madya',
  dina: 'hari', dinten: 'hari', wulan: 'bulan', sasi: 'bulan', taun: 'tahun', warsa: 'tahun',
  esuk: 'pagi', enjang: 'pagi', awan: 'siang', siyang: 'siang',
  sore: 'sore', sonten: 'sore', bengi: 'malam', dalu: 'malam',
  langit: 'langit', mega: 'awan', udan: 'hujan', jawah: 'hujan',
  angin: 'angin', bayu: 'angin', geni: 'api', agni: 'api',
  lemah: 'tanah', siti: 'tanah', bumi: 'bumi', segara: 'laut', samodra: 'samudra',
  gunung: 'gunung', redi: 'gunung', kali: 'sungai', lepen: 'sungai', alas: 'hutan', wana: 'hutan',
  srengenge: 'matahari', surya: 'matahari', rembulan: 'bulan', candra: 'bulan', lintang: 'bintang',
  pitik: 'ayam', kucing: 'kucing', asu: 'anjing', sapi: 'sapi', kebo: 'kerbau',
  jaran: 'kuda', kuda: 'kuda', manuk: 'burung', peksi: 'burung', wedhus: 'kambing',
  duwit: 'uang', arta: 'uang', rega: 'harga', regi: 'harga',
  ati: 'hati', manah: 'hati', galih: 'hati', sirah: 'kepala', mustaka: 'kepala',
  mripat: 'mata', soca: 'mata', tangan: 'tangan', asta: 'tangan',
  sikil: 'kaki', sampeyan: 'kaki', suku: 'kaki', lambe: 'bibir', lathi: 'bibir',
  klambi: 'baju', rasukan: 'pakaian', sandhang: 'pakaian', pangan: 'pangan', papan: 'tempat tinggal',

  // Angka
  siji: 'satu', setunggal: 'satu', loro: 'dua', kalih: 'dua', telu: 'tiga', tiga: 'tiga',
  papat: 'empat', sekawan: 'empat', lima: 'lima', gangsal: 'lima',
  enem: 'enam', pitu: 'tujuh', wolu: 'delapan', sanga: 'sembilan',
  sepuluh: 'sepuluh', sedasa: 'sepuluh', satus: 'seratus', sewu: 'seribu'
};

// Normalisasi fonetik untuk pencocokan kamus (menyamakan variasi e/ê/é/è dan akhiran o/a)
function normPhonetic(str) {
  return str.toLowerCase()
    .replace(/[éèêë]/g, 'e')
    .replace(/dh/g, 'dh')
    .replace(/th/g, 'th');
}

const PHONETIC_MAP = {};
Object.keys(JAVANESE_DICT).forEach(k => {
  const nk = normPhonetic(k);
  if (!PHONETIC_MAP[nk]) PHONETIC_MAP[nk] = [];
  PHONETIC_MAP[nk].push({ word: k, id: JAVANESE_DICT[k] });
});

/**
 * Pecah string tanpa spasi (scriptio continua) menjadi deretan kata Jawa
 * menggunakan Dynamic Programming dengan dukungan aturan peluluhan 'ha' -> 'a/i/u/e/o'.
 */
function segmentClauseWords(rawClause) {
  const clean = rawClause.trim();
  if (!clean) return [];

  // Hasilkan kandidat variasi huruf awal 'h' -> luluh jika relevan
  // Eksplorasi DP: pada tiap indeks i, coba slice kata panjang 2..12
  const n = clean.length;
  const dp = new Array(n + 1).fill(null);
  dp[0] = { score: 0, words: [] };

  for (let i = 0; i < n; i++) {
    if (!dp[i]) continue;
    for (let len = 2; len <= Math.min(12, n - i); len++) {
      const sub = clean.slice(i, i + len);
      const candidates = [sub];
      // Aturan klasik aksara Jawa: suku kata diawali 'h' bisa dibaca vokal murni (ha->a, hi->i, hu->u, he->e, ho->o)
      if (sub.startsWith('h') && len >= 3 && 'aiueoéèê'.includes(sub[1])) {
        candidates.push(sub.slice(1));
      }

      for (const cand of candidates) {
        const key = normPhonetic(cand);
        const match = PHONETIC_MAP[key];
        if (match) {
          // Bonus besar untuk kata kamus valid; bonus kuadrat panjang untuk preferensi kata panjang utuh
          const wordScore = 10 + cand.length * cand.length * 1.4;
          const totalScore = dp[i].score + wordScore;
          if (!dp[i + len] || totalScore > dp[i + len].score) {
            dp[i + len] = {
              score: totalScore,
              words: [...dp[i].words, { jv: cand, id: match[0].id, matched: true }]
            };
          }
        }
      }
    }

    // Fallback 1 karakter (bila potongan tidak ada di kamus agar DP tidak pernah buntu)
    const ch = clean.slice(i, i + 1);
    const prev = dp[i];
    const lastWord = prev.words.length ? prev.words[prev.words.length - 1] : null;
    if (lastWord && !lastWord.matched) {
      const combined = lastWord.jv + ch;
      const wordsCopy = prev.words.slice(0, -1);
      wordsCopy.push({ jv: combined, id: null, matched: false });
      const totalScore = prev.score - 0.5;
      if (!dp[i + 1] || totalScore > dp[i + 1].score) {
        dp[i + 1] = { score: totalScore, words: wordsCopy };
      }
    } else {
      const totalScore = prev.score - 1.5;
      if (!dp[i + 1] || totalScore > dp[i + 1].score) {
        dp[i + 1] = {
          score: totalScore,
          words: [...prev.words, { jv: ch, id: null, matched: false }]
        };
      }
    }
  }

  return dp[n] ? dp[n].words : [{ jv: clean, id: null, matched: false }];
}

/**
 * Menerima daftar klausa/baris dari orthography.js lalu menghasilkan:
 *  - sentenceLatin: kalimat bacaan Latin tersambung dengan kapitalisasi & tanda baca rapi
 *  - sentenceIndonesian: terjemahan kalimat dalam Bahasa Indonesia
 */
function translateJavaneseTokens(tokens) {
  if (!tokens || !tokens.length) {
    return { sentenceLatin: '—', sentenceIndonesian: '—' };
  }

  // Pecah berdasarkan tanda baca pada_lingsa (,) dan pada_lungsi (.)
  const segments = [];
  let buf = [];
  for (const tok of tokens) {
    if (tok.isPunct) {
      if (buf.length) {
        segments.push({ type: 'clause', text: buf.join('') });
        buf = [];
      }
      if (tok.punctChar) {
        segments.push({ type: 'punct', char: tok.punctChar });
      }
    } else {
      buf.push(tok.latin);
    }
  }
  if (buf.length) {
    segments.push({ type: 'clause', text: buf.join('') });
  }

  const outLatParts = [];
  const outIdParts = [];
  let needCap = true;

  for (const seg of segments) {
    if (seg.type === 'punct') {
      if (outLatParts.length) outLatParts[outLatParts.length - 1] += seg.char;
      if (outIdParts.length) outIdParts[outIdParts.length - 1] += seg.char;
      if (seg.char === '.') needCap = true;
    } else {
      const words = segmentClauseWords(seg.text);
      const latWords = words.map((w, idx) => {
        let s = w.jv;
        if (needCap && idx === 0 && s.length) {
          s = s[0].toUpperCase() + s.slice(1);
        }
        return s;
      });
      const idWords = words.map((w, idx) => {
        let s = w.id ? w.id : w.jv;
        if (needCap && idx === 0 && s.length) {
          s = s[0].toUpperCase() + s.slice(1);
        }
        return s;
      });
      needCap = false;
      outLatParts.push(latWords.join(' '));
      outIdParts.push(idWords.join(' '));
    }
  }

  return {
    sentenceLatin: outLatParts.join(' ').trim() || '—',
    sentenceIndonesian: outIdParts.join(' ').trim() || '—'
  };
}

window.AksaraLexicon = {
  JAVANESE_DICT,
  segmentClauseWords,
  translateJavaneseTokens
};
