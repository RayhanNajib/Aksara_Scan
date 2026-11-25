"""
Pembangun Model Bahasa Jawa (Unigram + Trigram Karakter) & Kamus Dwibahasa Diperluas
untuk Dekoder Beam & Penerjemah Bertingkat AksaraScan v3.0 (Fase 4d, 4e, 4f).

Sumber terverifikasi:
  - `dataset/external/corpus_v2/sentences_validated.jsonl` (71.228 kalimat Jawa)
  - `dataset/external/corpus_v2/lexicon_expanded.json` (10.588 entri kamus MIT/CC-BY-SA)
  - `dataset/external/nusax/*.csv` & `dataset/external/nusa_writes/*-mt-*.csv` (alignment kata paralel)
  - Entri bawaan `app/web_camscanner/javanese_lexicon.js`

Keluaran:
  - `app/web_camscanner/javanese_lm_data.js` (`window.AKSARA_LM_DATA`, kompak < 650 KB)
"""
import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT_DIR = os.path.join(ROOT_DIR, "dataset", "external")
CORPUS_PATH = os.path.join(EXT_DIR, "corpus_v2", "sentences_validated.jsonl")
LEX_EXP_PATH = os.path.join(EXT_DIR, "corpus_v2", "lexicon_expanded.json")
OUT_JS = os.path.join(ROOT_DIR, "app", "web_camscanner", "javanese_lm_data.js")


def extract_parallel_word_and_phrase_alignments(max_dict_add=4000, max_phrases=500):
    """Tambahkan pasangan kata & frasa 2-kata dari NusaX-MT & NusaTranslation berbasis ko-okurensi & rasio Dice."""
    jv_count = Counter()
    id_count = Counter()
    cooc = Counter()
    phrase_cooc = Counter()
    pairs_seen = 0

    csv_files = glob.glob(os.path.join(EXT_DIR, "nusax", "*.csv")) + glob.glob(os.path.join(EXT_DIR, "nusa_writes", "*mt*.csv"))
    for p in csv_files:
        if "lexicon" in p or "javanese.csv" in p:
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                jv = (row.get("javanese") or row.get("jav") or row.get("text") or "").lower()
                ind = (row.get("indonesian") or row.get("ind") or "").lower()
                if not jv or not ind:
                    continue
                jw = re.findall(r"[a-zéèê]+", jv)
                iw = re.findall(r"[a-z]+", ind)
                if not (2 <= len(jw) <= 25 and 2 <= len(iw) <= 25):
                    continue
                pairs_seen += 1
                j_set = set(jw)
                i_set = set(iw)
                for w in j_set:
                    jv_count[w] += 1
                for w in i_set:
                    id_count[w] += 1
                for wj in j_set:
                    for wi in i_set:
                        cooc[(wj, wi)] += 1
                # Bigram phrases
                for k in range(len(jw) - 1):
                    bg_j = f"{jw[k]} {jw[k+1]}"
                    for m in range(len(iw) - 1):
                        bg_i = f"{iw[m]} {iw[m+1]}"
                        if abs(k - m) <= 2:
                            phrase_cooc[(bg_j, bg_i)] += 1

    aligned_words = {}
    for (wj, wi), c in cooc.most_common(40000):
        if c < 4:
            break
        dice = (2.0 * c) / (jv_count[wj] + id_count[wi])
        if dice >= 0.42 and wj not in aligned_words and len(wj) >= 2:
            aligned_words[wj] = wi
        if len(aligned_words) >= max_dict_add:
            break

    aligned_phrases = {}
    for (pj, pi), c in phrase_cooc.most_common(4000):
        if c < 5:
            break
        if pj not in aligned_phrases:
            aligned_phrases[pj] = pi
        if len(aligned_phrases) >= max_phrases:
            break

    return aligned_words, aligned_phrases, pairs_seen


def main():
    with open(LEX_EXP_PATH, "r", encoding="utf-8") as f:
        lex_exp = json.load(f)

    raw_bilingual = lex_exp.get("bilingual_entries", {})
    top_freq = lex_exp.get("top_word_freq", {})

    # Bersihkan kamus dwibahasa agar ringkas (ambil arti utama singkat per entri)
    clean_dict = {}
    for k, v in raw_bilingual.items():
        if not re.fullmatch(r"[a-zéèê]{2,18}", k):
            continue
        # Potong penjelasan kamus panjang ke 1-3 kata arti pertama
        short_v = re.split(r"[;,(/]", v)[0].strip()
        if short_v and len(short_v) <= 28:
            clean_dict[k] = short_v

    aligned_words, aligned_phrases, n_pairs = extract_parallel_word_and_phrase_alignments()
    for k, v in aligned_words.items():
        clean_dict.setdefault(k, v)

    # Kumpulkan Unigram Log-Probabilities dari korpus 71k kalimat Jawa
    word_counts = Counter()
    trigram_counts = defaultdict(Counter)
    total_words = 0

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            lat = rec["latin"].lower()
            for w in re.findall(r"[a-zéèê]+", lat):
                if 2 <= len(w) <= 18:
                    word_counts[w] += 1
                    total_words += 1
                    padded = f"^{w}$"
                    for i in range(len(padded) - 2):
                        trigram_counts[padded[i:i + 2]][padded[i + 2]] += 1

    # Simpan top 7000 kata Jawa beserta log10 probabilitas terkuantisasi (2 desimal)
    unigram_log10 = {}
    vocab_total = max(1, sum(c for _, c in word_counts.most_common(7000)))
    for w, c in word_counts.most_common(7000):
        if c >= 2 or w in clean_dict:
            lp = math.log10((c + 0.5) / (vocab_total + 7000 * 0.5))
            unigram_log10[w] = round(lp, 2)
    # Pastikan seluruh kata kamus dasar juga masuk vocab LM minimal
    for w in list(clean_dict.keys())[:4500]:
        if w not in unigram_log10:
            unigram_log10[w] = -5.2

    # Trigram karakter Jawa (simpan konteks paling informatif agar < 120 KB)
    trigram_log10 = {}
    for ctx, nxt_map in trigram_counts.items():
        tot = sum(nxt_map.values())
        if tot < 15:
            continue
        probs = {}
        for ch, c in nxt_map.items():
            if c >= 3:
                probs[ch] = round(math.log10((c + 0.2) / (tot + 30 * 0.2)), 2)
        if probs:
            trigram_log10[ctx] = probs

    # Gabungkan kamus hingga ~4500 entri terpenting (prioritaskan kata yang muncul di korpus / aligned)
    scored_dict_keys = sorted(
        clean_dict.keys(),
        key=lambda w: (word_counts.get(w, 0) + word_counts.get(re.sub(r"[éèê]", "e", w), 0), -len(w)),
        reverse=True,
    )
    compact_dict = {k: clean_dict[k] for k in scored_dict_keys[:4500]}

    payload = {
        "version": "3.0",
        "stats": {
            "corpus_words": total_words,
            "unigram_vocab": len(unigram_log10),
            "dict_entries": len(compact_dict),
            "phrase_entries": len(aligned_phrases),
            "trigram_contexts": len(trigram_log10),
            "parallel_pairs_scanned": n_pairs,
        },
        "unigramLog10": unigram_log10,
        "trigramLog10": trigram_log10,
        "dict": compact_dict,
        "phrases": aligned_phrases,
    }

    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write("// Generated by tools/lm/build_lm_and_lexicon.py (AksaraScan v3.0)\n")
        f.write("window.AKSARA_LM_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")

    size_kb = os.path.getsize(OUT_JS) / 1024.0
    print(f"[LM+Lexicon] Tersimpan {OUT_JS} ({size_kb:.1f} KB) | stats={payload['stats']}")


if __name__ == "__main__":
    main()
