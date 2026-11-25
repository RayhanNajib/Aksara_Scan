"""
Pembangun Korpus Teks Jawa & Kamus Dwibahasa Tervalidasi — AksaraScan v3.0 (Fase 2a & 4f).
Menggabungkan kalimat Jawa dari sumber berlisensi OK:
  - NusaX (datasets/mt/{train,valid,test}.csv + datasets/lexicon/javanese.csv)
  - NusaWrites (nusa_kalimat-mt-jav-*.csv + nusa_alinea-paragraph-jav-*.csv)
  - Sastra Jawa CSV (dictionary.csv, dasanama.csv)
  - Gatra-1 Javanese (parquet)
  - Wikipedia Bahasa Jawa (jvwiki-latest-pages-articles.xml.bz2, opsional difilter bersih)
  - Paribasan & kalimat pendek kasus sulit (taling vs pepet, pasangan+suku, cecak/layar/wignyan)

Menghasilkan:
  1. `dataset/external/corpus_v2/sentences_validated.jsonl` (kalimat Latin -> Unicode 55-kelas tervalidasi + skor kesulitan)
  2. `dataset/external/corpus_v2/lexicon_expanded.json` (kamus dwibahasa Jawa->Indonesia & tabel kata valid untuk DP/Beam)
"""
import bz2
import csv
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter

from orthography_rules import latin_to_javanese_55, validate_javanese_unicode

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT_DIR = os.path.join(ROOT_DIR, "dataset", "external")
OUT_DIR = os.path.join(EXT_DIR, "corpus_v2")

# Kumpulan paribasan & kalimat khusus kasus sulit (taling vs pepet, pasangan bertumpuk, sandhangan panyigeg)
HARD_PARIBASAN_AND_CASES = [
    ("Wédang bubuk tanpa gula, aja ngamuk mundhak tuwa.", "Minuman seduh bubuk tanpa gula, jangan marah-marah nanti makin tua."),
    ("Becik ketitik ala ketara.", "Kebaikan akan terlihat, keburukan akan tampak."),
    ("Jer basuki mawa béya.", "Setiap keberhasilan membutuhkan pengorbanan/biaya."),
    ("Sapa nandur bakal ngundhuh.", "Siapa menanam akan menuai."),
    ("Ajining dhiri dumunung ana ing lathi.", "Harga diri seseorang terletak pada ucapannya."),
    ("Rukun agawe santosa, crah agawe bubrah.", "Kerukunan membuat sentosa, pertengkaran membuat hancur."),
    ("Kacang mangsa ninggal lanjaran.", "Anak tidak jauh sifatnya dari orang tuanya."),
    ("Mikul dhuwur mendhem jero.", "Menjunjung tinggi martabat dan menjaga rahasia keluarga."),
    ("Adigang adigung adiguna.", "Mengandalkan kekuatan, kebesaran, dan kepandaian secara sombong."),
    ("Ing ngarsa sung tuladha, ing madya mangun karsa, tut wuri handayani.", "Di depan memberi teladan, di tengah membangun kemauan, di belakang memberi dorongan."),
    ("Mangan sega goreng nganggo tempe lan tahu ing warung.", "Makan nasi goreng memakai tempe dan tahu di warung."),
    ("Krama kothi sinau nulis aksara Jawa ing buku tulis.", "Krama kothi belajar menulis aksara Jawa di buku tulis."),
    ("Bapak tindak menyang pasar mundhut gedhang, gula, lan uyah.", "Ayah pergi ke pasar membeli pisang, gula, dan garam."),
    ("Ibu masak jangan bayem, sambel trasi, karo iwak pindhang.", "Ibu memasak sayur bayam, sambal terasi, dan ikan pindang."),
    ("Tembang macapat pangkur, sinom, pocung, lan mijil.", "Tembang macapat pangkur, sinom, pocung, dan mijil."),
    ("Kucing ireng turu angler ing ndhuwur meja kayu.", "Kucing hitam tidur pulas di atas meja kayu."),
    ("Srengéngé esuk padhang sumunar ing langit wetan.", "Matahari pagi bersinar terang di langit timur."),
    ("Kembang mlathi wangi banget nalika sore.", "Bunga melati harum sekali ketika sore."),
    ("Téla pohung dibakar nganggo areng panas.", "Singkong dibakar memakai arang panas."),
    ("Péngin pinter kudu sregep maca buku saben dina.", "Ingin pintar harus rajin membaca buku setiap hari."),
    ("Méga mendhung peteng pratandha bakal udan deres.", "Awan mendung gelap pertanda akan hujan deras."),
    ("Kéré munggah balé.", "Orang miskin mendapat kedudukan tinggi."),
    ("Kêrêp bêbêg kêpêpêt ing dalan ciyut.", "Sering terhambat terjepit di jalan sempit."),
    ("Pépéklan pêpêt béda uniné ing aksara.", "Bunyi é dan ê berbeda ucapannya di aksara."),
    ("Cakra krêtêg ngêmu swara krêpêt lan kréta.", "Cakra keret mengandung suara krepet dan kereta."),
]


def is_clean_javanese_text(txt):
    if not txt or len(txt) < 6 or len(txt) > 160:
        return False
    # Hanya huruf Latin dasar + é/è/ê + spasi + koma/titik
    if re.search(r"[0-9@#$%^&*_+=<>{}\[\]|\\/~`]", txt):
        return False
    words = re.findall(r"[a-zA-ZéèêëÉÈÊË]+", txt)
    if len(words) < 2 or len(words) > 22:
        return False
    # Hindari kata terlalu panjang/singkatan huruf kapital
    if any(len(w) > 18 for w in words):
        return False
    return True


def score_difficulty(uni):
    """Hitung bobot kesulitan ortografi (cecak/layar/wignyan/pangkon/taling/pepet/pasangan)."""
    feats = {
        "taling": uni.count("\uA9BA"),
        "pepet": uni.count("\uA9BC"),
        "cecak": uni.count("\uA981"),
        "layar": uni.count("\uA982"),
        "wignyan": uni.count("\uA983"),
        "pangkon_pasangan": uni.count("\uA9C0"),
        "suku": uni.count("\uA9B8"),
        "wulu": uni.count("\uA9B6"),
    }
    return feats


def build_corpus_and_lexicon(max_wiki_sentences=8000):
    os.makedirs(OUT_DIR, exist_ok=True)
    sentences = []
    lexicon = {}
    word_freq = Counter()
    seen_latin = set()

    def add_pair(jav, ind="", src="manual"):
        jav = re.sub(r"\s+", " ", (jav or "").strip())
        if not is_clean_javanese_text(jav):
            return
        key = jav.lower()
        if key in seen_latin:
            return
        # Konversi ke Unicode 55-kelas dan validasi ketat
        uni = latin_to_javanese_55(jav, add_adeg=(len(jav.split()) >= 4))
        ok, errs, clusters = validate_javanese_unicode(uni, strict_55=True)
        if not ok or not clusters:
            return
        seen_latin.add(key)
        feats = score_difficulty(uni)
        sentences.append({
            "source": src,
            "latin": jav,
            "indonesian": (ind or "").strip(),
            "unicode": uni,
            "clusters": len(clusters),
            "feats": feats,
        })
        for w in re.findall(r"[a-zA-Zéèê]+", jav.lower()):
            word_freq[w] += 1

    # 1. Hard Paribasan
    for jv, id_ in HARD_PARIBASAN_AND_CASES:
        add_pair(jv, id_, "hard_paribasan")

    # 2. NusaX MT & Lexicon
    nusax_dir = os.path.join(EXT_DIR, "nusax")
    for p in glob.glob(os.path.join(nusax_dir, "*.csv")):
        fn = os.path.basename(p).lower()
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            cols = [c.lower() for c in (reader.fieldnames or [])]
            for row in reader:
                if "lexicon" in p or fn == "javanese.csv":
                    # lexicon csv columns vary: inspect dynamically
                    jv_w = (row.get("javanese") or row.get("word") or row.get("source") or "").strip().lower()
                    id_w = (row.get("indonesian") or row.get("meaning") or row.get("target") or "").strip().lower()
                    if jv_w and id_w and " " not in jv_w:
                        lexicon.setdefault(jv_w, id_w)
                else:
                    jv = row.get("javanese") or row.get("jav") or ""
                    ind = row.get("indonesian") or row.get("ind") or ""
                    add_pair(jv, ind, "nusax_mt")

    # 3. NusaWrites
    nw_dir = os.path.join(EXT_DIR, "nusa_writes")
    for p in glob.glob(os.path.join(nw_dir, "*.csv")):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                jv = row.get("text") or row.get("jav") or row.get("sentence") or row.get("paragraph") or ""
                ind = row.get("ind") or row.get("indonesian") or ""
                # Pecah paragraf menjadi kalimat pendek agar cocok untuk baris halaman sintetis
                for part in re.split(r"(?<=[.!?])\s+", jv):
                    add_pair(part, ind if part == jv else "", "nusa_writes")

    # 4. Sastra Jawa CSV (MIT)
    sj_dir = os.path.join(EXT_DIR, "sastra_jawa")
    for p in glob.glob(os.path.join(sj_dir, "*.csv")):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vals = [str(v or "").strip() for v in row.values()]
                if len(vals) >= 2 and vals[0] and vals[1]:
                    w = vals[0].lower()
                    if re.fullmatch(r"[a-z]{2,16}", w):
                        lexicon.setdefault(w, vals[1].lower())

    # 5. Wikipedia Bahasa Jawa dump (jika ada)
    wiki_bz2 = os.path.join(EXT_DIR, "jvwiki", "jvwiki-latest-pages-articles.xml.bz2")
    if os.path.exists(wiki_bz2):
        wiki_added = 0
        try:
            with bz2.open(wiki_bz2, "rb") as f:
                context = ET.iterparse(f, events=("end",))
                for event, elem in context:
                    if elem.tag.endswith("text") and elem.text:
                        txt = elem.text
                        # Bersihkan markup wiki sederhana
                        if "#REDIRECT" in txt.upper() or "#ALIH" in txt.upper():
                            elem.clear()
                            continue
                        txt = re.sub(r"\{\{.*?\}\}|\[\[File:.*?\]\]|\[\[Berkas:.*?\]\]", " ", txt, flags=re.S)
                        txt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", txt)
                        txt = re.sub(r"['=*#<>]", " ", txt)
                        for part in re.split(r"(?<=[.])\s+", txt):
                            if wiki_added >= max_wiki_sentences:
                                break
                            before = len(sentences)
                            add_pair(part, "", "jvwiki")
                            if len(sentences) > before:
                                wiki_added += 1
                    elem.clear()
                    if wiki_added >= max_wiki_sentences:
                        break
        except Exception as e:
            print(f"[WARN] Ekstraksi jvwiki berhenti sebagian ({e}); terkumpul {wiki_added} kalimat.")

    # Save outputs
    sent_path = os.path.join(OUT_DIR, "sentences_validated.jsonl")
    with open(sent_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    lex_path = os.path.join(OUT_DIR, "lexicon_expanded.json")
    with open(lex_path, "w", encoding="utf-8") as f:
        json.dump({
            "bilingual_entries": lexicon,
            "top_word_freq": dict(word_freq.most_common(15000)),
        }, f, ensure_ascii=False, indent=2)

    print(f"[corpus_builder] Tersimpan {len(sentences)} kalimat Jawa tervalidasi & {len(lexicon)} entri kamus (+{len(word_freq)} kosakata unik).")


if __name__ == "__main__":
    build_corpus_and_lexicon()
