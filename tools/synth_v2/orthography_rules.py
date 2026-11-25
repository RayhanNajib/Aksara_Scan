"""
Validator Ortografi Blok Unicode Jawa (U+A980..U+A9DF) & Transliterator Dua Arah
(Latin <-> Aksara Jawa) untuk AksaraScan v3.0 (Fase 2a & Fase 4c).

Diturunkan dari standar Unicode Block Javanese (U+A980..U+A9DF) dan pedoman
Nulisa Aksara Jawa / CarakanJS:
  Urutan kanonik satu kluster aksara:
    [Base (Nglegena / Swara / Murda)]
    [+ Rekan U+A9B3]?
    [+ (Pangkon U+A9C0 + Nglegena)]*   (maks 1 pasangan aktif di ortografi umum, boleh 2 tumpuk di kata serapan; vokal induk mati)
    [+ Wyanjana (Cakra U+A9BF | Keret U+A9BD | Pengkal U+A9BE)]?
    [+ Swara (Wulu | Suku | Pepet | Taling | Taling+Tarung | Tarung)]?
    [+ Panyigeg (Cecak | Layar | Wignyan)]?
  ATAU konsonan mati di akhir kata/klausa:
    [Base + Pangkon U+A9C0] (tanpa Wyanjana/Swara/Panyigeg setelah Pangkon akhir).
"""
import re

# Blok Unicode Aksara Jawa (U+A980..U+A9DF)
PANGKON = "\uA9C0"
CEC_TELU = "\uA9B3"

LAYAR = "\uA982"
CECAK = "\uA981"
WIGNYAN = "\uA983"
PANYIGEG = {CECAK, LAYAR, WIGNYAN}

WULU = "\uA9B6"
SUKU = "\uA9B8"
PEPET = "\uA9BC"
TALING = "\uA9BA"
TARUNG = "\uA9B4"
SWARA_SIGNS = {WULU, SUKU, PEPET, TALING, TARUNG}

CAKRA = "\uA9BF"
KERET = "\uA9BD"
PENGKAL = "\uA9BE"
WYANJANA = {CAKRA, KERET, PENGKAL}

PADA_ADEG = "\uA9CB"
PADA_LINGSA = "\uA9C8"
PADA_LUNGSI = "\uA9C9"
PADA_ALL = {PADA_ADEG, PADA_LINGSA, PADA_LUNGSI, "\uA9C7", "\uA9CA", "\uA9CC", "\uA9CF"}

# 20 Nglegena inti
NGLEGENA_LIST = [
    ("ha", "\uA9B2"), ("na", "\uA9A4"), ("ca", "\uA995"), ("ra", "\uA9AB"), ("ka", "\uA98F"),
    ("da", "\uA9A2"), ("ta", "\uA9A0"), ("sa", "\uA9B1"), ("wa", "\uA9AE"), ("la", "\uA9AD"),
    ("pa", "\uA9A5"), ("dha", "\uA99D"), ("ja", "\uA997"), ("ya", "\uA9AA"), ("nya", "\uA99A"),
    ("ma", "\uA9A9"), ("ga", "\uA992"), ("ba", "\uA9A7"), ("tha", "\uA99B"), ("nga", "\uA994"),
]
LATIN_TO_NGLEGENA = {lat: ch for lat, ch in NGLEGENA_LIST}
NGLEGENA_TO_CONS = {ch: lat[:-1] if lat != "ha" else "h" for lat, ch in NGLEGENA_LIST}
NGLEGENA_CHARS = set(NGLEGENA_TO_CONS.keys())

# Pa cerek & Nga lelet (khusus e pepet pada r/l)
PA_CEREK = "\uA989"
NGA_LELET = "\uA98A"
ALL_BASE_CONSONANTS = NGLEGENA_CHARS | {
    "\uA991", "\uA993", "\uA996", "\uA998", "\uA99F", "\uA9A1", "\uA9A6", "\uA9A8", "\uA9AF",
    PA_CEREK, NGA_LELET,
}
ALL_SWARA_INDEP = {"\uA984", "\uA986", "\uA988", "\uA98C", "\uA98E"}


def validate_javanese_unicode(text, strict_55=False):
    """
    Validasi struktur ortografi string Unicode Aksara Jawa.
    Mengembalikan (is_valid: bool, errors: list[str], clusters: list[str]).
    """
    if not text or not text.strip():
        return False, ["teks kosong"], []

    s = re.sub(r"\s+", "", text)
    errors = []
    clusters = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]
        # 1. Tanda baca pada
        if ch in PADA_ALL:
            clusters.append(ch)
            i += 1
            continue

        # 2. Vokal mandiri / angka / cerek / lelet
        if ch in ALL_SWARA_INDEP or ("\uA9D0" <= ch <= "\uA9D9"):
            clusters.append(ch)
            i += 1
            continue

        if ch in (PA_CEREK, NGA_LELET):
            cl = [ch]
            i += 1
            # Pa cerek / nga lelet tidak boleh diberi sandhangan vokal lagi (sudah mengandung pepet)
            if i < n and s[i] in SWARA_SIGNS:
                errors.append(f"Pa cerek/Nga lelet tidak boleh memakai sandhangan vokal {s[i]} di indeks {i}")
            while i < n and s[i] in PANYIGEG:
                cl.append(s[i])
                i += 1
            clusters.append("".join(cl))
            continue

        if ch not in ALL_BASE_CONSONANTS:
            errors.append(f"Karakter tak dikenal / posisi yatim U+{ord(ch):04X} ({ch}) di indeks {i}")
            i += 1
            continue

        cl = [ch]
        i += 1

        # Opsional cecak telu
        if i < n and s[i] == CEC_TELU:
            cl.append(s[i])
            i += 1

        # Pasangan atau Pangkon akhir
        pasangan_count = 0
        ended_with_pangkon = False
        while i < n and s[i] == PANGKON:
            cl.append(PANGKON)
            i += 1
            if i < n and s[i] in ALL_BASE_CONSONANTS:
                pasangan_count += 1
                if pasangan_count > 2:
                    errors.append(f"Tumpukan pasangan melebihi 2 tingkat di indeks {i}")
                cl.append(s[i])
                i += 1
                if i < n and s[i] == CEC_TELU:
                    cl.append(s[i])
                    i += 1
            else:
                ended_with_pangkon = True
                break

        if ended_with_pangkon:
            # Setelah pangkon penutup tidak boleh ada wyanjana, swara, atau panyigeg
            if i < n and (s[i] in WYANJANA or s[i] in SWARA_SIGNS or s[i] in PANYIGEG):
                errors.append(f"Kombinasi terlarang setelah pangkon akhir (U+{ord(s[i]):04X}) di indeks {i}")
            clusters.append("".join(cl))
            continue

        # Wyanjana (maks 1)
        wyanjana_seen = None
        if i < n and s[i] in WYANJANA:
            wyanjana_seen = s[i]
            cl.append(wyanjana_seen)
            i += 1
            if i < n and s[i] in WYANJANA:
                errors.append(f"Duplikasi sandhangan wyanjana di indeks {i}")

        # Swara (maks 1, kecuali taling diikuti tarung)
        swara_seen = []
        while i < n and s[i] in SWARA_SIGNS:
            sw = s[i]
            swara_seen.append(sw)
            cl.append(sw)
            i += 1

        if swara_seen:
            if len(swara_seen) > 2 or (len(swara_seen) == 2 and swara_seen != [TALING, TARUNG]):
                errors.append(f"Kombinasi sandhangan swara tidak sah {swara_seen} pada kluster {''.join(cl)}")
            if wyanjana_seen == KERET:
                errors.append(f"Keret (sudah berbunyi rê) tidak boleh digabung dengan sandhangan swara {swara_seen}")
            if wyanjana_seen == CAKRA and PEPET in swara_seen:
                errors.append("Cakra + Pepet tidak sah; gunakan Keret (ꦽ)")

        # Panyigeg Wanda (maks 1: cecak / layar / wignyan)
        panyigeg_seen = []
        while i < n and s[i] in PANYIGEG:
            panyigeg_seen.append(s[i])
            cl.append(s[i])
            i += 1
        if len(panyigeg_seen) > 1:
            errors.append(f"Lebih dari satu panyigeg wanda {panyigeg_seen} pada kluster {''.join(cl)}")

        # Cek kalau masih ada tanda vokal/wyanjana tertinggal sesudah panyigeg (urutan Unicode terbalik!)
        if i < n and (s[i] in SWARA_SIGNS or s[i] in WYANJANA or s[i] == PANGKON):
            # Pangkon setelah panyigeg tidak mungkin valid untuk satu kluster tanpa base baru
            if s[i] != PANGKON:
                errors.append(f"Urutan Unicode terbalik: U+{ord(s[i]):04X} muncul setelah panyigeg di indeks {i}")

        if strict_55:
            for c_ch in cl:
                if c_ch in ALL_SWARA_INDEP or c_ch in (PA_CEREK, NGA_LELET) or c_ch == CEC_TELU:
                    errors.append(f"Karakter U+{ord(c_ch):04X} belum masuk skema 55 kelas")

        clusters.append("".join(cl))

    return (len(errors) == 0), errors, clusters


# --- Transliterator Latin -> Aksara Jawa (khusus skema 55 kelas bila strict_55=True) ---

CONS_ORDER = ["nya", "nga", "dha", "tha", "ha", "na", "ca", "ra", "ka", "da", "ta", "sa", "wa", "la", "pa", "ja", "ya", "ma", "ga", "ba"]
CONS_MAP = {
    "h": "ha", "n": "na", "c": "ca", "r": "ra", "k": "ka",
    "d": "da", "t": "ta", "s": "sa", "w": "wa", "l": "la",
    "p": "pa", "dh": "dha", "j": "ja", "y": "ya", "ny": "nya",
    "m": "ma", "g": "ga", "b": "ba", "th": "tha", "ng": "nga",
}
LATIN_CONSONANTS_RE = r"(?:dh|th|ny|ng|[hncrkdtswlpjymgb])"


def _norm_jv_latin(word):
    w = word.lower().strip()
    w = w.replace("v", "p").replace("f", "p").replace("z", "j").replace("q", "k").replace("x", "ks")
    return w


def latin_to_javanese_55(sentence, add_adeg=True):
    """
    Konversi kalimat Latin Jawa menjadi Aksara Jawa Unicode kanonik yang hanya memakai
    55 kelas atomik (sehingga 100% cocok dengan dataset latih YOLO).
    Catatan: pada mode 55 kelas murni, 'rê'/'lê' dialihkan atau dihindari bila memicu pa cerek/nga lelet,
    kecuali kita memakai ra+pepet hanya jika tidak dipaksa validasi cerek.
    """
    # Tokenize keeping punctuation (, and .)
    raw_tokens = re.findall(r"[a-zA-ZéèêëÉÈÊË]+|[,.]", sentence)
    out_uni = []
    if add_adeg and raw_tokens:
        out_uni.append(PADA_ADEG)

    # Parse seluruh klausa tanpa spasi (scriptio continua) di antara tanda baca
    clauses = []
    buf = []
    for tok in raw_tokens:
        if tok in (",", "."):
            if buf:
                clauses.append(("text", "".join(buf)))
                buf = []
            clauses.append(("punct", tok))
        else:
            buf.append(_norm_jv_latin(tok))
    if buf:
        clauses.append(("text", "".join(buf)))

    for kind, val in clauses:
        if kind == "punct":
            out_uni.append(PADA_LINGSA if val == "," else PADA_LUNGSI)
            continue

        s = val
        i = 0
        n = len(s)
        prev_was_dead_cons = False  # Untuk memicu pasangan!

        def match_cons(pos):
            for c2 in ("dh", "th", "ny", "ng"):
                if s.startswith(c2, pos):
                    return c2, 2
            if pos < n and s[pos] in CONS_MAP:
                return s[pos], 1
            return None, 0

        while i < n:
            c, clen = match_cons(i)
            # Awal vokal (mis. 'aku', 'ing', 'aja') -> pakai nglegena 'ha'
            if c is None:
                if s[i] in "aiueoéèê":
                    base_name = "ha"
                    c = "h"
                    # Tidak maju i, huruf ini adalah vokalnya
                else:
                    i += 1
                    continue
            else:
                base_name = CONS_MAP[c]
                i += clen

            base_ch = LATIN_TO_NGLEGENA[base_name]

            # Cek apakah konsonan diikuti gugus wyanjana (ra/re/ya) sebelum vokal, mis. 'krama' (ka+cakra+ma)
            medial = ""
            c_next, c_nlen = match_cons(i)
            # Hanya jadikan panyigeg/wyanjana r/y jika setelahnya vokal!
            if c_next in ("r", "y") and (i + c_nlen) < n and s[i + c_nlen] in "aiueoéèê":
                if c_next == "r":
                    if s[i + c_nlen] in ("ê", "e") and s[i + c_nlen] == "ê":
                        medial = KERET
                        i += c_nlen + 1  # Keret sudah membawa vokal ê!
                    else:
                        medial = CAKRA
                        i += c_nlen
                else:
                    medial = PENGKAL
                    i += c_nlen

            # Jika tidak ada vokal sesudah konsonan (dan belum keret), cek apakah konsonan mati (sigeg)
            if medial == KERET:
                vowel_sign = ""
            elif i < n and s[i] in "aiueoéèê":
                v = s[i]
                i += 1
                if v == "a":
                    vowel_sign = ""
                elif v == "i":
                    vowel_sign = WULU
                elif v == "u":
                    vowel_sign = SUKU
                elif v in ("é", "è"):
                    vowel_sign = TALING
                elif v == "ê":
                    vowel_sign = PEPET
                elif v == "e":
                    # Heuristik: e tanpa diakritik -> pepet bila suku tertutup/tengah, atau taling? Gunakan pepet sebagai bawaan jika ambigu
                    vowel_sign = PEPET
                elif v == "o":
                    vowel_sign = TALING + TARUNG
            else:
                # Konsonan mati tanpa vokal -> pasang pangkon!
                if prev_was_dead_cons:
                    out_uni.append(PANGKON + base_ch)
                else:
                    out_uni.append(base_ch)
                # Lihat apakah karakter berikutnya masih huruf; bila masih ada konsonan selanjutnya akan memicu pasangan!
                # Bila ini di ujung klausa, langsung tambah PANGKON penutup!
                nxt_c, _ = match_cons(i)
                if i >= n or (nxt_c is None and s[i] not in "aiueoéèê"):
                    out_uni.append(PANGKON)
                    prev_was_dead_cons = False
                else:
                    prev_was_dead_cons = True
                continue

            # Tambah base (atau pasangan jika konsonan sebelumnya mati!)
            if prev_was_dead_cons:
                out_uni.append(PANGKON + base_ch)
                prev_was_dead_cons = False
            else:
                out_uni.append(base_ch)

            if medial:
                out_uni.append(medial)
            if vowel_sign:
                out_uni.append(vowel_sign)

            # Cek panyigeg wanda (cecak 'ng', layar 'r', wignyan 'h') di akhir suku kata (diikuti konsonan baru atau akhir klausa)
            # Panyigeg terpicu bila konsonan akhir diikuti akhir kata/klausa ATAU konsonan lain (bukan vokal!)
            if i < n:
                p_c, p_len = match_cons(i)
                if p_c in ("ng", "r", "h"):
                    after = i + p_len
                    # Syarat panyigeg: sesudahnya bukan vokal (atau akhir klausa)
                    if after >= n or s[after] not in "aiueoéèê":
                        if p_c == "ng":
                            out_uni.append(CECAK)
                            i += p_len
                        elif p_c == "r":
                            out_uni.append(LAYAR)
                            i += p_len
                        elif p_c == "h":
                            out_uni.append(WIGNYAN)
                            i += p_len

    return "".join(out_uni)


def javanese_to_latin_syllables(uni_str):
    """Dekoder aksara Unicode -> rangkaian suku kata Latin untuk uji dua arah."""
    ok, errs, clusters = validate_javanese_unicode(uni_str, strict_55=False)
    rev_base = {ch: lat for lat, ch in NGLEGENA_LIST}
    out = []
    for cl in clusters:
        if cl in PADA_ALL:
            if cl == PADA_LINGSA:
                out.append(",")
            elif cl == PADA_LUNGSI:
                out.append(".")
            continue
        # Parse cluster: base + optional (pangkon+base)* + wyanjana + swara + panyigeg
        idx = 0
        lat = rev_base.get(cl[0], "?")
        idx = 1
        while idx < len(cl) and cl[idx] == PANGKON:
            idx += 1
            if idx < len(cl) and cl[idx] in rev_base:
                lat = (lat[:-1] if lat.endswith("a") else lat) + rev_base[cl[idx]]
                idx += 1
            else:
                lat = lat[:-1] if lat.endswith("a") else lat
        while idx < len(cl):
            ch = cl[idx]
            strip_a = lambda x: x[:-1] if x.endswith("a") else x
            if ch == CAKRA:
                lat = strip_a(lat) + "ra"
            elif ch == KERET:
                lat = strip_a(lat) + "rê"
            elif ch == PENGKAL:
                lat = strip_a(lat) + "ya"
            elif ch == TALING and idx + 1 < len(cl) and cl[idx + 1] == TARUNG:
                lat = strip_a(lat) + "o"
                idx += 1
            elif ch == TALING:
                lat = strip_a(lat) + "é"
            elif ch == WULU:
                lat = strip_a(lat) + "i"
            elif ch == SUKU:
                lat = strip_a(lat) + "u"
            elif ch == PEPET:
                lat = strip_a(lat) + "ê"
            elif ch == CECAK:
                lat += "ng"
            elif ch == LAYAR:
                lat += "r"
            elif ch == WIGNYAN:
                lat += "h"
            idx += 1
        out.append(lat)
    return " ".join(out), ok, errs


def run_self_tests():
    # Tes Positif: urutan sah HARUS lolos
    pos_cases = [
        "꧋ꦮꦺꦢꦁꦧꦸꦧꦸꦏ꧀ꦠꦤ꧀ꦥꦒꦸꦭ꧈ꦲꦗꦔꦩꦸꦏ꧀ꦩꦸꦤ꧀ꦝꦏ꧀ꦠꦸꦮ꧉",
        "ꦲꦏ꧀ꦱꦫꦗꦮ",
        "ꦩꦔꦤ꧀ꦱꦼꦒ",
        "ꦏꦿꦩꦏꦺꦴꦛꦶ",
        "ꦱꦂꦪ",  # layar
        "ꦄꦏ꧀ꦱꦫ",
    ]
    for pc in pos_cases:
        ok, errs, cls = validate_javanese_unicode(pc)
        assert ok, f"Tes positif GAGAL untuk {pc}: {errs}"

    # Tes Negatif: urutan tak sah HARUS ditolak
    neg_cases = [
        "\uA98F\uA9B6\uA9B8",  # wulu + suku (vokal ganda)
        "\uA98F\uA9C0\uA9B6",  # pangkon akhir diikuti wulu
        "\uA98F\uA9BD\uA9B6",  # keret diikuti wulu
        "\uA98F\uA981\uA9B6",  # wulu muncul setelah cecak (urutan terbalik)
        "\uA98F\uA981\uA982",  # cecak + layar ganda
        "\uA9B6\uA98F",        # wulu yatim tanpa base di depan
    ]
    for nc in neg_cases:
        ok, errs, _ = validate_javanese_unicode(nc)
        assert not ok, f"Tes negatif GAGAL (harusnya ditolak): {repr(nc)}"

    # Uji dua arah: Latin -> Aksara -> Latin -> Aksara (roundtrip stabil!)
    rt_samples = [
        "Wédang bubuk tanpa gula, aja ngamuk mundhak tuwa.",
        "Krama kothi mangan sega.",
        "Sapa mangan tempe karo tahu ing warung.",
    ]
    for sent in rt_samples:
        u1 = latin_to_javanese_55(sent)
        ok, errs, _ = validate_javanese_unicode(u1, strict_55=True)
        assert ok, f"Hasil transliterasi tidak sah: {u1} | {errs}"
        lat1, ok2, _ = javanese_to_latin_syllables(u1)
        assert ok2
        # Normalize stripped roundtrip Latin -> u2
        lat_clean = lat1.replace(" ,", ",").replace(" .", ".").replace(" ", "")
        u2 = latin_to_javanese_55(lat_clean)
        assert u1 == u2, f"Roundtrip tidak identik:\n  u1={u1}\n  u2={u2}\n  lat1={lat1}"

    print("[PASS] Semua tes positif, tes negatif, dan uji dua arah Aksara<->Latin berhasil!")


if __name__ == "__main__":
    run_self_tests()
