---
name: "AksaraScan — Manuskrip Modern v4.0"
version: "4.0.0"
description: "Split workstation Operate UI & editorial Read documentation in Malam Keraton Dark & Kertas Dluwang Light."
inspiration: "ScanRead workflow clarity · Linear 1px hairline surfaces · Javanese Dluwang & Wulung manuscript palette"
colors:
  dark:
    bg_primary: "#0e1422"
    bg_surface: "#151e31"
    bg_panel: "#1b263d"
    bg_panel_hover: "#22304d"
    bg_sunken: "#0a0f1a"
    text_main: "#f4ead5"
    text_muted: "#d2c4ab"
    text_sub: "#afa188"
    accent: "#d4a24c"
    accent_text: "#ecc574"
    accent_btn_text: "#161007"
    nav_active_bg: "#243354"
    nav_active_text: "#f4ead5"
    status_ok: "#7ec487"
    status_warn: "#f4927e"
  light:
    bg_primary: "#f6efe0"
    bg_surface: "#ffffff"
    bg_panel: "#faf6ec"
    bg_panel_hover: "#f2e9d5"
    bg_sunken: "#ebe1cb"
    text_main: "#24160e"
    text_muted: "#4d3728"
    text_sub: "#654d3c"
    accent: "#c89236"
    accent_text: "#804f07"
    accent_btn_text: "#1a1106"
    nav_active_bg: "#1f2a44"
    nav_active_text: "#f4ead5"
    status_ok: "#236133"
    status_warn: "#a8311b"
typography:
  display:
    fontFamily: "'Newsreader', 'Fraunces', Georgia, serif"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.02em"
  ui:
    fontFamily: "'Plus Jakarta Sans', system-ui, -apple-system, sans-serif"
    fontSize: "15px"
    lineHeight: 1.55
    measureMax: "68ch"
  script:
    fontFamily: "'Noto Sans Javanese', serif"
    fontSize: "clamp(26px, 3.2vw, 36px)"
    lineHeight: 1.75
  telemetry:
    fontFamily: "'JetBrains Mono', ui-monospace, 'Cascadia Mono', monospace"
    fontVariantNumeric: "tabular-nums"
rounded:
  sm: "6px"
  md: "10px"
  lg: "14px"
  pill: "999px"
spacing:
  step_4: "4px"
  step_8: "8px"
  step_12: "12px"
  step_16: "16px"
  step_24: "24px"
  step_32: "32px"
  step_48: "48px"
components:
  topbar:
    height: "64px"
    maxWidth: "1280px"
    backdropFilter: "blur(10px)"
  button-primary:
    backgroundColor: "{colors.dark.accent}"
    textColor: "{colors.dark.accent_btn_text}"
    rounded: "{rounded.pill}"
---

## Overview

**AksaraScan v4.0** menggunakan mode **Operate** pada halaman utama (*Pemindai*, *Batch*, *Lab & Referensi*) untuk ketepatan deteksi dan kejelasan alur kerja, serta mode **Read** pada tab *Tentang* untuk dokumentasi riset dan atribusi lisensi. Seluruh antarmuka mempertahankan identitas visual manuskrip Jawa (**Malam Keraton** `#0e1422` pada mode gelap dan **Kertas Dluwang** `#f6efe0` pada mode terang) tanpa gradien dekoratif, kartu bersarang, atau label *kicker* kapital.

---

## Colors & WCAG AA Contrast Audit

Seluruh pasangan teks dan latar belakang di bawah ini telah dihitung secara matematis dengan rumus luminans relatif WCAG 2.1 dan semuanya melampaui batas minimum **4.5:1**.

### Mode Gelap — Malam Keraton (`data-theme="dark"`)

| Peran Token | Foreground | Background | Rasio Kontras | Status |
|---|---|---|---|---|
| Teks Utama (`--text-main` di `--bg-primary`) | `#f4ead5` | `#0e1422` | `15.39:1` | Lulus AAA |
| Teks Utama di Permukaan (`--bg-surface`) | `#f4ead5` | `#151e31` | `13.93:1` | Lulus AAA |
| Teks Utama di Panel (`--bg-panel`) | `#f4ead5` | `#1b263d` | `12.63:1` | Lulus AAA |
| Teks Sekunder (`--text-muted` di `--bg-surface`) | `#d2c4ab` | `#151e31` | `9.69:1` | Lulus AAA |
| Teks Sekunder (`--text-muted` di `--bg-panel`) | `#d2c4ab` | `#1b263d` | `8.79:1` | Lulus AAA |
| Teks Sub/Placeholder (`--text-sub` di `--bg-panel`) | `#afa188` | `#1b263d` | `5.95:1` | Lulus AA |
| Teks Sub/Placeholder (`--text-sub` di `--bg-sunken`) | `#afa188` | `#0a0f1a` | `7.55:1` | Lulus AAA |
| Aksen Teks (`--accent-text` di `--bg-surface`) | `#ecc574` | `#151e31` | `10.15:1` | Lulus AAA |
| Tab Aktif (`--nav-active-text` di `--nav-active-bg`) | `#f4ead5` | `#243354` | `10.48:1` | Lulus AAA |
| Tombol Aksen (`--btn-accent-text` di `--btn-accent-bg`) | `#161007` | `#d4a24c` | `8.16:1` | Lulus AAA |
| Status Baik (`--lontar` di `--bg-surface`) | `#7ec487` | `#151e31` | `8.03:1` | Lulus AAA |
| Status Peringatan (`--bata` di `--bg-surface`) | `#f4927e` | `#151e31` | `7.34:1` | Lulus AAA |

### Mode Terang — Kertas Dluwang (`data-theme="light"`)

| Peran Token | Foreground | Background | Rasio Kontras | Status |
|---|---|---|---|---|
| Teks Utama (`--text-main` di `--bg-primary`) | `#24160e` | `#f6efe0` | `15.33:1` | Lulus AAA |
| Teks Utama di Permukaan (`--bg-surface`) | `#24160e` | `#ffffff` | `17.56:1` | Lulus AAA |
| Teks Utama di Panel (`--bg-panel`) | `#24160e` | `#faf6ec` | `16.27:1` | Lulus AAA |
| Teks Sekunder (`--text-muted` di `--bg-surface`) | `#4d3728` | `#ffffff` | `11.10:1` | Lulus AAA |
| Teks Sekunder (`--text-muted` di `--bg-panel`) | `#4d3728` | `#faf6ec` | `10.28:1` | Lulus AAA |
| Teks Sub/Placeholder (`--text-sub` di `--bg-sunken`) | `#654d3c` | `#ebe1cb` | `6.03:1` | Lulus AA |
| Teks Sub/Placeholder (`--text-sub` di `--bg-panel`) | `#654d3c` | `#faf6ec` | `7.26:1` | Lulus AAA |
| Aksen Teks (`--accent-text` di `--bg-surface`) | `#804f07` | `#ffffff` | `6.91:1` | Lulus AA+ |
| Tab Aktif (`--nav-active-text` di `--nav-active-bg`) | `#f4ead5` | `#1f2a44` | `11.93:1` | Lulus AAA |
| Tombol Aksen (`--btn-accent-text` di `--btn-accent-bg`) | `#1a1106` | `#c89236` | `6.76:1` | Lulus AA+ |
| Status Baik (`--lontar` di `--bg-surface`) | `#236133` | `#ffffff` | `7.42:1` | Lulus AAA |
| Status Peringatan (`--bata` di `--bg-surface`) | `#a8311b` | `#ffffff` | `6.73:1` | Lulus AA+ |

---

## Typography

Seluruh font primer (**Newsreader**, **Plus Jakarta Sans**, dan **Noto Sans Javanese Bold/Regular**, seluruhnya berlisensi **SIL OFL 1.1**) di-*self-host* secara lokal di `app/web_camscanner/vendor/fonts/` melalui `@font-face` sehingga merender sempurna tanpa permintaan CDN eksternal ataupun saat offline.

- **Aksen Judul & Wordmark (`Newsreader`)**: italic ringan pada merek dan judul seksi, pelacakan rapat `-0.02em`, tanpa efek teks gradasi.
- **UI Fungsional & Paragraf (`Plus Jakarta Sans`)**: panjang baris baca dibatasi `65–72ch`; label kontrol ditulis dalam *sentence case* yang langsung menyebut aksi nyata.
- **Glif Aksara Jawa (`Noto Sans Javanese`)**: eksklusif pada area Unicode Aksara, kartu kamus, bank prototipe, dan overlay inspeksi.
- **Telemetri & Skor (`JetBrains Mono`, `tabular-nums`)**: dipakai terbatas untuk dimensi px, latensi ms, skor kemiripan, nilai ambang Otsu, dan metrik deteksi.

---

## Layout & Progressive Disclosure

1. **Topbar Sticky (`64px`, `max-width: 1280px`)**: kiri merek (`ꦲ AksaraScan`), tengah 4 navigasi sejajar (`Pemindai`, `Batch`, `Lab & Referensi`, `Tentang`), kanan status mesin ringkas, tombol pill `Pindai Baru`, dan pengalih tema Terang/Gelap.
2. **Tahap Unggah Bebas Friksi**: sebelum ada gambar, hanya hero ringkas, dropzone luas (`PNG · JPG · WebP · Foto Buram · Tinta Warna · Mode Gelap`), 4 contoh 1-klik, riwayat pindai lokal (bila ada), dan alur 3 langkah nyata (`Unggah → Tajamkan & Deteksi → Baca & Koreksi`). Tidak ada panel pengaturan atau kartu hasil kosong sebelum foto masuk.
3. **Split Workbench Asimetris (`52% / 48%` Desktop, Tunggal pada Mobile `<= 920px`)**: setelah gambar dimuat, kanvas bersih (tanpa ornamen sudut palsu) di kiri dilengkapi micro-toolbar horizontal (*Kotak*, *Bandingkan Asli/Tajam* Before/After, *Biner*, *Putar 90°*, *Luruskan*, *Zoom*) + pengendali pra-pindai (*Auto-Sharpen Blur*, *Hapus Bayangan Kertas*, *Hapus Garis Buku*, penggeser ketajaman & kontras); kolom kanan menampilkan Aksara Jawa Unicode, Bacaan Latin, Arti Indonesia, dan glif interaktif.
4. **Inspektor Referensi & Koreksi 1-Klik**: klik kotak di kanvas, kartu glif, atau kata bergaris putus membuka laci inspektor dengan potongan gambar pengguna, Top-3 kandidat Stage-2, thumbnail referensi kanonik vs tulisan tangan dari `reference_prototypes.json`, skor kemiripan kosinus, serta pencatatan koreksi lokal.
5. **Footer Tunggal 1 Baris**: hak cipta singkat dan tautan cepat ke `Tentang` tanpa daftar NIM bertele-tele.

---

## Governance & Anti-Slop Checklist

- Bebas label *kicker/eyebrow* berhuruf kapital di atas judul utama.
- Bebas kartu bersarang (`card` di dalam `card`) atau batas tepi kiri berwarna tebal `> 1px`.
- Bebas ornamen sudut palsu (`.wayang-corner`), teks gradasi (`background-clip: text`), glow neon dekoratif, penomoran artifisial, maupun ikon emoji pengganti SVG.
- Permukaan browser (`::selection`, scrollbar, `:focus-visible`, `caret-color`, `accent-color`) sepenuhnya bertema sesuai mode Terang/Gelap.
- Responsif bersih hingga `390 px` dengan **0 scroll horizontal** dan transisi tunggal terarah yang menghormati `prefers-reduced-motion`.
