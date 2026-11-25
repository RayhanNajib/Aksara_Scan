---
name: "AksaraScan — Manuskrip Modern"
version: "3.0.0"
inspiration: "Linear hairline surfaces · Vercel Geist clarity · MonoLens split-workstation · Javanese Dluwang & Wulung heritage"
---

# AksaraScan Design System (`DESIGN.md`)

## 1. Filosofi Visual
Antarmuka memadukan ketelitian ruang kerja komputer vision modern (*split-pane workstation*, *hairline borders*, *monospaced telemetry*) dengan karakter palet manuskrip Jawa (**Malam Keraton** untuk mode gelap, **Kertas Dluwang** untuk mode terang). Tidak ada gradien neon, kartu berlebihan, atau ornamen dekoratif yang menghalangi keterbacaan.

## 2. Token Warna & Kontras Tema (WCAG AA Terverifikasi)

### Mode Gelap — Malam Keraton (`data-theme="dark"`, Default)
- `--bg-canvas`: `#0e1422` (Kanvas utama)
- `--bg-surface`: `#161f33` (Panel & kartu utama)
- `--bg-elevated`: `#1d2842` (Sub-kartu hasil & kontrol)
- `--bg-sunken`: `#0b101b` (Stage kanvas & track input)
- `--text-primary`: `#f4ead5` (*Paper*, kontras 13.8:1 vs `--bg-surface`)
- `--text-secondary`: `#c5b8a2`
- `--text-muted`: `#8f8472`
- `--accent-text`: `#ecc574` (*Prada light*, kontras > 9.5:1 di latar gelap)
- `--accent-gold`: `#c8963e` (*Prada stroke/border*)
- `--status-ok-text`: `#7ec487`
- `--status-warn-text`: `#f08069`

### Mode Terang — Kertas Dluwang (`data-theme="light"`)
- `--bg-canvas`: `#f6efe0` (Kertas dluwang hangat)
- `--bg-surface`: `#ffffff` (Panel kerja bersih)
- `--bg-elevated`: `#f9f5eb` (Sub-kartu terangkat)
- `--bg-sunken`: `#ebe1cb` (Kontrol latar tenggelam)
- `--text-primary`: `#261810` (*Sogan pekat*, kontras 15.2:1 vs putih)
- `--text-secondary`: `#5b4333` (Kontras 7.9:1 vs putih)
- `--text-muted`: `#785f4e`
- `--accent-text`: `#8f5c10` (Perunggu gelap WCAG AA >= 5.5:1 — mencegah teks emas pudar di mode terang)
- `--accent-gold`: `#c8963e` (Garis aksen & fokus)
- `--nav-active-bg`: `#1f2a44` / `--nav-active-text`: `#f4ead5` (Menjamin keterbacaan konsisten saat bertukar tema)
- `--status-ok-text`: `#2d6a3e`
- `--status-warn-text`: `#b53d24`

## 3. Tipografi
- **Judul Editorial**: `Fraunces` (`600, 700`) — berwibawa tanpa kesesakan
- **UI & Paragraf**: `Plus Jakarta Sans` (`400, 500, 600, 700`)
- **Glif Aksara Jawa**: `Noto Sans Javanese` (`500, 700`, `line-height: 1.75`)
- **Telemetri & Lab Citra**: `JetBrains Mono` (`400, 500, 600`)

## 4. Arsitektur Navigasi & Tata Letak
1. **Topbar Tunggal Sticky (64px)**: Brand mark + 4 tab navigasi sejajar (`Pemindai`, `Lab Citra`, `Kamus`, `Tentang`) + lencana status mesin + pengalih tema Terang/Gelap yang sinkron penuh.
2. **Pemindai (Split-Screen Workspace)**: Kolom kiri untuk masukan citra (drag-drop, kamera, paste, 1-klik sampel, pilihan mesin, parameter deteksi lipat); kolom kanan terfokus langsung pada Unicode Jawa, bacaan Latin (koreksi klik kata OOV), arti bahasa Indonesia, dan galeri glif.
3. **Footer Minimalis**: Satu baris ringkas berisi hak cipta dan tautan cepat ke tab Tentang & Tim.
