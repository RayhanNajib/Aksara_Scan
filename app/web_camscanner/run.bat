@echo off
rem ============================================================
rem  AksaraScan - Jalankan Lokal (run.bat)
rem  Serve web_camscanner via HTTP + buka browser otomatis.
rem  HTTP dipakai agar Service Worker (PWA offline) bisa aktif;
rem  tanpa HTTP pun tetap jalan via file:// (fallback model.js).
rem ============================================================
setlocal
cd /d "%~dp0"
set PORT=8899

where python >nul 2>nul
if %errorlevel%==0 (
  set PY=python
) else (
  where py >nul 2>nul
  if %errorlevel%==0 (
    set PY=py
  ) else (
    echo [!] Python tidak ditemukan. Membuka langsung via file://
    start "" "%~dp0index.html"
    goto :eof
  )
)

echo [*] Menjalankan AksaraScan di http://127.0.0.1:%PORT%
echo [*] Tutup jendela ini untuk menghentikan server.
start "" /min cmd /c "timeout /t 2 >nul & start http://127.0.0.1:%PORT%/index.html"
"%PY%" -m http.server %PORT% --bind 127.0.0.1
