@echo off
rem ===========================================================================
rem  Double-click to start the webcam eye-tracking DEMO (3 Tamil sentences).
rem  Opens Chrome at http://localhost:8000/?demo=1 . Needs internet (the
rem  WebGazer eye-tracking library loads from a CDN) and a webcam.
rem  Demo mode saves and uploads nothing. Close this window to stop.
rem ===========================================================================
cd /d "%~dp0Tool\TamilReadingWeb"
echo.
echo  Webcam eye-tracking demo: http://localhost:8000/?demo=1
echo  1. Allow the camera.  2. Calibrate (click each red dot 3 times while looking at it).
echo  3. Read 3 sentences: SPACE on the dot, read, ENTER.  4. See FFD / FPD / RPD / TD per word.
echo  Keep this window open. Close it to stop.
echo.
start "" cmd /c "timeout /t 2 /nobreak >nul & start chrome http://localhost:8000/?demo=1"
C:\Python313\python.exe -m http.server 8000
