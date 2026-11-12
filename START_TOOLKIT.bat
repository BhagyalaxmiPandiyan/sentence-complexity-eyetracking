@echo off
rem ===========================================================================
rem  Double-click to start the Sentence Difficulty Toolkit (live demo).
rem  It opens in your browser at http://127.0.0.1:7860
rem  Keep the black window open while you use the toolkit. Close it to stop.
rem ===========================================================================
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0toolkit"
echo.
echo  Starting the Sentence Difficulty Toolkit ...
echo  The browser opens in about 20 seconds.
echo  Wait 1-2 minutes before using Phase 2 and Phase 3 (models are loading).
echo  Keep this window open. Press Ctrl+C here to stop the toolkit.
echo.
start "" cmd /c "timeout /t 20 /nobreak >nul & start http://127.0.0.1:7860"
C:\Python313\python.exe app.py
pause
