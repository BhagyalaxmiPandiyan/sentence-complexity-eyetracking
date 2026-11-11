@echo off
rem One complete participant session: calibration, reading task, results file.
cd /d "%~dp0"
rem Reading task: natural_reading.py (whole sentence, word measures)
rem or hybrid_spr.py (part-by-part self-paced reading).
set TASK=natural_reading.py
if not exist .venv\Scripts\python.exe (
    echo The software is not installed yet. Double-click setup.bat first.
    pause
    exit /b 1
)

echo ============================================================
echo  Tamil Reading Study
echo ============================================================
echo.
set PID=
set /p PID=Participant code from the researcher (e.g. P07):
if "%PID%"=="" (
    echo A participant code is required.
    pause
    exit /b 1
)
set LIST=
set /p LIST=List number from the researcher (0 or 1):
if "%LIST%"=="" set LIST=0

echo.
echo STEP 1 of 2: Calibration (about 3 minutes)
echo  - Sit about an arm's length from the screen, face well lit.
echo  - Rest your chin on your hands and keep your head still.
echo  - Look at each dot, press SPACE, keep looking until it moves on.
echo  - At the end a red dot follows your gaze. Press Q to continue.
echo.
pause
.venv\Scripts\python.exe gaze_calibration.py --label %PID%
if errorlevel 1 goto failed

echo.
echo STEP 2 of 2: Reading task (100 sentences, about 20 minutes)
echo  - Keep your head still, as in calibration.
echo  - The whole sentence appears at once (1 to 4 lines). Read it
echo    normally, then press ENTER.
echo  - There are 3 short breaks. Rest, then sit as before.
echo  - Answer questions with Y (yes) or N (no).
echo.
pause
.venv\Scripts\python.exe %TASK% --participant %PID% --list %LIST%
if errorlevel 1 goto failed

rem Word measures (FFD/FPD/RPD/TD) for the newest session; a failure
rem here does not stop the results file being made.
if not "%TASK%"=="natural_reading.py" goto package
set LAST=
for /f "delims=" %%d in ('dir /b /ad /o:d "data\%PID%_*" 2^>nul') do set LAST=data\%%d
if defined LAST .venv\Scripts\python.exe analyze_reading.py "%LAST%"

:package
echo.
.venv\Scripts\python.exe package_results.py %PID%
pause
exit /b 0

:failed
echo.
echo The session did not finish. Take a photo of this window and send it
echo to the researcher. Any data recorded so far is kept.
.venv\Scripts\python.exe package_results.py %PID%
pause
exit /b 1
