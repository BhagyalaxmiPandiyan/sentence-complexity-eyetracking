@echo off
rem One-time installation of the reading study software.
cd /d "%~dp0"
echo Installing the reading study software.
echo This needs internet and takes about 5 minutes.
echo.

set PY=
py -3.12 --version >nul 2>&1
if not errorlevel 1 set PY=py -3.12
if defined PY goto havepy
python --version 2>nul | findstr /C:"Python 3.12" >nul
if not errorlevel 1 set PY=python
if defined PY goto havepy
rem The python.org installer puts Python here when PATH was not ticked.
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto havepy
if exist "%ProgramFiles%\Python312\python.exe" set PY="%ProgramFiles%\Python312\python.exe"
if defined PY goto havepy

echo Python 3.12 was not found.
echo 1. Download it from https://www.python.org/downloads/release/python-31210/
echo    (choose "Windows installer (64-bit)")
echo 2. During installation tick "Add python.exe to PATH"
echo 3. Run setup.bat again.
pause
exit /b 1

:havepy
if not exist .venv\Scripts\python.exe (
    %PY% -m venv .venv
    if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
.venv\Scripts\python.exe check_setup.py
pause
exit /b 0

:failed
echo.
echo Installation failed. Take a photo of this window and send it to the researcher.
pause
exit /b 1
