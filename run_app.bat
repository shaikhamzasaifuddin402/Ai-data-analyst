@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem === Data Scientist Workspace Launcher (Auto Port) ===
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ==============================================
echo   Data Scientist Workspace - Launcher (Windows)
echo ==============================================

echo [INFO] Checking Python availability...
set "PY_EXE="
where py >nul 2>nul && set "PY_EXE=py -3"
if "%PY_EXE%"=="" where python >nul 2>nul && set "PY_EXE=python"
if "%PY_EXE%"=="" (
  echo [ERROR] Python 3.8+ is required but not found in PATH.
  echo        Install Python from https://www.python.org/downloads/
  pause
  exit /b 1
)

%PY_EXE% -c "import sys; exit(0 if sys.version_info[:2] >= (3,8) else 1)"
if errorlevel 1 goto :badpython

echo [INFO] Preparing virtual environment...
set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [INFO] Creating venv at %VENV_DIR% ...
  %PY_EXE% -m venv "%VENV_DIR%"
)

echo [INFO] Activating virtual environment...
if exist "%VENV_DIR%\Scripts\activate.bat" (
  call "%VENV_DIR%\Scripts\activate.bat"
) else (
  call "%VENV_DIR%\Scripts\activate"
)

rem Keep console unicode-safe
set "PYTHONUTF8=1"
rem Avoid Transformers importing TensorFlow submodules (Keras 3 conflicts)
set "TRANSFORMERS_NO_TF=1"

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

set "REQ_FILE=%SCRIPT_DIR%requirements.txt"
if exist "%REQ_FILE%" (
  echo [INFO] Installing dependencies from requirements.txt ...
  python -m pip install -r "%REQ_FILE%"
) else (
  echo [WARN] requirements.txt not found. Installing core packages only...
  python -m pip install streamlit pandas numpy scikit-learn plotly
)

rem === Find a free port among a list ===
set "PORT="
set "PORT_CANDIDATES=8501 8502 8503 8504 8505 8506 8507 8508 8509 8510"
for %%P in (%PORT_CANDIDATES%) do (
  rem netstat returns 0 if a match is found (port in use), 1 if not found
  netstat -ano | findstr :%%P >nul
  if errorlevel 1 (
    set "PORT=%%P"
    goto :foundport
  )
)
:foundport
if "%PORT%"=="" set "PORT=8503"

echo [INFO] Starting Streamlit app on port %PORT% ...
python -m streamlit run "%SCRIPT_DIR%app.py" --server.port=%PORT% --server.maxUploadSize=200

rem Optionally open in default browser (comment out to disable)
start "" "http://localhost:%PORT%/"

goto :end

:badpython
echo [ERROR] Detected Python below 3.8. Please install Python 3.8 or newer.
pause
exit /b 1

:end
endlocal