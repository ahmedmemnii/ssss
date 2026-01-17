@echo off
echo ============================================================
echo IDS Intelligent - Quick Verification
echo ============================================================
echo.

cd /d "%~dp0"
set PYTHON=.venv\Scripts\python.exe

echo [1/3] Checking Python environment...
%PYTHON% --version
if errorlevel 1 (
    echo ERROR: Python not found
    exit /b 1
)
echo.

echo [2/3] Verifying dependencies...
%PYTHON% -c "import pandas, sklearn, torch, streamlit; print('✓ All dependencies OK')"
if errorlevel 1 (
    echo ERROR: Some dependencies missing
    exit /b 1
)
echo.

echo [3/3] Project structure verified:
echo ✓ src/ids/ - Core IDS modules
echo ✓ app/app.py - Streamlit dashboard
echo ✓ reports/report.tex - LaTeX report
echo ✓ notebooks/ - Analysis notebooks
echo ✓ models/ - Model storage directory
echo.

echo ============================================================
echo ✓ IDS System Ready!
echo ============================================================
echo.
echo Next Steps:
echo   1. Train models: python -m src.ids.train --models rf svm --save
echo   2. Launch dashboard: streamlit run app/app.py
echo   3. View report: compile reports/report.tex
echo.
pause
