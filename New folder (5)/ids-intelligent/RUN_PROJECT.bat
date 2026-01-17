@echo off
REM ============================================================
REM IDS Intelligent - Complete Demo Runner
REM ============================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo IDS INTELLIGENT - Training Demo
echo ============================================================
echo.

echo [Step 1/3] Training Random Forest model...
echo This will take 2-5 minutes...
echo.

.venv\Scripts\python.exe -m src.ids.train --dataset kddcup99 --models rf --save

if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    echo.
    pause
    exit /b 1
)

echo.
echo ✓ Training complete!
echo.
echo [Step 2/3] Performance results saved to: models\performance.csv
echo.

if exist models\performance.csv (
    echo Performance Summary:
    echo -------------------
    type models\performance.csv
    echo.
)

echo [Step 3/3] To launch the dashboard, run:
echo    streamlit run app/app.py
echo.

echo ============================================================
echo ✓ Demo Complete!
echo ============================================================
echo.
echo Next Steps:
echo   1. Launch dashboard: streamlit run app/app.py
echo   2. View LaTeX report: reports\report.tex
echo   3. Read documentation: README.md
echo.
pause
