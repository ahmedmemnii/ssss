@echo off
REM Navigate to project directory
cd /d "C:\Users\Have Fun\Desktop\New folder (5)\ids-intelligent"

echo ============================================================
echo IDS Intelligent - Quick Start
echo ============================================================
echo.

echo Current directory: %CD%
echo.

:main
REM Option menu
echo What would you like to do?
echo.
echo [1] Train a Random Forest model (Quick - 5 min)
echo [2] Train all models (20-30 min)
echo [3] Launch Streamlit dashboard
echo [4] Just verify the setup
echo [5] Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto train_rf
if "%choice%"=="2" goto train_all
if "%choice%"=="3" goto dashboard
if "%choice%"=="4" goto verify
if "%choice%"=="5" goto end

:train_rf
echo.
echo Training Random Forest model...
"C:\Users\Have Fun\Desktop\New folder (5)\.venv\Scripts\python.exe" -m src.ids.train --dataset kddcup99 --models rf --save
goto show_results

:train_all
echo.
echo Training all 6 models (this will take 20-30 minutes)...
"C:\Users\Have Fun\Desktop\New folder (5)\.venv\Scripts\python.exe" -m src.ids.train --dataset kddcup99 --models rf svm knn iso kmeans ae --save
goto show_results

:dashboard
echo.
echo Launching Streamlit dashboard...
echo (This will open in your web browser at http://localhost:8501)
echo.
"C:\Users\Have Fun\Desktop\New folder (5)\.venv\Scripts\streamlit.exe" run app\app.py
goto end

:verify
echo.
echo Verifying installation...
"C:\Users\Have Fun\Desktop\New folder (5)\.venv\Scripts\python.exe" -c "import pandas, sklearn, torch, streamlit; print('✓ All dependencies installed correctly')"
echo.
echo Project structure:
dir /b
echo.
goto ask_again

:show_results
echo.
echo ============================================================
echo ✓ Training Complete!
echo ============================================================
echo.
if exist models\performance.csv (
    echo Performance Results:
    type models\performance.csv
    echo.
)
echo Models saved in: models\
echo.
goto ask_again

:ask_again
echo.
set /p again="Would you like to do something else? (y/n): "
if /i "%again%"=="y" goto menu_again
goto end

:menu_again
cls
goto main

:end
echo.
echo Thank you for using IDS Intelligent!
pause
