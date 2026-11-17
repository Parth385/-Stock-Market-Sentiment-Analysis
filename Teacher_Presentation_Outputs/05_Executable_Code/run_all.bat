@echo off
cls
echo ================================================================================
echo   BEYOND SENTIMENT: EXECUTING LIVE RESEARCH & PRESENTATION CODE DEMO
echo ================================================================================
echo.
python "%~dp0live_runners\run_all_presentation_demos.py"
echo.
echo ================================================================================
echo   DEMO EXECUTION FINISHED. Press any key to exit...
echo ================================================================================
pause >nul
