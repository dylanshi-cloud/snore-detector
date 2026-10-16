@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   打鼾检测器 - 今晚开始采集
echo   睡前双击这个文件, 早上按 Ctrl+C 出报告
echo ============================================
echo.
python snore_detect.py %*
echo.
pause
