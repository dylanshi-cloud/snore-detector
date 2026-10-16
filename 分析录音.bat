@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "F=%~1"
if "%F%"=="" (
  echo ============================================================
  echo   打鼾检测 - 分析录音
  echo ============================================================
  echo   下次可以直接把录音文件拖到这个图标上, 一步到位。
  echo   ^(支持 m4a / mp3 / wav / aac, 不用手动转格式^)
  echo.
  set /p "F=现在把文件路径粘贴到这里, 然后回车: "
)
if "%F%"=="" (
  echo 没给文件, 退出。
  pause
  exit /b 1
)
echo.
echo 正在分析: %F%
echo.
python snore_detect.py --replay "%F%"
echo.
pause
