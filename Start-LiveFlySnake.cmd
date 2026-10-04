@echo off
powershell.exe -NoProfile -File "%~dp0Start-LiveFlySnake.ps1" %*
if errorlevel 1 pause
