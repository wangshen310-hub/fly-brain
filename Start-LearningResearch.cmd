@echo off
powershell.exe -NoProfile -File "%~dp0Start-LearningResearch.ps1" %*
if errorlevel 1 pause
