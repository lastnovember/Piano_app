@echo off
cd /d "%~dp0"
py -3.12 midi_piano.py %*
pause
