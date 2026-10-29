@echo off
chcp 65001 >nul
title Создание ярлыка DIBot на Рабочем столе
cd /d "%~dp0"
python run.py --shortcut
echo.
pause
