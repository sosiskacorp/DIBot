@echo off
chcp 65001 >nul
title 🎓 DIBot - Учебный ИИ-ассистент НГТУ
cd /d "%~dp0"

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo ======================================================================
    echo   [ОШИБКА] Python не найден в системе!
    echo   Установите Python 3.10+ с официального сайта python.org
    echo   Обязательно отметьте галочку: "Add Python to PATH" при установке.
    echo ======================================================================
    echo.
    pause
    exit /b 1
)

python run.py %*
if %ERRORLEVEL% neq 0 (
    if %ERRORLEVEL% neq 130 (
        echo.
        echo [Сессия завершена с кодом %ERRORLEVEL%]
        pause
    )
)
