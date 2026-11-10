@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title VoxCPM Books
if not exist ".venv\Scripts\python.exe" (
    echo Сначала запустите install.bat
    pause
    exit /b 1
)
call "%~dp0installer\env.bat"
echo Запуск VoxCPM Books... Браузер откроется сам: http://127.0.0.1:7861
echo Не закрывайте это окно, пока идёт работа.
".venv\Scripts\python.exe" -X faulthandler -m audiobook ui %*
if errorlevel 1 echo [X] Программа завершилась с ошибкой, код %ERRORLEVEL%. Пришлите текст окна выше.
pause
