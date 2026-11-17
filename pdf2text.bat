@echo off
chcp 65001 >nul
setlocal EnableExtensions
rem Перетащите один или несколько PDF на этот файл.
rem Рядом с каждым PDF появятся:
rem   «<имя> — разметка.txt»  — текст в формате VoxCPM Books (можно править и загружать в проект)
rem   «<имя> — озвучка.txt»   — как текст прочитает модель: фрагменты и паузы
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Сначала запустите install.bat
    pause
    exit /b 1
)
if "%~1"=="" (
    echo Перетащите PDF-файл на pdf2text.bat
    pause
    exit /b 1
)
call "%~dp0installer\env.bat"
pushd "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m audiobook pdf2text %*
popd
pause
