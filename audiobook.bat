@echo off
chcp 65001 >nul
setlocal EnableExtensions
rem Командная строка VoxCPM Books. Примеры:
rem   audiobook.bat make "D:\Книги\книга.pdf" --ref "D:\голос.wav" --ref-text "Текст образца." --formats mp3,m4b
rem   audiobook.bat synth книга
rem   audiobook.bat export книга --formats m4b
rem   audiobook.bat status книга
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Сначала запустите install.bat
    exit /b 1
)
call "%~dp0installer\env.bat"
pushd "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m audiobook %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
