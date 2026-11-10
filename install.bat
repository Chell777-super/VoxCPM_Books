@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title VoxCPM Books - установка

rem ============================================================
rem  Настройки (можно поменять при необходимости)
rem ============================================================
rem  Python для окружения (uv скачает его сам)
set "PY_VER=3.11"
rem  Версия PyTorch и сборка CUDA. cu126 работает с драйвером NVIDIA >= 560,
rem  cu128 - с драйвером >= 570. Для RTX 3090 подходят обе.
set "TORCH_VER=2.10.0"
set "TORCH_CUDA=cu126"
set "VENV=.venv"
set "MODEL_DIR=models\VoxCPM2"

call "%~dp0installer\env.bat"

echo.
echo  ==========================================================
echo     VoxCPM Books - установка  (uv + PyTorch %TORCH_CUDA%)
echo  ==========================================================
echo.

rem ---------------------------------------------------- 1. GPU
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [!] nvidia-smi не найден. Установите свежий драйвер NVIDIA: https://www.nvidia.com/drivers
) else (
    echo [GPU]
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)
echo.

rem ---------------------------------------------------- 2. uv
where uv >nul 2>nul
if not errorlevel 1 goto :uv_ok
if exist "%USERPROFILE%\.local\bin\uv.exe" goto :uv_path
echo [1/6] Устанавливаю uv ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
:uv_path
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo [X] Не удалось установить uv. Поставьте вручную: https://docs.astral.sh/uv/
    goto :fail
)
:uv_ok
for /f "delims=" %%i in ('where uv') do (
    set "UV_EXE=%%i"
    goto :uv_found
)
:uv_found
echo [1/6] uv: %UV_EXE%

rem ---------------------------------------------------- 3. venv
if exist "%VENV%\Scripts\python.exe" goto :venv_ok
echo [2/6] Создаю окружение Python %PY_VER% ...
uv venv "%VENV%" --python %PY_VER%
if errorlevel 1 goto :fail
:venv_ok
set "PY=%~dp0%VENV%\Scripts\python.exe"
echo [2/6] Окружение: %PY%

rem ---------------------------------------------------- 4. PyTorch CUDA
echo [3/6] PyTorch %TORCH_VER% (%TORCH_CUDA%) - около 2.5 ГБ, наберитесь терпения ...
uv pip install --python "%PY%" "torch==%TORCH_VER%" "torchaudio==%TORCH_VER%" --index-url https://download.pytorch.org/whl/%TORCH_CUDA%
if not errorlevel 1 goto :torch_ok
echo [!] torch %TORCH_VER% недоступен для %TORCH_CUDA%, ставлю последнюю версию ...
uv pip install --python "%PY%" torch torchaudio --index-url https://download.pytorch.org/whl/%TORCH_CUDA%
if errorlevel 1 goto :fail
:torch_ok

rem ---------------------------------------------------- 5. зависимости
echo [4/6] Зависимости приложения ...
if not exist ".cache" mkdir ".cache"
"%PY%" installer\setup_helper.py constraints ".cache\constraints.txt"
if errorlevel 1 goto :fail
uv pip install --python "%PY%" -r installer\requirements-books.txt -c ".cache\constraints.txt"
if errorlevel 1 goto :fail

echo [5/6] triton-windows (ускорение torch.compile) ...
"%PY%" installer\setup_helper.py triton

rem ---------------------------------------------------- 6. модель
echo [6/6] Модель VoxCPM2 (~5 ГБ, один раз) ...
"%PY%" installer\setup_helper.py download "%MODEL_DIR%"
if errorlevel 1 goto :fail

echo.
echo  ---------------------- проверка ----------------------
"%PY%" installer\setup_helper.py check
echo.
echo  ==========================================================
echo     Готово!  Запуск веб-интерфейса:   run.bat
echo     Командная строка:                 audiobook.bat --help
echo  ==========================================================
echo.
pause
exit /b 0

:fail
echo.
echo [X] Установка прервана из-за ошибки (см. сообщения выше).
echo     Частые причины: нет интернета, антивирус блокирует uv, мало места на диске.
pause
exit /b 1
