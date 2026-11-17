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
rem  PyTorch: latest = самая новая версия с самой новой CUDA, которую поддерживает
rem  драйвер (CUDA 13.x при драйвере >= 580), с автоматическим откатом на проверенную
rem  2.10 + CUDA 12.6. stable = сразу проверенная версия.
set "TORCH_CHANNEL=latest"
set "VENV=.venv"
set "MODEL_DIR=models\VoxCPM2"

call "%~dp0installer\env.bat"

echo.
echo  ==========================================================
echo     VoxCPM Books - установка  (uv + PyTorch %TORCH_CHANNEL%)
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
echo [3/6] PyTorch (%TORCH_CHANNEL%) под ваш драйвер - около 3 ГБ, наберитесь терпения ...
"%PY%" installer\setup_helper.py torch %TORCH_CHANNEL%
if errorlevel 1 goto :fail

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
