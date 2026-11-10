@echo off
rem Общие переменные окружения для install.bat / run.bat / audiobook.bat
rem Все кэши держим внутри папки программы: так torch.compile/triton не ломаются
rem из-за кириллицы в имени пользователя Windows (C:\Users\Гарик\...).
set "VB_ROOT=%~dp0.."
for %%I in ("%VB_ROOT%") do set "VB_ROOT=%%~fI"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "UV_LINK_MODE=copy"
set "HF_HOME=%VB_ROOT%\.cache\huggingface"
set "TORCHINDUCTOR_CACHE_DIR=%VB_ROOT%\.cache\inductor"
set "TRITON_CACHE_DIR=%VB_ROOT%\.cache\triton"
set "TOKENIZERS_PARALLELISM=false"
set "GRADIO_ANALYTICS_ENABLED=False"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "VOXBOOKS_MODEL=%VB_ROOT%\models\VoxCPM2"
set "VOXBOOKS_PROJECTS=%VB_ROOT%\projects"
exit /b 0
