"""Помощник установщика VoxCPM Books (вызывается из install.bat / install.sh).

    python installer/setup_helper.py torch [latest|stable]  — PyTorch под драйвер NVIDIA
    python installer/setup_helper.py constraints <файл>    — зафиксировать установленный torch
    python installer/setup_helper.py triton                 — поставить triton-windows под версию torch
    python installer/setup_helper.py download [папка]       — скачать VoxCPM2
    python installer/setup_helper.py check                  — проверить окружение
"""

import os
import re
import shutil
import subprocess
import sys

# Проверенная связка (на ней VoxCPM2 тестировали авторы) — запасной вариант.
STABLE_TORCH = ("2.10.0", "cu126")
# Сборки PyTorch по CUDA (от новых к старым) и минимальная версия драйвера NVIDIA для каждой.
CUDA_BUILDS = [("cu132", 580), ("cu130", 580), ("cu128", 570), ("cu126", 560)]
TORCH_INDEX = "https://download.pytorch.org/whl/"


def _uv():
    return os.environ.get("UV_EXE") or shutil.which("uv") or "uv"


def _pip(*args) -> int:
    return subprocess.call([_uv(), "pip", "install", "--python", sys.executable, *args])


def _torch_version():
    import torch

    return torch.__version__.split("+")[0]


def _driver_version():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        return float(".".join(out[0].strip().split(".")[:2])) if out else None
    except Exception:
        return None


_GPU_TEST = (
    "import torch,sys;"
    "assert torch.cuda.is_available(),'CUDA недоступна';"
    "x=torch.randn(512,512,device='cuda',dtype=torch.bfloat16);"
    "torch.cuda.synchronize();(x@x).float().sum().item();"
    "print('  GPU-тест ок:',torch.__version__,'CUDA',torch.version.cuda,torch.cuda.get_device_name(0))"
)


def _gpu_ok() -> bool:
    return subprocess.call([sys.executable, "-c", _GPU_TEST]) == 0


def _install_torchaudio(torch_ver: str, tag: str) -> int:
    major, minor = (int(x) for x in torch_ver.split(".")[:2])
    if (major, minor) >= (2, 11):
        # с 2.11 torchaudio — чистый Python и не привязан к версии torch
        return _pip("torchaudio>=2.11", "--no-deps")
    return _pip(f"torchaudio=={torch_ver}", "--no-deps", "--index-url", TORCH_INDEX + tag)


def cmd_torch(channel="latest"):
    """Ставит самый новый PyTorch с самой новой CUDA, которую поддерживает драйвер.

    Если сборка не скачалась или не прошла GPU-тест — следующая, в конце
    проверенная STABLE_TORCH. channel=stable — сразу проверенная версия.
    """
    drv = _driver_version()
    print(f"Драйвер NVIDIA: {drv or 'не найден'}")
    candidates = []
    if channel == "latest":
        for tag, min_drv in CUDA_BUILDS:
            if drv is None or drv >= min_drv:
                candidates.append((None, tag))
        if drv is not None and drv < 580:
            print("  Подсказка: драйвер >= 580 даст CUDA 13 и самый свежий PyTorch.")
    candidates.append(STABLE_TORCH)

    for ver, tag in candidates:
        spec = f"torch=={ver}" if ver else "torch"
        print(f"\n→ Пробую {spec} ({tag}) …")
        rc = _pip(spec, "--index-url", TORCH_INDEX + tag, "--reinstall-package", "torch")
        if rc != 0:
            print(f"  сборка {tag} недоступна")
            continue
        installed = subprocess.run(
            [sys.executable, "-c", "import torch;print(torch.__version__.split('+')[0])"], capture_output=True, text=True
        ).stdout.strip()
        if drv is not None and not _gpu_ok():
            print(f"  torch {installed} ({tag}) не работает с вашей видеокартой/драйвером — пробую следующий")
            continue
        if _install_torchaudio(installed, tag) != 0:
            print("  ⚠ torchaudio не установился")
            continue
        print(f"✅ PyTorch {installed} + {tag}")
        return
    print("✖ Не удалось установить рабочий PyTorch. Обновите драйвер NVIDIA и повторите.")
    sys.exit(1)


def cmd_constraints(path):
    import torch
    import torchaudio

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"torch=={torch.__version__.split('+')[0]}\n")
        f.write(f"torchaudio=={torchaudio.__version__.split('+')[0]}\n")
    print(f"torch {torch.__version__}, torchaudio {torchaudio.__version__} зафиксированы")


# Запасная таблица: torch minor -> ветка triton (из метаданных torch на PyPI)
TRITON_FOR_TORCH = {
    (2, 6): "3.2", (2, 7): "3.3", (2, 8): "3.4", (2, 9): "3.5", (2, 10): "3.6",
    (2, 11): "3.6", (2, 12): "3.7", (2, 13): "3.7", (2, 14): "3.8", (2, 15): "3.9",
}


def _triton_branch_from_metadata():
    """Версию triton torch сам указывает в метаданных (для Linux) — берём её же для Windows."""
    try:
        from importlib.metadata import requires

        for r in requires("torch") or []:
            m = re.match(r"triton\s*(?:==|~=)\s*(\d+)\.(\d+)", r)
            if m:
                return f"{m.group(1)}.{m.group(2)}"
    except Exception:
        pass
    return None


def cmd_triton():
    if sys.platform != "win32":
        print("Linux: triton уже входит в сборку torch.")
        return
    major, minor = (int(x) for x in _torch_version().split(".")[:2])
    branch = _triton_branch_from_metadata() or TRITON_FOR_TORCH.get((major, minor))
    if branch is None:
        print(f"Нет известной версии triton-windows для torch {major}.{minor} — пропускаю (будет eager-режим).")
        return
    a, b = branch.split(".")
    spec = f"triton-windows>={a}.{b},<{a}.{int(b) + 1}"
    print(f"Устанавливаю {spec} …")
    if _pip(spec) != 0:
        print("⚠ triton-windows не установился — приложение будет работать без torch.compile (медленнее ~1.5–2×).")


def _have_model(d):
    return os.path.isfile(os.path.join(d, "config.json")) and os.path.isfile(os.path.join(d, "model.safetensors"))


def cmd_download(target="models/VoxCPM2", repo="openbmb/VoxCPM2"):
    if _have_model(target):
        print(f"Модель уже скачана: {target}")
        return
    os.makedirs(target, exist_ok=True)
    errors = []
    for endpoint in (None, "https://hf-mirror.com"):
        try:
            if endpoint:
                os.environ["HF_ENDPOINT"] = endpoint
                print(f"Пробую зеркало {endpoint} …")
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo, local_dir=target)
            if _have_model(target):
                print(f"✅ Модель скачана в {target}")
                return
        except Exception as e:  # noqa: BLE001
            errors.append(f"{endpoint or 'huggingface.co'}: {e}")
            print(f"⚠ {errors[-1]}")
    try:
        print("Пробую ModelScope …")
        _pip("modelscope")
        from modelscope import snapshot_download as ms_download

        ms_download("OpenBMB/VoxCPM2", local_dir=target)
        if _have_model(target):
            print(f"✅ Модель скачана в {target}")
            return
    except Exception as e:  # noqa: BLE001
        errors.append(f"modelscope: {e}")
    print("✖ Не удалось скачать модель. Скачайте вручную https://huggingface.co/openbmb/VoxCPM2 в папку " + target)
    sys.exit(1)


def cmd_check():
    ok = True
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(here, "src"))
    if any(ord(c) > 127 for c in here) or " " in here:
        print(f"⚠ Путь {here} содержит пробелы или не-латинские символы — torch.compile/triton может не работать.")
        print("  Рекомендуется переместить папку, например, в C:\\VoxBooks")
    try:
        import torch

        print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} ГБ, bf16: {torch.cuda.is_bf16_supported()}")
        else:
            ok = False
            print("✖ CUDA недоступна! Обновите драйвер NVIDIA (>= 560) и переустановите.")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"✖ torch: {e}")
    try:
        import triton

        print(f"triton {triton.__version__} — torch.compile доступен")
    except Exception:
        print("· triton не найден — будет eager-режим (работает, но медленнее)")
    try:
        import gradio
        import pymupdf  # noqa: F401
        import razdel  # noqa: F401

        print(f"gradio {gradio.__version__}, PyMuPDF, razdel — ок")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"✖ {e}")
    try:
        import imageio_ffmpeg

        print(f"ffmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"✖ ffmpeg: {e}")
    try:
        import voxcpm  # noqa: F401

        print("voxcpm — ок")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"✖ voxcpm: {e}")
    print("✅ Всё готово" if ok else "⚠ Есть проблемы, см. выше")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    cmd, rest = args[0], args[1:]
    {"torch": cmd_torch, "constraints": cmd_constraints, "triton": cmd_triton, "download": cmd_download, "check": cmd_check}[cmd](*rest)
