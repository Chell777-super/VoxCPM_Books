"""Помощник установщика VoxCPM Books (вызывается из install.bat / install.sh).

    python installer/setup_helper.py constraints <файл>   — зафиксировать установленный torch
    python installer/setup_helper.py triton                — поставить triton-windows под версию torch
    python installer/setup_helper.py download [папка]      — скачать VoxCPM2
    python installer/setup_helper.py check                 — проверить окружение
"""

import os
import shutil
import subprocess
import sys


def _torch_version():
    import torch

    return torch.__version__.split("+")[0]


def cmd_constraints(path):
    import torch
    import torchaudio

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"torch=={torch.__version__.split('+')[0]}\n")
        f.write(f"torchaudio=={torchaudio.__version__.split('+')[0]}\n")
    print(f"torch {torch.__version__}, torchaudio {torchaudio.__version__} зафиксированы")


# torch minor -> совместимая ветка triton (https://github.com/woct0rdho/triton-windows)
TRITON_FOR_TORCH = {(2, 6): "3.2", (2, 7): "3.3", (2, 8): "3.4", (2, 9): "3.5", (2, 10): "3.6", (2, 11): "3.7"}


def cmd_triton():
    if sys.platform != "win32":
        print("Linux: triton уже входит в сборку torch.")
        return
    major, minor = (int(x) for x in _torch_version().split(".")[:2])
    branch = TRITON_FOR_TORCH.get((major, minor))
    if branch is None:
        print(f"Нет известной версии triton-windows для torch {major}.{minor} — пропускаю (будет eager-режим).")
        return
    a, b = branch.split(".")
    spec = f"triton-windows>={a}.{b},<{a}.{int(b) + 1}"
    uv = os.environ.get("UV_EXE") or shutil.which("uv") or "uv"
    print(f"Устанавливаю {spec} …")
    rc = subprocess.call([uv, "pip", "install", "--python", sys.executable, spec])
    if rc != 0:
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
        uv = os.environ.get("UV_EXE") or shutil.which("uv") or "uv"
        subprocess.call([uv, "pip", "install", "--python", sys.executable, "modelscope"])
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
    {"constraints": cmd_constraints, "triton": cmd_triton, "download": cmd_download, "check": cmd_check}[cmd](*rest)
