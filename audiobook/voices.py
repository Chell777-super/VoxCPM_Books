"""Библиотека голосов: образец + точная транскрипция, общие для всех книг.

    voices/<имя>/voice.wav       образец (как загружен: wav/mp3/flac)
    voices/<имя>/meta.json       {"transcript": "...", "style": "...", "note": "..."}

С транскрипцией голос используется в режиме Hi-Fi клонирования (VoxCPM2:
reference_wav + prompt_wav + prompt_text) — максимальная похожесть.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .book import safe_filename

DEFAULT_VOICES_DIR = os.environ.get("VOXBOOKS_VOICES", "voices")
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


@dataclass
class VoiceEntry:
    name: str
    audio: str
    transcript: str = ""
    style: str = ""

    @property
    def hifi(self) -> bool:
        return bool(self.transcript.strip())


def list_voices(voices_dir: str | Path = DEFAULT_VOICES_DIR) -> List[str]:
    d = Path(voices_dir)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and _find_audio(p))


def _find_audio(folder: Path) -> Optional[Path]:
    for ext in AUDIO_EXTS:
        f = folder / f"voice{ext}"
        if f.exists():
            return f
    return None


def save_voice(
    name: str, audio_src: str, transcript: str = "", style: str = "", voices_dir: str | Path = DEFAULT_VOICES_DIR
) -> VoiceEntry:
    if not name.strip():
        raise ValueError("Укажите имя голоса")
    if not audio_src or not os.path.exists(audio_src):
        raise FileNotFoundError("Нет файла образца")
    folder = Path(voices_dir) / safe_filename(name, 50)
    folder.mkdir(parents=True, exist_ok=True)
    ext = Path(audio_src).suffix.lower() if Path(audio_src).suffix.lower() in AUDIO_EXTS else ".wav"
    for old in folder.glob("voice.*"):
        if old.resolve() != Path(audio_src).resolve():
            old.unlink()
    dst = folder / f"voice{ext}"
    if Path(audio_src).resolve() != dst.resolve():
        shutil.copyfile(audio_src, dst)
    meta = {"transcript": transcript.strip(), "style": style.strip()}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return VoiceEntry(folder.name, str(dst), meta["transcript"], meta["style"])


def load_voice(name: str, voices_dir: str | Path = DEFAULT_VOICES_DIR) -> VoiceEntry:
    folder = Path(voices_dir) / name
    audio = _find_audio(folder)
    if audio is None:
        raise FileNotFoundError(f"Голос «{name}» не найден в {voices_dir}")
    meta = {}
    if (folder / "meta.json").exists():
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return VoiceEntry(name, str(audio), meta.get("transcript", ""), meta.get("style", ""))


def delete_voice(name: str, voices_dir: str | Path = DEFAULT_VOICES_DIR) -> None:
    folder = Path(voices_dir) / name
    if folder.is_dir():
        shutil.rmtree(folder)


def check_reference(path: str) -> str:
    """Короткий отчёт об образце: длительность, каналы, клиппинг + рекомендации документации VoxCPM2."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(path, sr=None, mono=False)
        channels = 1 if y.ndim == 1 else y.shape[0]
        mono = y if y.ndim == 1 else y.mean(axis=0)
        dur = len(mono) / sr
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    except Exception as e:  # noqa: BLE001
        return f"⚠ Не удалось прочитать образец: {e}"
    msg = [f"Образец: {dur:.1f} с, {sr} Гц, {'моно' if channels == 1 else f'{channels} канала'}."]
    if dur < 5:
        msg.append("⚠ Короче 5 с — тембр передастся хуже (рекомендуется 5–30 с, лучше 10–20 с).")
    elif dur > 30:
        msg.append(
            "⚠ Длиннее 30 с. В Hi-Fi режиме образец добавляется к КАЖДОМУ сегменту — синтез замедлится. "
            "Лучше обрезать до 10–20 с (и транскрипцию тоже)."
        )
    else:
        msg.append("✓ Длительность в рекомендуемом диапазоне.")
    if peak >= 0.999:
        msg.append("⚠ Есть клиппинг (перегруз) — лучше записать тише.")
    return " ".join(msg)
