"""Пост-обработка сегментов, склейка глав и экспорт (WAV → MP3 / M4B) через ffmpeg.

Всё работает потоково: в памяти держится максимум один сегмент, поэтому даже
20-часовая книга собирается на машине с 16 ГБ ОЗУ.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------- #
# Обработка сигнала
# --------------------------------------------------------------------------- #


def _frame_db(wav: np.ndarray, frame: int) -> np.ndarray:
    n = len(wav) // frame
    if n == 0:
        return np.array([-120.0])
    fr = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def trim_silence(wav: np.ndarray, sr: int, rel_db: float = -40.0, keep_ms: int = 60) -> np.ndarray:
    """Обрезает тишину/дыхание по краям (порог относительно пика сегмента)."""
    if wav.size == 0:
        return wav
    frame = max(1, sr // 100)  # 10 мс
    db = _frame_db(wav, frame)
    thr = db.max() + rel_db
    active = np.where(db > thr)[0]
    if active.size == 0:
        return wav[:0]
    keep = int(keep_ms * sr / 1000)
    start = max(0, active[0] * frame - keep)
    end = min(len(wav), (active[-1] + 1) * frame + keep)
    return wav[start:end]


def loudness_normalize(
    wav: np.ndarray, sr: int, target_db: float = -20.0, peak_db: float = -1.5, max_gain_db: float = 12.0
) -> np.ndarray:
    """Выравнивание громкости сегмента по RMS «речевых» кадров с ограничением пика."""
    if wav.size == 0:
        return wav
    frame = max(1, sr // 50)  # 20 мс
    db = _frame_db(wav, frame)
    speech = db[db > db.max() - 30]
    if speech.size == 0:
        return wav
    cur = 10 * np.log10(np.mean(10 ** (speech / 10)) + 1e-12)
    gain_db = float(np.clip(target_db - cur, -max_gain_db, max_gain_db))
    out = wav * (10 ** (gain_db / 20))
    peak = float(np.max(np.abs(out))) + 1e-9
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        out = out * (limit / peak)
    return out.astype(np.float32)


def apply_fades(wav: np.ndarray, sr: int, fade_in_ms: int = 8, fade_out_ms: int = 25) -> np.ndarray:
    wav = wav.astype(np.float32, copy=True)
    fi = min(len(wav) // 2, int(sr * fade_in_ms / 1000))
    fo = min(len(wav) // 2, int(sr * fade_out_ms / 1000))
    if fi > 0:
        wav[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        wav[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return wav


def postprocess(wav: np.ndarray, sr: int, target_db: float = -20.0) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    wav = np.nan_to_num(wav)
    wav = wav - float(np.mean(wav))  # DC offset
    wav = trim_silence(wav, sr)
    wav = loudness_normalize(wav, sr, target_db=target_db)
    return apply_fades(wav, sr)


def write_segment(path: str | Path, wav: np.ndarray, sr: int) -> None:
    tmp = str(path) + ".tmp.flac"
    sf.write(tmp, wav, sr, format="FLAC", subtype="PCM_16")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run_ffmpeg(args: List[str]) -> None:
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg не найден. Установите пакет imageio-ffmpeg или добавьте ffmpeg в PATH.")
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    proc = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, **kwargs)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg: " + proc.stderr.decode("utf-8", "replace")[-2000:])


def _run_ffmpeg_pcm(wav_paths: Sequence[str], args: List[str]) -> None:
    """Запускает ffmpeg, подавая звук из WAV-файлов «сырым» PCM s16le через stdin.

    Зачем: ffmpeg из Ubuntu (6.1) при чтении WAV ищет в PCM-данных поток S/PDIF и на некоторых
    наших файлах ошибочно принимает их за MPEG-аудио («mp1») — длительность и звук портятся.
    С явным форматом входа (-f s16le) ничего не угадывается. Заодно не нужен concat-список.
    Вход 0 — этот поток; в args нумерация дополнительных входов (-i …) начинается с 1.
    """
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg не найден. Установите пакет imageio-ffmpeg или добавьте ffmpeg в PATH.")
    if not wav_paths:
        raise ValueError("нет аудио для кодирования")
    sr = sf.info(str(wav_paths[0])).samplerate
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", *args]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kwargs)
    err_chunks: List[bytes] = []

    import threading

    t = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()), daemon=True)
    t.start()
    try:
        for p in wav_paths:
            with sf.SoundFile(str(p)) as f:
                if f.samplerate != sr or f.channels != 1:
                    raise RuntimeError(f"{p}: ожидалось моно {sr} Гц")
                for block in f.blocks(blocksize=sr * 10, dtype="int16"):
                    proc.stdin.write(np.ascontiguousarray(block).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    finally:
        rc = proc.wait()
        t.join(timeout=5)
    if rc != 0:
        raise RuntimeError("ffmpeg: " + b"".join(err_chunks).decode("utf-8", "replace")[-2000:])


def _meta_escape(s: str) -> str:
    for ch in ("\\", "=", ";", "#", "\n"):
        s = s.replace(ch, "\\" + ch)
    return s


# --------------------------------------------------------------------------- #
# Сборка
# --------------------------------------------------------------------------- #


@dataclass
class ExportOptions:
    formats: Tuple[str, ...] = ("mp3",)  # mp3 | m4b | wav
    mp3_bitrate: str = "320k"  # максимум для MP3
    aac_bitrate: str = "256k"
    sample_rate: int = 48000  # 48000 (оригинал) / 44100 / 24000
    target_db: float = -20.0
    split_level: int = 3  # файлы по: 1 — разделам, 2 — главам, 3 — параграфам
    marker_level: int = 3  # метки глав в M4B до этого уровня
    folders: bool = True  # MP3 по папкам: Раздел / Глава / файлы §
    max_part_mb: int = 0  # делить M4B на части не больше N МБ (0 — одним файлом)
    preset: str = ""


# Лимиты пересылки: Telegram без Premium — 2 ГБ, с Premium — 4 ГБ (берём с запасом)
TELEGRAM_PART_MB = 1900
TELEGRAM_PREMIUM_PART_MB = 3900


def iphone_preset(opts: Optional[ExportOptions] = None) -> ExportOptions:
    """M4B для iPhone (BookPlayer / Apple Books): AAC 96 кбит/с моно 44.1 кГц (~43 МБ на час —
    для речи на слух без потерь), метки по каждому §, части до 1.9 ГБ (влезают в Telegram)."""
    base = opts or ExportOptions()
    return ExportOptions(
        formats=("m4b",),
        mp3_bitrate=base.mp3_bitrate,
        aac_bitrate="96k",
        sample_rate=44100,
        target_db=base.target_db,
        split_level=base.split_level,
        marker_level=3,
        folders=base.folders,
        max_part_mb=TELEGRAM_PART_MB,
        preset="iphone",
    )


def bitrate_kbps(b: str) -> float:
    b = str(b).strip().lower()
    return float(b[:-1]) if b.endswith("k") else float(b) / 1000


def estimate_m4b_mb(duration_sec: float, aac_bitrate: str) -> float:
    """Оценка размера M4B: аудио + ~2 % на контейнер, метки и обложку."""
    return duration_sec * bitrate_kbps(aac_bitrate) * 1000 / 8 / 1e6 * 1.02 + 0.5


def assemble_chapter(
    items: Iterable[tuple],
    out_wav: str | Path,
    sr: int,
    lead_in_ms: int = 300,
) -> tuple:
    """Склеивает [(путь, пауза_мс[, метка])] в один WAV.

    Возвращает (длительность_с, [(метка, начало_с)]) — метка ставится там, где она указана
    (начало главы/параграфа внутри файла).
    """
    total = 0
    markers = []
    with sf.SoundFile(str(out_wav), "w", samplerate=sr, channels=1, subtype="PCM_16", format="WAV") as out:
        lead = np.zeros(int(sr * lead_in_ms / 1000), dtype=np.float32)
        out.write(lead)
        total += len(lead)
        for item in items:
            path, pause_ms = item[0], item[1]
            mark = item[2] if len(item) > 2 else None
            if mark:
                markers.append((mark, 0.0 if total == len(lead) else total / sr))  # первая метка — с начала файла
            wav, file_sr = sf.read(path, dtype="float32")
            if file_sr != sr:
                raise RuntimeError(f"Частота дискретизации {path} = {file_sr}, ожидалось {sr}")
            out.write(wav)
            pause = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)
            out.write(pause)
            total += len(wav) + len(pause)
    return total / sr, markers


def encode_mp3(
    wav_path: str,
    out_path: str,
    opts: ExportOptions,
    title: str = "",
    album: str = "",
    artist: str = "",
    track: Optional[int] = None,
    cover: Optional[str] = None,
) -> None:
    args = []
    if cover and os.path.exists(cover):
        args += ["-i", cover, "-map", "0:a", "-map", "1:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic"]
    # CBR + -compression_level 0 = LAME -q 0 (самый тщательный кодер, максимальное качество)
    args += ["-c:a", "libmp3lame", "-b:a", opts.mp3_bitrate, "-compression_level", "0"]
    args += ["-ar", str(opts.sample_rate), "-ac", "1"]
    args += ["-id3v2_version", "3"]
    for k, v in (("title", title), ("album", album), ("artist", artist), ("genre", "Audiobook")):
        if v:
            args += ["-metadata", f"{k}={v}"]
    if track is not None:
        args += ["-metadata", f"track={track}"]
    _run_ffmpeg_pcm([wav_path], args + [out_path])


def build_m4b(
    chapter_wavs: Sequence[str],
    markers: Sequence[Tuple[str, float]],
    total_duration: float,
    out_path: str,
    opts: ExportOptions,
    title: str = "",
    artist: str = "",
    cover: Optional[str] = None,
    work_dir: Optional[str] = None,
    album: str = "",
    track: Optional[str] = None,
) -> None:
    """Склеивает WAV-файлы в M4B. markers — [(название, начало_с)] от начала файла."""
    work = Path(work_dir or Path(out_path).parent)
    meta = work / "_chapters.txt"
    with open(meta, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        if title:
            f.write(f"title={_meta_escape(title)}\n")
        if album or title:
            f.write(f"album={_meta_escape(album or title)}\n")
        if artist:
            f.write(f"artist={_meta_escape(artist)}\nalbum_artist={_meta_escape(artist)}\n")
        if track:
            f.write(f"track={_meta_escape(track)}\n")
        f.write("genre=Audiobook\n")
        f.write("media_type=2\n")  # тег stik=Audiobook: Apple Books/BookPlayer видят аудиокнигу, а не музыку
        ms = sorted(markers, key=lambda m: m[1])
        for k, (name, start) in enumerate(ms):
            end = ms[k + 1][1] if k + 1 < len(ms) else total_duration
            if end - start < 0.05:
                continue
            f.write(f"\n[CHAPTER]\nTIMEBASE=1/1000\nSTART={int(round(start * 1000))}\nEND={int(round(end * 1000))}\n")
            f.write(f"title={_meta_escape(name)}\n")
    args = ["-i", str(meta)]
    has_cover = bool(cover and os.path.exists(cover))
    if has_cover:
        args += ["-i", cover]
    args += ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
    if has_cover:
        args += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"]
    args += ["-c:a", "aac", "-b:a", opts.aac_bitrate, "-ar", str(opts.sample_rate), "-ac", "1"]
    args += ["-movflags", "+faststart", "-f", "mp4", out_path]
    try:
        _run_ffmpeg_pcm(list(chapter_wavs), args)
    finally:
        for p in (meta,):
            try:
                p.unlink()
            except OSError:
                pass


def fmt_duration(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


ProgressFn = Callable[[float, str], None]
