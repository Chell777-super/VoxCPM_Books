"""Проект аудиокниги на диске: настройки, текст, сегменты, аудио, результат.

Структура папки::

    projects/<имя>/
        project.json     настройки
        book.json        структура книги (главы/абзацы)
        book.txt         тот же текст в формате для ручной правки
        segments.json    сегменты + статус синтеза каждого
        voice/           образец голоса
        audio/<hash>.flac  готовые сегменты (имя = хэш текста+голоса)
        output/          MP3 / M4B
        cover.jpg

Аудио сегмента адресуется хэшем ``текст + голос``. Поэтому если поправить текст
и заново разбить книгу, уже озвученные неизменённые сегменты подхватятся сами,
а синтез продолжится с места остановки.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

from .audio import ExportOptions
from .book import Book, safe_filename
from .engine import SynthConfig, VoiceConfig
from .normalize import NormalizeOptions
from .pdf_extract import ExtractOptions
from .segmenter import Segment, SegmentOptions


def _dc_from_dict(cls, d: Optional[dict]):
    d = d or {}
    names = {f.name for f in fields(cls)}
    obj = cls(**{k: v for k, v in d.items() if k in names})
    # tuple-поля из JSON приходят списками
    for f in fields(cls):
        if isinstance(getattr(obj, f.name), list) and str(f.type).startswith(("Tuple", "tuple")):
            setattr(obj, f.name, tuple(getattr(obj, f.name)))
    return obj


@dataclass
class ProjectSettings:
    extract: ExtractOptions = field(default_factory=ExtractOptions)
    normalize: NormalizeOptions = field(default_factory=NormalizeOptions)
    segment: SegmentOptions = field(default_factory=SegmentOptions)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    export: ExportOptions = field(default_factory=ExportOptions)

    def to_dict(self) -> dict:
        return {k: asdict(getattr(self, k)) for k in ("extract", "normalize", "segment", "voice", "synth", "export")}

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectSettings":
        return cls(
            extract=_dc_from_dict(ExtractOptions, d.get("extract")),
            normalize=_dc_from_dict(NormalizeOptions, d.get("normalize")),
            segment=_dc_from_dict(SegmentOptions, d.get("segment")),
            voice=_dc_from_dict(VoiceConfig, d.get("voice")),
            synth=_dc_from_dict(SynthConfig, d.get("synth")),
            export=_dc_from_dict(ExportOptions, d.get("export")),
        )


@dataclass
class SegState:
    """Сегмент + результат синтеза."""

    idx: int
    chapter: int
    text: str
    kind: str = "text"
    pause_ms: int = 0
    para_end: bool = False
    key: str = ""
    status: str = "pending"  # pending | done | flagged | error
    seconds: float = 0.0
    cps: float = 0.0
    attempts: int = 0
    seed: int = 0
    note: str = ""


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.settings = ProjectSettings()
        self.book: Optional[Book] = None
        self.segments: List[SegState] = []
        self.chapter_titles: List[str] = []
        self.chapter_levels: List[int] = []
        if (self.root / "project.json").exists():
            self.load()

    # --------------------------------------------------------------- paths
    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def voice_dir(self) -> Path:
        return self.root / "voice"

    @property
    def cover_path(self) -> Path:
        return self.root / "cover.jpg"

    def seg_path(self, s: SegState) -> Path:
        return self.audio_dir / f"{s.key}.flac"

    @property
    def name(self) -> str:
        return self.root.name

    # ------------------------------------------------------------- create
    @classmethod
    def create(cls, projects_dir: str | Path, source_file: str, name: Optional[str] = None) -> "Project":
        base = safe_filename(name or Path(source_file).stem)
        root = Path(projects_dir) / base
        p = cls(root)
        for d in (p.root, p.audio_dir, p.output_dir, p.voice_dir):
            d.mkdir(parents=True, exist_ok=True)
        dst = p.root / ("source" + Path(source_file).suffix.lower())
        if Path(source_file).resolve() != dst.resolve():
            shutil.copyfile(source_file, dst)
        p.save()
        return p

    @property
    def source_file(self) -> Optional[Path]:
        for ext in (".pdf", ".txt", ".md"):
            f = self.root / f"source{ext}"
            if f.exists():
                return f
        return None

    # ------------------------------------------------------------ persist
    def save(self) -> None:
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_write(self.root / "project.json", json.dumps(self.settings.to_dict(), ensure_ascii=False, indent=1))
            if self.book is not None:
                _atomic_write(self.root / "book.json", self.book.to_json())
                _atomic_write(self.root / "book.txt", self.book.to_text())
            self.save_segments()

    def save_segments(self) -> None:
        with self.lock:
            data = {
                "chapters": self.chapter_titles,
                "levels": self.chapter_levels,
                "segments": [asdict(s) for s in self.segments],
            }
            _atomic_write(self.root / "segments.json", json.dumps(data, ensure_ascii=False))

    def load(self) -> None:
        with self.lock:
            self.settings = ProjectSettings.from_dict(json.loads((self.root / "project.json").read_text("utf-8")))
            if (self.root / "book.json").exists():
                self.book = Book.load(self.root / "book.json")
            sp = self.root / "segments.json"
            if sp.exists():
                data = json.loads(sp.read_text("utf-8"))
                self.chapter_titles = data.get("chapters", [])
                self.chapter_levels = data.get("levels") or [2] * len(self.chapter_titles)
                self.segments = [SegState(**s) for s in data.get("segments", [])]
            self.refresh_status()

    # ------------------------------------------------------------ segments
    def voice_signature(self) -> str:
        v = self.settings.voice
        parts = [v.mode, v.style or "", v.prompt_text or ""]
        if v.reference_wav and os.path.exists(v.reference_wav):
            h = hashlib.sha1()
            with open(v.reference_wav, "rb") as f:
                h.update(f.read())
            parts.append(h.hexdigest())
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]

    def _key(self, text: str, vsig: str) -> str:
        return hashlib.sha1(f"{vsig}|{text}".encode("utf-8")).hexdigest()[:16]

    def set_segments(self, book: Book, segments: List[Segment]) -> None:
        with self.lock:
            old: Dict[str, SegState] = {s.key: s for s in self.segments}
            self.book = book
            self.chapter_titles = [c.title or ("Вступление" if i == 0 else f"Часть {i + 1}") for i, c in enumerate(book.chapters)]
            self.chapter_levels = [c.level for c in book.chapters]
            vsig = self.voice_signature()
            self.segments = []
            for sg in segments:
                st = SegState(
                    idx=sg.idx, chapter=sg.chapter, text=sg.text, kind=sg.kind, pause_ms=sg.pause_ms, para_end=sg.para_end
                )
                st.key = self._key(sg.text, vsig)
                prev = old.get(st.key)
                if prev:
                    for f in ("status", "seconds", "cps", "attempts", "seed", "note"):
                        setattr(st, f, getattr(prev, f))
                self.segments.append(st)
            self.refresh_status()
            self.save()

    def rekey_for_voice(self) -> None:
        """Голос изменился — пересчитать ключи (старое аудио к новому голосу не подходит)."""
        with self.lock:
            vsig = self.voice_signature()
            for s in self.segments:
                s.key = self._key(s.text, vsig)
            self.refresh_status()
            self.save()

    def refresh_status(self) -> None:
        """Сверяет статусы с файлами на диске (для возобновления после сбоя)."""
        for s in self.segments:
            exists = bool(s.key) and self.seg_path(s).exists()
            if exists and s.status == "pending":
                s.status = "done"
            elif not exists and s.status in ("done", "flagged"):
                s.status = "pending"

    def update_segment_text(self, idx: int, text: str) -> None:
        with self.lock:
            s = self.segments[idx]
            s.text = text.strip()
            s.key = self._key(s.text, self.voice_signature())
            s.status, s.note = "pending", ""
            self.refresh_status()
            self.save_segments()

    # -------------------------------------------------------------- stats
    def stats(self) -> dict:
        with self.lock:
            total = len(self.segments)
            done = sum(1 for s in self.segments if s.status in ("done", "flagged"))
            flagged = sum(1 for s in self.segments if s.status == "flagged")
            errors = sum(1 for s in self.segments if s.status == "error")
            chars = sum(len(s.text) for s in self.segments)
            done_chars = sum(len(s.text) for s in self.segments if s.status in ("done", "flagged"))
            audio_sec = sum(s.seconds for s in self.segments if s.status in ("done", "flagged"))
            return dict(
                total=total,
                done=done,
                flagged=flagged,
                errors=errors,
                chars=chars,
                done_chars=done_chars,
                audio_sec=audio_sec,
                chapters=len(self.chapter_titles),
            )

    def chapter_segments(self, ci: int) -> List[SegState]:
        return [s for s in self.segments if s.chapter == ci]

    def file_groups(self, split_level: int) -> List[tuple]:
        """Группирует главы в выходные файлы: [(название_файла, [индексы глав])].

        Новый файл начинается с заголовка уровня <= split_level (1 — разделы, 2 — главы,
        3 — параграфы). Заголовок без собственного текста («Раздел I») приклеивается к
        следующему файлу: «Раздел I. … Глава 1. …» звучит в начале файла главы 1.
        """
        has_body = {ci for ci in range(len(self.chapter_titles)) if any(s.kind == "text" for s in self.chapter_segments(ci))}
        groups: List[list] = []
        for ci, title in enumerate(self.chapter_titles):
            lvl = self.chapter_levels[ci] if ci < len(self.chapter_levels) else 2
            starts = lvl <= split_level
            if not groups or (starts and any(c in has_body for c in groups[-1][1])):
                groups.append([title, [ci], lvl])
            else:
                groups[-1][1].append(ci)
                # Название файла — самый «мелкий» заголовок <= split_level до начала текста
                if starts and not any(c in has_body for c in groups[-1][1][:-1]):
                    groups[-1][0] = title
        return [(t, cis) for t, cis, _ in groups]


def list_projects(projects_dir: str | Path) -> List[str]:
    d = Path(projects_dir)
    if not d.exists():
        return []
    items = [p for p in d.iterdir() if (p / "project.json").exists()]
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in items]
