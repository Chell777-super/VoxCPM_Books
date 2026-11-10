"""Структура книги: главы и абзацы + текстовый формат для ручной правки."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Chapter:
    title: str
    paragraphs: List[str] = field(default_factory=list)
    start_page: int | None = None

    @property
    def n_chars(self) -> int:
        return sum(len(p) for p in self.paragraphs) + len(self.title)


@dataclass
class Book:
    title: str = ""
    author: str = ""
    chapters: List[Chapter] = field(default_factory=list)
    source: str = ""

    @property
    def n_chars(self) -> int:
        return sum(c.n_chars for c in self.chapters)

    # ------------------------------------------------------------------ JSON
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, data: str) -> "Book":
        raw = json.loads(data)
        chapters = [Chapter(**c) for c in raw.pop("chapters", [])]
        return cls(chapters=chapters, **raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Book":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------ text format
    # Простой формат для ручной правки:
    #   # Название главы
    #   (пустая строка между абзацами)
    #   Абзац текста...
    def to_text(self) -> str:
        out = []
        if self.title:
            out.append(f"%title {self.title}")
        if self.author:
            out.append(f"%author {self.author}")
        if out:
            out.append("")
        for ch in self.chapters:
            out.append(f"# {ch.title}".rstrip())
            out.append("")
            for p in ch.paragraphs:
                out.append(p)
                out.append("")
        return "\n".join(out).rstrip() + "\n"

    @classmethod
    def from_text(cls, text: str, default_title: str = "") -> "Book":
        """Разбор текстового формата (см. :meth:`to_text`).

        Если строк-заголовков ``# ...`` нет, пытаемся найти главы по шаблонам
        («Глава 1», «Chapter IV» …), иначе вся книга — одна глава.
        """
        book = cls(title=default_title)
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        has_md_headings = any(l.startswith("# ") or l == "#" for l in lines)

        chapters: List[Chapter] = []
        buf: List[str] = []

        def flush_par():
            par = " ".join(s for s in buf if s)
            buf.clear()
            if par:
                if not chapters:
                    chapters.append(Chapter(title=""))
                chapters[-1].paragraphs.append(par)

        from .pdf_extract import looks_like_heading  # локальный импорт: без цикла

        for line in lines:
            s = line.strip()
            if s.startswith("%title "):
                book.title = s[7:].strip()
                continue
            if s.startswith("%author "):
                book.author = s[8:].strip()
                continue
            is_heading = False
            title = ""
            if has_md_headings:
                if s.startswith("#"):
                    is_heading, title = True, s.lstrip("#").strip()
            elif s and looks_like_heading(s):
                is_heading, title = True, s
            if is_heading:
                flush_par()
                chapters.append(Chapter(title=title))
            elif not s:
                flush_par()
            else:
                buf.append(s)
        flush_par()

        # Убираем пустые главы без заголовка
        book.chapters = [c for c in chapters if c.paragraphs or c.title]
        if not book.chapters:
            book.chapters = [Chapter(title="")]
        return book


_SAFE_RE = re.compile(r"[^\w\-. ]+", re.UNICODE)


def safe_filename(name: str, max_len: int = 60) -> str:
    name = _SAFE_RE.sub("", name).strip().replace(" ", "_")
    return (name[:max_len] or "book").strip("._") or "book"
