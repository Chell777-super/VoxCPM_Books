"""Обработка текста без синтеза: PDF/TXT → размеченный текст + предпросмотр озвучки.

Разметка (формат правки, его же понимает «Применить правки» и проект из .txt)::

    %title Название книги
    # Раздел I. Общие положения
    ## Глава 1. Предпринимательство и предпринимательское право
    #### Основные понятия            ← подзаголовок внутри текста
    **Бизнес** — деятельность, …     ← определение: пауза после термина
    ### § 1. Понятие предпринимательства
    Абзацы отделяются пустой строкой.

Предпросмотр озвучки показывает текст ровно так, как его прочитает модель:
после нормализации, с разбивкой на сегменты и паузами между ними.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .audio import fmt_duration
from .book import LEVEL_NAMES, Book
from .normalize import NormalizeOptions
from .pdf_extract import ExtractOptions, ExtractReport, extract_any
from .segmenter import Segment, SegmentOptions, estimate_duration_sec, segment_book


@dataclass
class TextResult:
    book: Book
    segments: List[Segment]
    markup: str
    speech: str
    summary: str
    report: Optional[ExtractReport] = None
    warnings: List[str] = field(default_factory=list)


def _summary(book: Book, segs: List[Segment], rep: Optional[ExtractReport]) -> str:
    chars = sum(len(s.text) for s in segs)
    by_level = {}
    for c in book.chapters:
        by_level[c.level] = by_level.get(c.level, 0) + 1
    struct = ", ".join(f"{LEVEL_NAMES.get(k, k)}: {v}" for k, v in sorted(by_level.items()))
    lines = [
        f"Структура: {struct or '—'}",
        f"Сегментов: {len(segs)}, символов к озвучке: {chars:,}".replace(",", " "),
        f"Ориентировочная длительность аудио: {fmt_duration(estimate_duration_sec(chars))}",
    ]
    if rep is not None:
        lines.append(
            f"Страниц: {rep.pages}; пропущено страниц (оглавление, выходные данные): {rep.removed_pages}; "
            f"удалено колонтитулов: {rep.removed_headers}, строк сносок: {rep.removed_footnotes}; "
            f"главы определены: {rep.chapter_source}"
        )
    return "\n".join(lines)


def render_speech(book: Book, segs: List[Segment], header: str = "") -> str:
    """Текст «как прочитает модель»: сегмент на строку, паузы, разделители частей."""
    out = []
    if header:
        out += [f"# {line}" for line in header.splitlines()]
        out.append("# Каждая строка — отдельный фрагмент синтеза; ⏸ — пауза тишиной после него.")
        out.append("")
    cur = None
    for s in segs:
        if s.chapter != cur:
            cur = s.chapter
            ch = book.chapters[s.chapter]
            mark = "#" * max(1, min(3, ch.level))
            out.append("")
            out.append(f"{'─' * 8} {mark} {ch.title or '(без названия)'} {'─' * 8}")
        kind = "▶" if s.kind == "heading" else " "
        out.append(f"{s.idx + 1:05d} {kind} {s.text}   ⏸ {s.pause_ms / 1000:.2f} с")
    return "\n".join(out).strip() + "\n"


def process_book(
    book: Book,
    segment: Optional[SegmentOptions] = None,
    normalize: Optional[NormalizeOptions] = None,
    report: Optional[ExtractReport] = None,
) -> TextResult:
    segs = segment_book(book, segment or SegmentOptions(), normalize or NormalizeOptions())
    summary = _summary(book, segs, report)
    return TextResult(
        book=book,
        segments=segs,
        markup=book.to_text(),
        speech=render_speech(book, segs, summary),
        summary=summary,
        report=report,
        warnings=list(report.warnings) if report else [],
    )


def process_file(
    path: str,
    extract: Optional[ExtractOptions] = None,
    segment: Optional[SegmentOptions] = None,
    normalize: Optional[NormalizeOptions] = None,
) -> TextResult:
    book, rep = extract_any(path, extract or ExtractOptions())
    if not book.title:
        book.title = Path(path).stem.replace("_", " ")
    return process_book(book, segment, normalize, rep)


def process_markup(
    markup: str,
    segment: Optional[SegmentOptions] = None,
    normalize: Optional[NormalizeOptions] = None,
    default_title: str = "",
) -> TextResult:
    """Предпросмотр по уже размеченному (и, возможно, отредактированному) тексту."""
    return process_book(Book.from_text(markup, default_title=default_title), segment, normalize)


def write_outputs(res: TextResult, out_markup: str | Path, out_speech: Optional[str | Path] = None) -> List[str]:
    Path(out_markup).write_text(res.markup, encoding="utf-8")
    files = [str(out_markup)]
    if out_speech:
        Path(out_speech).write_text(res.speech, encoding="utf-8")
        files.append(str(out_speech))
    return files
