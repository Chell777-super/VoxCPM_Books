"""Режим «Сценарий»: озвучка речей и монологов с ремарками, смехом, паузами и настроением частей.

Формат (Markdown-подобный, как пишут сценарии):

    Пояснения автора до первой «---» не читаются.
    ---
    **[Интро]** *(0:00–0:13, играют одни струнные)*        ← название части и ремарка — не читаются
    Счастливый человек… А кто сейчас счастливый?          ← каждая строка — реплика с короткой паузой
    *(≈0:13, вступает бас)*                                ← строка-ремарка — не читается

    **[Аутро]** {тихо, нежно, медленно}                    ← стиль всей части (инструкция модели)
    *(пауза 2 с)*                                          ← явная пауза
    *(тихо)* Мне бы домик…                                 ← стиль одной строки
    Не помню уже *(смеётся)*.                              ← смех → тег [laughing]
    ---
    Заметки после последней «---» не читаются.

Паузы: между строками — короткая, на пустой строке — длиннее, между частями — ещё длиннее.
Очень короткие строки («Некому!») приклеиваются к соседней: модель плохо произносит фразы короче ~1 с.

Стиль работает в клонировании по образцу БЕЗ текста образца и в дизайне голоса; в Hi-Fi режиме
(образец + точный текст) модель инструкции игнорирует — движок их тогда просто не передаёт.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .book import Book, Chapter
from .normalize import NormalizeOptions, normalize_text, speakable
from .segmenter import Segment, SegmentOptions, split_paragraph

# Метки, которые переживают нормализацию текста (символы из Private Use Area)
_LAUGH, _SIGH = "", ""
TAGS = {_LAUGH: "[laughing]", _SIGH: "[sigh]"}

SECTION_RE = re.compile(r"^\s*(?:\*\*|__)?\s*\[([^\]\n]{1,60})\]\s*(?:\*\*|__)?\s*(.*)$")
STYLE_RE = re.compile(r"\{([^{}]{1,200})\}")
# ремарка: *(…)* или _(…)_ ; строка целиком в скобках тоже ремарка
DIRECTION_RE = re.compile(r"[*_]\(([^()]{1,200})\)[*_]")
WHOLE_LINE_DIRECTION_RE = re.compile(r"^\s*[*_]?\(([^()]{1,200})\)[*_]?\s*$")
PAUSE_RE = re.compile(r"пауз\w*(?:\s+(?:в\s+)?(\d+(?:[.,]\d+)?)(?:\s*[-–—]\s*(\d+(?:[.,]\d+)?))?\s*(?:с|сек|секунд\w*)?)?", re.I)
LAUGH_RE = re.compile(r"сме[её]тся|смех|хохоч|хохот|усмеха|ухмыл|посмеива|ха-ха|хи-хи|хе-хе", re.I)
SIGH_RE = re.compile(r"вздох|вздыха", re.I)
SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")

# Русские ремарки → инструкция модели (по документации VoxCPM2 лучше понимает английский)
_STYLE_WORDS = [
    (r"шёпот|шепот|шепч", "whispering"),
    (r"тих", "speaking softly, quietly"),
    (r"громк|крич|кричит|выкрик", "loud, shouting"),
    (r"ирони|насмешл|саркаст|язвит", "ironic, sarcastic tone"),
    (r"грустн|печальн|с горечью|горько", "sad, bitter tone"),
    (r"нежн|ласков|тепл", "tender, warm and gentle"),
    (r"медленн|не спеша|растягива", "slowly"),
    (r"быстр|скороговор|торопл", "fast"),
    (r"устал", "tired voice"),
    (r"зл|раздраж|сердит|гневн", "angry, irritated"),
    (r"весел|радостн|бодр", "cheerful"),
    (r"задумчив|размышля", "thoughtful, pensive"),
    (r"с напором|страстн|горяч|эмоцион", "passionate, emphatic"),
]


def style_from_direction(direction: str) -> str:
    """«тихо, с горечью» → «speaking softly, quietly, sad, bitter tone». Пусто — если слов стиля нет."""
    found = []
    for rx, en in _STYLE_WORDS:
        if re.search(rx, direction, re.I) and en not in found:
            found.append(en)
    return ", ".join(found)


@dataclass
class ScriptOptions:
    pause_line_ms: int = 450  # между строками внутри абзаца
    pause_paragraph_ms: int = 900  # пустая строка
    pause_section_ms: int = 1500  # между частями [Интро] → [Монолог]
    min_line_chars: int = 12  # короче (< ~1 с речи) — приклеиваем к соседней строке
    max_chars: int = 300
    read_section_titles: bool = False
    laughter: str = "tag"  # tag | skip | text


@dataclass
class _Line:
    text: str  # текст с метками _LAUGH/_SIGH, без ремарок
    style: str = ""
    pause_before_ms: int = 0  # явная пауза перед строкой
    para_start: bool = False


@dataclass
class _Section:
    title: str
    style: str = ""
    lines: List[_Line] = field(default_factory=list)


def _strip_markup(s: str) -> str:
    s = re.sub(r"\*\*|__", "", s)
    s = re.sub(r"(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)", "", s)
    return s.strip()


def _pause_ms(direction: str) -> Optional[int]:
    m = PAUSE_RE.search(direction)
    if not m:
        return None
    vals = [float(x.replace(",", ".")) for x in m.groups() if x]
    sec = (sum(vals) / len(vals)) if vals else 1.2  # «пауза в одну-две секунды» → среднее
    return int(max(0.2, min(10.0, sec)) * 1000)


def _body(text: str) -> str:
    """Текст между первой и последней «---» (если разделителей два и больше), иначе после первой."""
    lines = text.replace("\r\n", "\n").split("\n")
    seps = [i for i, l in enumerate(lines) if SEPARATOR_RE.match(l)]
    if len(seps) >= 2:
        return "\n".join(lines[seps[0] + 1 : seps[-1]])
    if len(seps) == 1:
        after = "\n".join(lines[seps[0] + 1 :])
        before = "\n".join(lines[: seps[0]])
        # одна «---»: читаем ту половину, где есть разметка частей
        return after if SECTION_RE.search(after) or not SECTION_RE.search(before) else before
    return text


def parse_script(text: str, opts: Optional[ScriptOptions] = None) -> List[_Section]:
    opts = opts or ScriptOptions()
    sections: List[_Section] = []
    pending_pause = 0
    pending_style = ""
    blank = True

    def cur() -> _Section:
        if not sections:
            sections.append(_Section(title=""))
        return sections[-1]

    for raw in _body(text).split("\n"):
        line = raw.strip()
        if not line:
            blank = True
            continue
        if line.startswith("//") or line.startswith(">"):
            continue  # комментарий
        m = SECTION_RE.match(line)
        if m:
            title, rest = m.group(1).strip(), m.group(2)
            sec = _Section(title=title)
            st = STYLE_RE.search(rest)
            if st:
                sec.style = st.group(1).strip()
            for d in DIRECTION_RE.findall(rest) + re.findall(r"\(([^()]{1,200})\)", STYLE_RE.sub("", rest)):
                p = _pause_ms(d)
                if p:
                    pending_pause = max(pending_pause, p)
            sections.append(sec)
            blank = True
            continue
        whole = WHOLE_LINE_DIRECTION_RE.match(line)
        if whole:  # строка-ремарка: пауза / стиль следующей строки / смех / просто пометка
            d = whole.group(1)
            p = _pause_ms(d)
            if p:
                pending_pause = max(pending_pause, p)
            elif LAUGH_RE.search(d) or SIGH_RE.search(d):
                tag = _LAUGH if LAUGH_RE.search(d) else _SIGH
                if opts.laughter == "tag":
                    cur().lines.append(_Line(text=tag, pause_before_ms=pending_pause, para_start=blank))
                    pending_pause, blank = 0, False
            else:
                pending_style = style_from_direction(d) or pending_style
            continue

        # обычная строка: стиль из { } и ремарки внутри
        line_style = pending_style
        st = STYLE_RE.search(line)
        if st:
            line_style = st.group(1).strip()
            line = STYLE_RE.sub("", line)
        lead = re.match(r"^\s*[*_]\(([^()]{1,200})\)[*_]\s*", line)
        if lead and not (LAUGH_RE.search(lead.group(1)) or SIGH_RE.search(lead.group(1)) or PAUSE_RE.search(lead.group(1))):
            line_style = style_from_direction(lead.group(1)) or line_style
            line = line[lead.end():]

        def repl(md):
            d = md.group(1)
            if LAUGH_RE.search(d) or SIGH_RE.search(d):
                if opts.laughter == "tag":
                    return f" {_LAUGH if LAUGH_RE.search(d) else _SIGH} "
                if opts.laughter == "text":
                    return f" {d} "
                return " "
            return " "  # прочие ремарки внутри строки не читаем

        line = DIRECTION_RE.sub(repl, line)
        line = _strip_markup(line)
        if not line or not (speakable(line) or _LAUGH in line or _SIGH in line):
            continue
        cur().lines.append(_Line(text=line, style=line_style, pause_before_ms=pending_pause, para_start=blank))
        pending_pause, pending_style, blank = 0, "", False
    return [s for s in sections if s.lines]


def _finalize_text(text: str, norm: NormalizeOptions) -> str:
    """Нормализация с сохранением меток смеха/вздоха → теги модели."""
    t = normalize_text(text, norm)
    for k, v in TAGS.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+", " ", t).strip()
    # «Не помню уже [laughing].» → «Не помню уже. [laughing]» — смех после конца фразы
    t = re.sub(r"\s*(\[[\w-]+\])\s*([.!?…]+)", r"\2 \1", t)
    t = re.sub(r"\s+([.,!?…])", r"\1", t)
    return t


def _spoken_len(text: str) -> int:
    return len(re.sub(r"[]", "", text).strip())


def segment_script(
    text: str,
    opts: Optional[ScriptOptions] = None,
    norm: Optional[NormalizeOptions] = None,
) -> Tuple[Book, List[Segment]]:
    """Сценарий → (Book для интерфейса, сегменты для синтеза).

    Текст сегмента может начинаться с инструкции модели «(style)» — её добавляет только этот режим;
    движок не дублирует её глобальным стилем и убирает в Hi-Fi режиме.
    """
    opts = opts or ScriptOptions()
    norm = norm or NormalizeOptions(expand_abbreviations=True)
    sections = parse_script(text, opts)
    seg_opts = SegmentOptions(target_chars=min(opts.max_chars, 230), max_chars=opts.max_chars, min_chars=10)

    book = Book(title="", chapters=[])
    segs: List[Segment] = []

    def add(ci: int, txt: str, style: str, pause: int, kind: str = "text"):
        if not txt:
            return
        full = f"({style}){txt}" if style else txt
        segs.append(Segment(idx=len(segs), chapter=ci, text=full, kind=kind, pause_ms=pause, para_end=True))

    for ci, sec in enumerate(sections):
        book.chapters.append(Chapter(title=sec.title or f"Часть {ci + 1}", paragraphs=[l.text for l in sec.lines], level=2))
        if segs:
            segs[-1].pause_ms = max(segs[-1].pause_ms, opts.pause_section_ms)
        if opts.read_section_titles and sec.title:
            add(ci, _finalize_text(sec.title + ".", norm), "", 900, "heading")

        # склеиваем короткие строки с соседними (внутри абзаца, при одинаковом стиле, без явной паузы)
        merged: List[_Line] = []
        for ln in sec.lines:
            prev = merged[-1] if merged else None
            short = _spoken_len(ln.text) < opts.min_line_chars
            prev_short = prev is not None and _spoken_len(prev.text) < opts.min_line_chars
            if (
                prev is not None
                and (short or prev_short)
                and not ln.para_start
                and not ln.pause_before_ms
                and ln.style == prev.style
                and _spoken_len(prev.text) + _spoken_len(ln.text) < opts.max_chars
            ):
                prev.text = f"{prev.text} {ln.text}"
            else:
                merged.append(_Line(ln.text, ln.style, ln.pause_before_ms, ln.para_start))

        for k, ln in enumerate(merged):
            if segs and segs[-1].chapter == ci:
                gap = opts.pause_paragraph_ms if ln.para_start else opts.pause_line_ms
                segs[-1].pause_ms = max(segs[-1].pause_ms, gap)
            if ln.pause_before_ms and segs:
                segs[-1].pause_ms = max(segs[-1].pause_ms, ln.pause_before_ms)
            style = ln.style or sec.style
            t = _finalize_text(ln.text, norm)
            if not t:
                continue
            parts = split_paragraph(t, seg_opts) if len(t) > opts.max_chars else [t]
            for j, part in enumerate(parts):
                add(ci, part, style, opts.pause_line_ms if j < len(parts) - 1 else 0)
    if segs:
        segs[-1].pause_ms = max(segs[-1].pause_ms, 800)  # «хвост» тишины в конце трека
    for i, s in enumerate(segs):
        s.idx = i
    return book, segs


# ----------------------------------------------------------------- для движков
INSTRUCTION_RE = re.compile(r"^\(([^()]{1,200})\)")
TAG_RE = re.compile(r"\[(?:laughing|sigh|Uhm|Shh|Question-\w+|Surprise-\w+|Dissatisfaction-\w+)\]", re.I)


def split_instruction(text: str) -> Tuple[str, str]:
    """«(тихо)Текст» → ("тихо", "Текст")."""
    m = INSTRUCTION_RE.match(text)
    return (m.group(1), text[m.end():]) if m else ("", text)


def spoken_text(text: str) -> str:
    """Текст без инструкции и тегов — для контроля темпа речи."""
    return TAG_RE.sub("", split_instruction(text)[1]).strip()
