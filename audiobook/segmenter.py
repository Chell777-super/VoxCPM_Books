"""Разбиение книги на сегменты для синтеза.

Правила (по убыванию важности):

1. Сегмент никогда не пересекает границу главы.
2. Сегмент никогда не режет предложение, если оно помещается в ``max_chars``.
3. Длинное предложение режется по самой «сильной» внутренней границе ближе к
   середине: ``;`` → ``:`` → `` — `` → ``,`` → пробел.
4. Абзац делится на сегменты *оптимально* (динамическое программирование):
   длины выравниваются вокруг ``target_chars``, разрезы по границам предложений
   дешевле, чем по запятым, а крошечные «хвосты» не возникают.
5. Очень короткие соседние абзацы (реплики диалога «— Да.») объединяются,
   чтобы не синтезировать по два слова — модель на таких фрагментах менее стабильна.

После каждого сегмента задаётся пауза: внутри абзаца — короткая, в конце абзаца —
длиннее, после заголовка и на смене сцены (``* * *``) — ещё длиннее.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from .book import SUBHEAD_PREFIX, Book
from .normalize import NormalizeOptions, normalize_text, speakable


@dataclass
class SegmentOptions:
    target_chars: int = 230
    max_chars: int = 380
    min_chars: int = 50
    merge_short_paragraphs: bool = True
    read_chapter_titles: bool = True
    pause_segment_ms: int = 220
    pause_paragraph_ms: int = 650
    pause_heading_ms: int = 1100
    pause_scene_ms: int = 1500
    pause_chapter_end_ms: int = 2000


@dataclass
class Segment:
    idx: int
    chapter: int
    text: str
    kind: str = "text"  # text | heading
    pause_ms: int = 0
    para_end: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Предложения
# --------------------------------------------------------------------------- #

_INITIALS_RE = re.compile(r"\b([A-ZА-ЯЁ])\.\s?(?=[A-ZА-ЯЁ])")
_SPLIT_RE = re.compile(r"(?<=[.!?…])([»\"”’)\]]*)\s+(?=[—–\-«\"„(\[]?\s?[A-ZА-ЯЁ0-9])")


def _regex_sentences(text: str) -> List[str]:
    guarded = _INITIALS_RE.sub(lambda m: m.group(1) + "․", text)  # защищаем инициалы
    parts = []
    last = 0
    for m in _SPLIT_RE.finditer(guarded):
        end = m.start() + len(m.group(1))
        parts.append(guarded[last:end])
        last = m.end()
    parts.append(guarded[last:])
    return [p.replace("․", ".").strip() for p in parts if p.strip()]


def split_sentences(text: str, lang: str = "ru") -> List[str]:
    if lang == "ru":
        try:
            from razdel import sentenize

            sents = [s.text.strip() for s in sentenize(text) if s.text.strip()]
            return _fix_dialog_splits(sents)
        except ImportError:
            pass
    return _fix_dialog_splits(_regex_sentences(text))


def _fix_dialog_splits(sents: List[str]) -> List[str]:
    """«— Привет! — сказал он.» не должно делиться на два предложения."""
    out: List[str] = []
    for s in sents:
        if out and re.match(r"^[—–-]\s*[a-zа-яё]", s):
            out[-1] = out[-1] + " " + s
        elif out and s[:1].islower():
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Длинные предложения
# --------------------------------------------------------------------------- #

# (регулярка, «сила» границы: чем меньше, тем естественнее пауза)
_CLAUSE_BOUNDARIES = [
    (re.compile(r"[;]\s+"), 1),
    (re.compile(r":\s+"), 1),
    (re.compile(r"\s+[—–]\s+"), 2),
    (re.compile(r",\s+"), 2),
    (re.compile(r"\s+"), 4),
]
_STRENGTH_PENALTY = {0: 0.0, 1: 0.35, 2: 0.6, 4: 2.5}


def split_long(sentence: str, max_chars: int) -> List[Tuple[str, int]]:
    """Режет предложение на куски ≤ max_chars. Возвращает [(кусок, сила_границы_после)]."""
    sentence = sentence.strip()
    n = len(sentence)
    if n <= max_chars:
        return [(sentence, 0)]
    for rx, strength in _CLAUSE_BOUNDARIES:
        best = None
        for m in rx.finditer(sentence):
            sep = m.group(0)
            if sep[:1] in ",;:":
                left, right = sentence[: m.start() + 1], sentence[m.end():]
            elif sep.strip() in ("—", "–"):
                # тире переносим в начало правой части: «…слово — и дальше»
                left, right = sentence[: m.start()], sentence[m.start():]
            else:
                left, right = sentence[: m.start()], sentence[m.end():]
            left, right = left.strip(), right.strip()
            if len(left) < n * 0.2 or len(right) < n * 0.2:
                continue
            score = abs(len(left) - n / 2)
            if best is None or score < best[0]:
                best = (score, left, right)
        if best:
            _, left, right = best
            res = split_long(left, max_chars)
            res[-1] = (res[-1][0], max(res[-1][1], strength))
            return res + split_long(right, max_chars)
    # Совсем без пробелов (очень редкий случай) — режем жёстко
    chunks = [sentence[i : i + max_chars] for i in range(0, n, max_chars)]
    return [(c, 4) for c in chunks[:-1]] + [(chunks[-1], 0)]


# --------------------------------------------------------------------------- #
# Оптимальная упаковка
# --------------------------------------------------------------------------- #


def pack_pieces(pieces: List[Tuple[str, int]], target: int, max_chars: int, min_chars: int) -> List[str]:
    """Оптимально группирует куски в сегменты (DP).

    Стоимость сегмента = ((len - ideal)/ideal)^2 + штраф за слабую границу на конце
    + большой штраф за длину < min_chars.
    """
    n = len(pieces)
    if n == 0:
        return []
    total = sum(len(p) for p, _ in pieces) + n - 1
    if total <= max_chars and (total <= target * 1.25 or n == 1):
        return [" ".join(p for p, _ in pieces)]
    k = max(1, math.ceil(total / target))
    ideal = total / k

    lens = [len(p) for p, _ in pieces]
    INF = float("inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        length = -1
        for i in range(j - 1, -1, -1):
            length += lens[i] + 1
            if length > max_chars and i < j - 1:
                break
            seg_cost = ((length - ideal) / ideal) ** 2
            if length < min_chars and n > 1:
                seg_cost += 3.0
            seg_cost += _STRENGTH_PENALTY.get(pieces[j - 1][1], 1.0) if j < n else 0.0
            c = best[i] + seg_cost
            if c < best[j]:
                best[j] = c
                back[j] = i
    out = []
    j = n
    while j > 0:
        i = back[j]
        out.append(" ".join(p for p, _ in pieces[i:j]))
        j = i
    return out[::-1]


def split_paragraph(text: str, opts: SegmentOptions, lang: str = "ru") -> List[str]:
    pieces: List[Tuple[str, int]] = []
    for s in split_sentences(text, lang):
        pieces.extend(split_long(s, opts.max_chars))
    return pack_pieces(pieces, opts.target_chars, opts.max_chars, opts.min_chars)


# --------------------------------------------------------------------------- #
# Книга → сегменты
# --------------------------------------------------------------------------- #


def _heading_text(title: str) -> str:
    t = title.strip()
    if t.isupper() and len(t) > 4:  # «ГЛАВА 1. ВСТРЕЧА» → «Глава 1. встреча» → с заглавной после точки
        t = re.sub(r"(^|[.!?…]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), t.lower())
    if t and not re.search(r"[.!?…]$", t):
        t += "."
    return t


def segment_book(
    book: Book,
    seg_opts: Optional[SegmentOptions] = None,
    norm_opts: Optional[NormalizeOptions] = None,
) -> List[Segment]:
    seg_opts = seg_opts or SegmentOptions()
    norm_opts = norm_opts or NormalizeOptions()
    lang = norm_opts.lang
    segs: List[Segment] = []

    def add(ch: int, text: str, kind: str, pause: int, para_end: bool):
        if not speakable(text):
            if segs and segs[-1].chapter == ch:
                segs[-1].pause_ms = max(segs[-1].pause_ms, pause)
            return
        segs.append(Segment(idx=len(segs), chapter=ch, text=text, kind=kind, pause_ms=pause, para_end=para_end))

    for ci, ch in enumerate(book.chapters):
        if seg_opts.read_chapter_titles and ch.title.strip():
            add(ci, _heading_text(normalize_text(ch.title, norm_opts)), "heading", seg_opts.pause_heading_ms, True)

        # нормализуем абзацы и объединяем короткие реплики
        # элемент: (текст, смена_сцены_после, это_подзаголовок)
        paras: List[Tuple[str, bool, bool]] = []
        for p in ch.paragraphs:
            if p.strip() == "* * *":
                if paras:
                    paras[-1] = (paras[-1][0], True, paras[-1][2])
                continue
            if p.startswith(SUBHEAD_PREFIX):
                t = _heading_text(normalize_text(p[len(SUBHEAD_PREFIX):], norm_opts))
                if t:
                    paras.append((t, False, True))
                continue
            t = normalize_text(p, norm_opts)
            if t:
                paras.append((t, False, False))

        merged: List[Tuple[str, bool, bool]] = []
        for t, scene, sub in paras:
            if (
                seg_opts.merge_short_paragraphs
                and not sub
                and merged
                and not merged[-1][1]
                and not merged[-1][2]
                and (len(t) < seg_opts.min_chars or len(merged[-1][0]) < seg_opts.min_chars)
                and len(merged[-1][0]) < seg_opts.min_chars * 2.5
                and len(t) < seg_opts.min_chars * 2.5
                and len(merged[-1][0]) + 1 + len(t) <= seg_opts.target_chars
            ):
                merged[-1] = (merged[-1][0] + " " + t, scene, False)
            else:
                merged.append((t, scene, sub))

        for t, scene, sub in merged:
            if sub:
                # подзаголовок: пауза перед ним (у предыдущего сегмента) и после
                if segs and segs[-1].chapter == ci:
                    segs[-1].pause_ms = max(segs[-1].pause_ms, seg_opts.pause_heading_ms)
                add(ci, t, "heading", seg_opts.pause_heading_ms, True)
                continue
            parts = split_paragraph(t, seg_opts, lang)
            for k, part in enumerate(parts):
                last = k == len(parts) - 1
                if not last:
                    pause = seg_opts.pause_segment_ms
                elif scene:
                    pause = seg_opts.pause_scene_ms
                else:
                    pause = seg_opts.pause_paragraph_ms
                add(ci, part, "text", pause, last)

        if segs and segs[-1].chapter == ci:
            segs[-1].pause_ms = max(segs[-1].pause_ms, seg_opts.pause_chapter_end_ms)

    for i, s in enumerate(segs):
        s.idx = i
    return segs


def estimate_duration_sec(n_chars: int, chars_per_sec: float = 14.0) -> float:
    """Грубая оценка длительности аудио (русская начитка ≈ 13–15 симв./с)."""
    return n_chars / chars_per_sec
