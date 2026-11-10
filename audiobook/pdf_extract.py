"""Извлечение текста книги из PDF.

Главная задача — получить чистые *логические абзацы*, чтобы дальше ничего не
рвалось посередине предложения:

* удаляются колонтитулы и номера страниц (повторяющиеся строки в полях);
* удаляются сноски внизу страницы (мелкий шрифт) и маркеры сносок (верхний индекс);
* склеиваются переносы слов на концах строк («сло-\\nво» → «слово», но «кто-\\nто» → «кто-то»);
* абзацы склеиваются через границы страниц и колонок, если предложение не закончено;
* главы определяются по оглавлению PDF (outline), по крупному шрифту или по шаблонам
  («Глава 1», «Часть II», «Chapter 3», «Пролог» …).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .book import DEF_MARK, SUBHEAD_PREFIX, Book, Chapter


def _pymupdf():
    try:
        import pymupdf  # PyMuPDF >= 1.24

        return pymupdf
    except ImportError:  # старые версии
        import fitz

        return fitz

# --------------------------------------------------------------------------- #
# Регулярки и эвристики
# --------------------------------------------------------------------------- #

_HEADING_WORDS = (
    "глава|часть|книга|том|раздел|пролог|эпилог|предисловие|послесловие|введение|"
    "заключение|интерлюдия|приложение|от автора|вместо предисловия|"
    "chapter|part|book|prologue|epilogue|preface|foreword|introduction|afterword|"
    "conclusion|interlude|appendix"
)
_NUM_WORDS = (
    "перв|втор|трет|четв[её]рт|пят|шест|седьм|восьм|девят|десят|"
    "одиннадцат|двенадцат|тринадцат|четырнадцат|пятнадцат|шестнадцат|семнадцат|"
    "восемнадцат|девятнадцат|двадцат|тридцат|сорок|пятидесят|"
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
)
HEADING_RE = re.compile(
    rf"^\s*(?:{_HEADING_WORDS})\b\.?"
    rf"(?:\s+(?:[0-9]{{1,4}}|[IVXLCDM]{{1,8}}|(?:{_NUM_WORDS})[\w-]*)\b\.?)?"
    r"(?:\s*[.:—–-]?\s*.{0,70})?$",
    re.IGNORECASE,
)
ROMAN_OR_NUM_RE = re.compile(r"^\s*(?:[IVXLCDM]{1,8}|\d{1,3})\s*\.?\s*$")
SCENE_BREAK_RE = re.compile(r"^\s*(?:[*⁂✻✱•·#~]\s*){1,5}$|^\s*[-–—]{3,}\s*$")
PAGE_NUM_RE = re.compile(r"^\s*(?:[-–—]\s*)?(?:стр\.?|с\.|page|p\.)?\s*\d{1,4}\s*(?:[-–—]\s*)?$", re.I)
TOC_LEADER_RE = re.compile(r"(?:\.\s*){4,}\s*\d{1,4}\s*$|…{2,}\s*\d{1,4}\s*$")
SENT_END_RE = re.compile(r"[.!?…][»\"”’)\]]*\s*$")
_HYPHENS = "-‐‑­"
# Частицы, перед которыми дефис настоящий: кто-то, где-нибудь, скажи-ка, всё-таки
_KEEP_HYPHEN_RE = re.compile(r"^(?:то|либо|нибудь|ка|таки|де|с|тка|кась)(?![а-яё])", re.I)

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
    "​": "",
    "‌": "",
    "‍": "",
    "﻿": "",
    " ": " ",
    " ": " ",
    " ": " ",
}
_LIG_RE = re.compile("|".join(map(re.escape, _LIGATURES)))


def clean_raw(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return _LIG_RE.sub(lambda m: _LIGATURES[m.group(0)], s)


def ends_sentence(s: str) -> bool:
    return bool(SENT_END_RE.search(s))


_L1_WORDS = {"раздел", "часть", "книга", "том", "part", "book", "volume"}
SUBSECTION_RE = re.compile(r"^\s*(?:§\s*\d+|параграф\s+\d+)", re.IGNORECASE)
NUMBERED_SUB_RE = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.?\s+[A-ZА-ЯЁ]")
SERVICE_PAGE_RE = re.compile(r"\bISBN\b|\bУДК\b|\bББК\b|©")
NUM_EDGE_RE = re.compile(r"^\d{1,4}\s+\S.{8,}$|^.{8,}\S\s+\d{1,4}$")


def heading_level(s: str) -> int:
    """Уровень заголовка по тексту: 1 — Раздел/Часть/Том, 2 — Глава/Предисловие, 3 — §. 0 — не заголовок."""
    s = s.strip()
    if not s or len(s) > 120:
        return 0
    if SUBSECTION_RE.match(s):
        return 3 if not (len(s) > 100 and ends_sentence(s)) else 0
    if not looks_like_heading(s):
        return 0
    first = re.split(r"[\s.:]+", s.lower(), maxsplit=1)[0]
    return 1 if first in _L1_WORDS else 2


def looks_like_heading(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 90:
        return False
    if HEADING_RE.match(s):
        # «Глава» внутри обычной фразы («Глава семьи сказал, что…») — отсекаем
        # по длине и концевой пунктуации.
        return len(s) <= 60 or not ends_sentence(s)
    return False


# --------------------------------------------------------------------------- #
# Структуры
# --------------------------------------------------------------------------- #


@dataclass
class Line:
    text: str
    size: float
    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    bold: bool = False
    page_h: float = 0.0
    page_w: float = 0.0
    kind: str = "text"  # text | heading | subheading | scene
    level: int = 0  # для heading: 1 — раздел, 2 — глава, 3 — параграф
    cont: bool = False  # продолжение многострочного заголовка
    lead_emph: str = ""  # жирное/курсивное начало строки (термин определения)


@dataclass
class ExtractOptions:
    first_page: int = 1  # 1-based, включительно
    last_page: Optional[int] = None  # 1-based, включительно
    remove_headers: bool = True
    remove_footnotes: bool = True
    chapter_mode: str = "auto"  # auto | toc | font | regex | none
    toc_level: int = 3
    skip_toc_pages: bool = True  # выкидывать страницы оглавления
    skip_service_pages: bool = True  # УДК/ББК/ISBN/© — выходные данные
    margin_ratio: float = 0.085  # доля высоты страницы, считающаяся полем


@dataclass
class ExtractReport:
    pages: int = 0
    removed_headers: int = 0
    removed_footnotes: int = 0
    removed_pages: int = 0
    chapter_source: str = ""
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Чтение строк
# --------------------------------------------------------------------------- #


def _read_lines(doc, page_numbers: Iterable[int]) -> List[Line]:
    lines: List[Line] = []
    for pno in page_numbers:
        page = doc[pno]
        ph, pw = page.rect.height, page.rect.width
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            if b.get("type", 0) != 0:
                continue
            for ln in b.get("lines", []):
                dx, dy = ln.get("dir", (1, 0))
                if abs(dy) > 0.1 or dx < 0.9:  # повёрнутый текст (водяные знаки и т.п.)
                    continue
                spans = [s for s in ln.get("spans", []) if s.get("text")]
                if not spans:
                    continue
                max_size = max(s["size"] for s in spans)
                parts = []
                sizes = Counter()
                bold_chars = 0
                lead, lead_open = [], True
                for s in spans:
                    t = s["text"]
                    is_sup = bool(s.get("flags", 0) & 1) or s["size"] < max_size * 0.75
                    # маркеры сносок: ¹, *, 12 — мелким шрифтом/надстрочно
                    if is_sup and re.fullmatch(r"\s*[\d*†‡]{1,3}\s*", t) and len(spans) > 1:
                        continue
                    fl, fn = s.get("flags", 0), s.get("font", "").lower()
                    emph = bool(fl & 16 or fl & 2) or any(k in fn for k in ("bold", "italic", "oblique"))
                    if lead_open and (emph or not t.strip()):
                        lead.append(t)
                    elif t.strip():
                        lead_open = False
                    parts.append(t)
                    n = len(t.strip())
                    sizes[round(s["size"], 1)] += n
                    if s.get("flags", 0) & 16 or "bold" in s.get("font", "").lower():
                        bold_chars += n
                text = clean_raw("".join(parts))
                # надстрочные цифры-маркеры сносок: «слово¹» → «слово»
                text = re.sub(r"(?<=[^\W\d_]|[.,;:!?»…])[¹²³⁴⁵⁶⁷⁸⁹⁰]+", "", text)
                if not text.strip():
                    continue
                size = sizes.most_common(1)[0][0] if sizes else max_size
                x0, y0, x1, y1 = ln["bbox"]
                total = sum(sizes.values()) or 1
                lines.append(
                    Line(
                        text=re.sub(r"[ \t]+", " ", text).strip(),
                        size=size,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        page=pno,
                        bold=bold_chars / total > 0.6,
                        page_h=ph,
                        page_w=pw,
                        lead_emph="" if lead_open else re.sub(r"\s+", " ", clean_raw("".join(lead))).strip(),
                    )
                )
    return _merge_same_baseline(lines)


def _merge_same_baseline(lines: List[Line]) -> List[Line]:
    """PyMuPDF иногда отдаёт одну визуальную строку несколькими кусками."""
    out: List[Line] = []
    for ln in lines:
        if (
            out
            and out[-1].page == ln.page
            and abs(out[-1].y0 - ln.y0) < 0.35 * ln.size
            and abs(out[-1].y1 - ln.y1) < 0.35 * ln.size
            and 0 <= ln.x0 - out[-1].x1 < 3 * ln.size
        ):
            prev = out[-1]
            gap = " " if ln.x0 - prev.x1 > 0.15 * ln.size else ""
            prev.text = prev.text + gap + ln.text
            prev.x1 = ln.x1
            continue
        out.append(ln)
    return out


def _body_size(lines: List[Line]) -> float:
    c = Counter()
    for ln in lines:
        c[ln.size] += len(ln.text)
    return c.most_common(1)[0][0] if c else 10.0


# --------------------------------------------------------------------------- #
# Колонтитулы и сноски
# --------------------------------------------------------------------------- #


def _norm_key(s: str) -> str:
    s = re.sub(r"\d+", "#", s.lower())
    return re.sub(r"\s+", " ", s).strip(" .-–—")


def _remove_headers(lines: List[Line], opts: ExtractOptions, rep: ExtractReport) -> List[Line]:
    in_margin = []
    for ln in lines:
        top = ln.y1 < ln.page_h * opts.margin_ratio
        bottom = ln.y0 > ln.page_h * (1 - opts.margin_ratio)
        in_margin.append(top or bottom)

    pages_by_key = defaultdict(set)
    for ln, m in zip(lines, in_margin):
        if m:
            pages_by_key[_norm_key(ln.text)].add(ln.page)

    n_pages = len({ln.page for ln in lines}) or 1
    min_repeat = 3 if n_pages >= 6 else max(2, n_pages // 2 + 1)

    out = []
    for ln, m in zip(lines, in_margin):
        if m:
            key = _norm_key(ln.text)
            if PAGE_NUM_RE.match(ln.text):
                rep.removed_headers += 1
                continue
            # «20  Раздел I. Общие положения», «Глава 1. … право  19» — колонтитул с номером страницы
            if NUM_EDGE_RE.match(ln.text) and not ends_sentence(ln.text):
                rep.removed_headers += 1
                continue
            if key and len(pages_by_key[key]) >= min_repeat and (not ends_sentence(ln.text) or key == "#"):
                rep.removed_headers += 1
                continue
        out.append(ln)
    return out


def _remove_pages(lines: List[Line], opts: ExtractOptions, rep: ExtractReport) -> List[Line]:
    """Убирает страницы оглавления и выходных данных (УДК/ББК/ISBN)."""
    by_page = defaultdict(list)
    for ln in lines:
        by_page[ln.page].append(ln)
    drop = set()
    first_heading_page = None
    for p in sorted(by_page):
        ls = by_page[p]
        n = len(ls)
        leaders = sum(1 for l in ls if TOC_LEADER_RE.search(l.text))
        num_end = sum(1 for l in ls if re.search(r"\S\s+\d{1,4}$", l.text) and not ends_sentence(l.text))
        if opts.skip_toc_pages and ((leaders >= 3 and leaders >= 0.2 * n) or (num_end >= 6 and num_end >= 0.5 * n)):
            drop.add(p)
            continue
        if first_heading_page is None and any(heading_level(l.text) for l in ls):
            first_heading_page = p
        if opts.skip_service_pages and first_heading_page is None and any(SERVICE_PAGE_RE.search(l.text) for l in ls):
            drop.add(p)
    rep.removed_pages += len(drop)
    return [ln for ln in lines if ln.page not in drop]


def _remove_footnotes(lines: List[Line], body: float, rep: ExtractReport) -> List[Line]:
    by_page = defaultdict(list)
    for i, ln in enumerate(lines):
        by_page[ln.page].append(i)
    drop = set()
    for _, idxs in by_page.items():
        # последняя строка основного (не мелкого) текста на странице
        last_body_y = max((lines[i].y1 for i in idxs if lines[i].size >= body * 0.9), default=None)
        if last_body_y is None:
            continue
        for i in idxs:
            ln = lines[i]
            if ln.size < body * 0.88 and ln.y0 >= last_body_y - 1 and ln.y0 > ln.page_h * 0.5:
                drop.add(i)
    rep.removed_footnotes += len(drop)
    return [ln for i, ln in enumerate(lines) if i not in drop]


# --------------------------------------------------------------------------- #
# Заголовки и главы
# --------------------------------------------------------------------------- #


def _centered(ln: Line, left: Optional[float] = None) -> bool:
    """Строка выровнена по центру (а не просто короткая строка с абзацным отступом)."""
    sym = abs((ln.x0 + ln.x1) / 2 - ln.page_w / 2) < ln.page_w * 0.06
    narrow = (ln.x1 - ln.x0) < ln.page_w * 0.75
    if left is not None:
        # отступ слева заметно больше абзацного (≈ 1–1.5 em)
        return sym and narrow and ln.x0 - left > 2.2 * max(ln.size, 1.0)
    return sym and narrow


def _styled(ln: Line, body: float, left: Optional[float] = None) -> bool:
    """Строка оформлена как заголовок: жирная, крупнее, или по центру."""
    return ln.bold or ln.size >= body * 1.1 or _centered(ln, left)


def _gap_above(lines: List[Line], i: int) -> float:
    ln = lines[i]
    if i == 0 or lines[i - 1].page != ln.page:
        return 99.0
    return ln.y0 - lines[i - 1].y1


def _mark_headings(lines: List[Line], body: float, use_font: bool, use_regex: bool) -> None:
    margins = _page_margins(lines)
    for i, ln in enumerate(lines):
        left = margins.get(ln.page, (None, None))[0]
        t = ln.text
        if ln.kind == "heading":
            continue
        if SCENE_BREAK_RE.match(t):
            ln.kind = "scene"
            continue
        if len(t) > 120:
            continue
        big = ln.size >= body * 1.18
        lvl = heading_level(t) if use_regex else 0
        # Строка «Глава 1 ГК РФ …» внутри абзаца — не заголовок: требуем оформления
        if lvl and (_styled(ln, body, left) or _isolated(lines, i)):
            ln.kind, ln.level = "heading", lvl
        elif use_font and big and len(t) <= 100 and not TOC_LEADER_RE.search(t):
            ln.kind, ln.level = "heading", heading_level(t) or 2
        elif use_font and ROMAN_OR_NUM_RE.match(t) and (big or ln.bold):
            ln.kind, ln.level = "heading", 2
        elif use_regex and ROMAN_OR_NUM_RE.match(t) and _isolated(lines, i):
            ln.kind, ln.level = "heading", 2
        elif (
            use_regex
            and ln.bold
            and len(t) <= 90
            and not ends_sentence(t)
            and (_centered(ln, left) or NUMBERED_SUB_RE.match(t))
            and _gap_above(lines, i) > 0.3 * body
        ):
            ln.kind = "subheading"  # «Основные понятия», «1.1. Основные положения»
    _mark_heading_continuations(lines, body, margins)


def _mark_heading_continuations(lines: List[Line], body: float, margins: dict) -> None:
    """«Глава 1. Предпринимательство» + «и предпринимательское право» — один заголовок."""
    for i in range(1, len(lines)):
        prev, ln = lines[i - 1], lines[i]
        if prev.kind not in ("heading", "subheading") or ln.kind != "text" or prev.page != ln.page:
            continue
        h = max(prev.y1 - prev.y0, body)
        gap = ln.y0 - prev.y1
        left = margins.get(ln.page, (ln.x0, None))[0]
        indented_like_heading = ln.x0 - left > 2.2 * body or not _centered(prev)
        same_style = (ln.bold == prev.bold) and ln.size >= prev.size * 0.9 and indented_like_heading
        if (
            same_style
            and len(ln.text) <= 100
            and gap < 0.9 * h
            and not ends_sentence(prev.text)
            and (ln.text[:1].islower() or gap < 0.45 * h)
        ):
            ln.kind, ln.level, ln.cont = prev.kind, prev.level, True


def _isolated(lines: List[Line], i: int) -> bool:
    """Строка «IV» или «12» стоит отдельно (с отступами сверху и снизу)."""
    ln = lines[i]
    h = max(ln.y1 - ln.y0, 1.0)
    prev_gap = ln.y0 - lines[i - 1].y1 if i > 0 and lines[i - 1].page == ln.page else 99
    next_gap = lines[i + 1].y0 - ln.y1 if i + 1 < len(lines) and lines[i + 1].page == ln.page else 99
    centered = abs((ln.x0 + ln.x1) / 2 - ln.page_w / 2) < ln.page_w * 0.12
    return prev_gap > 0.8 * h and next_gap > 0.8 * h and centered


def _norm_title(s: str) -> str:
    s = re.sub(r"[^\w]+", " ", s.lower()).strip()
    return re.sub(r"\s+", " ", s)


def _apply_toc(doc, lines: List[Line], level: int) -> bool:
    """Размечает заголовки по закладкам PDF (outline). Возвращает True при успехе."""
    toc = [e for e in doc.get_toc(simple=True) if e[0] <= level and e[2] >= 1]
    if len(toc) < 2:
        return False
    for ln in lines:
        if ln.kind == "heading":
            ln.kind, ln.level, ln.cont = "text", 0, False
    by_page = defaultdict(list)
    for i, ln in enumerate(lines):
        by_page[ln.page].append(i)
    used = set()
    found = 0
    for toc_lvl, title, page1 in toc:
        p = page1 - 1
        nt = _norm_title(title)
        if not nt:
            continue
        lvl = heading_level(title) or min(3, toc_lvl)
        cand = by_page.get(p, [])[:20] + by_page.get(p + 1, [])[:6]
        for i in cand:
            if i in used:
                continue
            lt = _norm_title(lines[i].text)
            if lt and (lt == nt or nt.startswith(lt) or lt.startswith(nt)):
                lines[i].kind, lines[i].level = "heading", lvl
                used.add(i)
                found += 1
                # Многострочный заголовок: следующие строки, продолжающие название
                j = i + 1
                acc = lt
                while j < len(lines) and lines[j].page == lines[i].page and len(acc) < len(nt):
                    nxt = _norm_title(lines[j].text)
                    if nxt and nt.startswith((acc + " " + nxt).strip()):
                        lines[j].kind, lines[j].level, lines[j].cont = "heading", lvl, True
                        used.add(j)
                        acc = (acc + " " + nxt).strip()
                        j += 1
                    else:
                        break
                break
    return found >= max(2, len(toc) // 3)


# --------------------------------------------------------------------------- #
# Сборка абзацев
# --------------------------------------------------------------------------- #


def join_lines(a: str, b: str) -> str:
    """Склейка двух строк с корректной обработкой переносов."""
    if not a:
        return b
    if not b:
        return a
    if a[-1] == "­":
        return a[:-1] + b
    if a[-1] in _HYPHENS and len(a) >= 2 and a[-2].isalpha():
        first = b[0]
        if first.isalpha() and first.islower():
            if _KEEP_HYPHEN_RE.match(b):
                return a[:-1] + "-" + b
            return a[:-1] + b
        return a + b  # «северо-Западный», «1990-х»: оставляем дефис
    return a + " " + b


def _page_margins(lines: List[Line]):
    """Левая граница текста и правая граница (90-й перцентиль) для каждой страницы."""
    by_page = defaultdict(list)
    for ln in lines:
        if ln.kind == "text":
            by_page[ln.page].append(ln)
    res = {}
    for p, ls in by_page.items():
        xs0 = sorted(round(l.x0) for l in ls)
        xs1 = sorted(l.x1 for l in ls)
        left = Counter(xs0).most_common(1)[0][0]
        right = xs1[int(len(xs1) * 0.9)] if len(xs1) > 2 else xs1[-1]
        res[p] = (left, right)
    return res


DEF_REST_RE = re.compile(r"^\s*(\([^)]{1,120}\))?\s*[—–]\s")


def _mark_definition(ln: Line) -> str:
    """«**Бизнес** — деятельность…»: термин выделен жирным/курсивом и за ним (скобки) и тире."""
    term = ln.lead_emph.rstrip(" —–-:")
    if not (2 <= len(term) <= 80) or not ln.text.startswith(term) or ln.bold and len(term) > 0.8 * len(ln.text):
        return ln.text
    rest = ln.text[len(term):]
    if DEF_REST_RE.match(rest):
        return f"{DEF_MARK}{term}{DEF_MARK}{rest}"
    return ln.text


def _build(lines: List[Line], body: float) -> List[tuple]:
    """Возвращает список элементов [вид, текст, уровень]: heading | subheading | par | scene."""
    margins = _page_margins(lines)
    items: List[list] = []
    prev: Optional[Line] = None

    def new_par(ln: Line) -> bool:
        if prev is None or prev.kind != "text" or not items or items[-1][0] != "par":
            return True
        left, right = margins.get(ln.page, (ln.x0, ln.x1))
        pleft, pright = margins.get(prev.page, (prev.x0, prev.x1))
        indent = ln.x0 - left > 0.8 * body
        prev_short = prev.x1 < pright - 2.5 * body
        prev_end = ends_sentence(prev.text) or prev.text.endswith(":")
        starts_dialog = ln.text[:1] in "—–-" and ln.text[1:2] == " "
        same_flow = prev.page == ln.page and ln.y0 >= prev.y0
        if same_flow:
            h = max(prev.y1 - prev.y0, body)
            gap = ln.y0 - prev.y1
            if gap > 0.75 * h and prev_end:
                return True
            if not prev_end:
                return False  # предложение не закончено — тот же абзац
            return indent or prev_short or starts_dialog
        # переход на новую страницу/колонку
        if not prev_end:
            return False
        return indent or prev_short or starts_dialog

    for ln in lines:
        if ln.kind in ("heading", "subheading"):
            if ln.cont and items and items[-1][0] == ln.kind:
                items[-1][1] = join_lines(items[-1][1], ln.text)
            else:
                items.append([ln.kind, ln.text, ln.level])
        elif ln.kind == "scene":
            items.append(["scene", "* * *", 0])
        else:
            if TOC_LEADER_RE.search(ln.text):  # строка оглавления «Глава 1 ....... 5»
                prev = ln
                continue
            if new_par(ln):
                items.append(["par", _mark_definition(ln), 0])
            else:
                items[-1][1] = join_lines(items[-1][1], ln.text)
        prev = ln
    return [tuple(x) for x in items]


def _to_book(items: List[tuple], split_chapters: bool) -> List[Chapter]:
    chapters: List[Chapter] = []
    for kind, text, level in items:
        if kind == "heading" and split_chapters:
            chapters.append(Chapter(title=text, level=level or 2))
            continue
        if not chapters:
            chapters.append(Chapter(title="", level=2))
        if kind == "scene":
            if chapters[-1].paragraphs and chapters[-1].paragraphs[-1] != "* * *":
                chapters[-1].paragraphs.append("* * *")
        elif kind in ("subheading", "heading"):
            chapters[-1].paragraphs.append(SUBHEAD_PREFIX + text)
        else:
            chapters[-1].paragraphs.append(text)
    for ch in chapters:
        while ch.paragraphs and (ch.paragraphs[-1] == "* * *" or ch.paragraphs[-1].startswith(SUBHEAD_PREFIX)):
            ch.paragraphs.pop()
    return [c for c in chapters if c.paragraphs or c.title]


# --------------------------------------------------------------------------- #
# Публичное API
# --------------------------------------------------------------------------- #


def extract_pdf(path: str, opts: Optional[ExtractOptions] = None) -> tuple[Book, ExtractReport]:
    opts = opts or ExtractOptions()
    rep = ExtractReport()
    doc = _pymupdf().open(path)
    try:
        n = doc.page_count
        first = max(1, opts.first_page) - 1
        last = min(n, opts.last_page or n)
        pages = range(first, last)
        rep.pages = len(pages)

        lines = _read_lines(doc, pages)
        total_chars = sum(len(l.text) for l in lines)
        if total_chars < 40 * max(1, rep.pages):
            rep.warnings.append(
                "В PDF почти нет текстового слоя — похоже на скан. Сначала распознайте его "
                "(например, OCRmyPDF или ABBYY FineReader), затем загрузите снова."
            )
        if not lines:
            return Book(title=doc.metadata.get("title") or "", source=path), rep

        body = _body_size(lines)
        lines = _remove_pages(lines, opts, rep)
        if opts.remove_headers:
            lines = _remove_headers(lines, opts, rep)
        if opts.remove_footnotes:
            lines = _remove_footnotes(lines, body, rep)

        mode = opts.chapter_mode
        if mode in ("auto", "toc"):
            _mark_headings(lines, body, use_font=False, use_regex=False)  # только scene-break
            if _apply_toc(doc, lines, opts.toc_level):
                rep.chapter_source = "закладки (оглавление) PDF"
                _mark_headings(lines, body, use_font=False, use_regex=True)  # подзаголовки и §, которых нет в закладках
            elif mode == "toc":
                rep.warnings.append("Оглавление PDF не найдено или не совпало с текстом — использую шрифт и шаблоны.")
                mode = "auto"
            if not rep.chapter_source:
                _mark_headings(lines, body, use_font=True, use_regex=True)
                rep.chapter_source = "шрифт и шаблоны «Глава N»"
        elif mode == "font":
            _mark_headings(lines, body, use_font=True, use_regex=False)
            rep.chapter_source = "размер шрифта"
        elif mode == "regex":
            _mark_headings(lines, body, use_font=False, use_regex=True)
            rep.chapter_source = "шаблоны «Глава N»"
        else:
            _mark_headings(lines, body, use_font=False, use_regex=False)
            rep.chapter_source = "без разбиения"

        items = _build(lines, body)
        chapters = _to_book(items, split_chapters=mode != "none")

        # Слишком много «глав» — признак ложных срабатываний по шрифту
        if mode == "auto" and len([c for c in chapters if c.level <= 2]) > max(60, rep.pages // 2):
            rep.warnings.append(
                f"Найдено подозрительно много глав ({len(chapters)}). "
                "Попробуйте режим «шаблоны» или «оглавление»."
            )

        meta = doc.metadata or {}
        book = Book(
            title=(meta.get("title") or "").strip(),
            author=(meta.get("author") or "").strip(),
            chapters=chapters,
            source=str(path),
        )
        return book, rep
    finally:
        doc.close()


def extract_any(path: str, opts: Optional[ExtractOptions] = None) -> tuple[Book, ExtractReport]:
    """PDF или текстовый файл (.txt/.md в формате правки)."""
    low = str(path).lower()
    if low.endswith(".pdf"):
        return extract_pdf(path, opts)
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1251", "koi8-r", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    rep = ExtractReport(chapter_source="текстовый файл")
    return Book.from_text(clean_raw(text)), rep


def render_cover(path: str, out_path: str, zoom: float = 1.5) -> Optional[str]:
    """Рендер первой страницы PDF в JPEG для обложки аудиокниги."""
    try:
        fitz = _pymupdf()
        doc = fitz.open(path)
        try:
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            pix.save(out_path)
        finally:
            doc.close()
        return out_path
    except Exception:
        return None
