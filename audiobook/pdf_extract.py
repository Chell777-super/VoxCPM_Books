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

from .book import Book, Chapter


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
    kind: str = "text"  # text | heading | scene


@dataclass
class ExtractOptions:
    first_page: int = 1  # 1-based, включительно
    last_page: Optional[int] = None  # 1-based, включительно
    remove_headers: bool = True
    remove_footnotes: bool = True
    chapter_mode: str = "auto"  # auto | toc | font | regex | none
    toc_level: int = 1
    margin_ratio: float = 0.085  # доля высоты страницы, считающаяся полем


@dataclass
class ExtractReport:
    pages: int = 0
    removed_headers: int = 0
    removed_footnotes: int = 0
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
                for s in spans:
                    t = s["text"]
                    is_sup = bool(s.get("flags", 0) & 1) or s["size"] < max_size * 0.75
                    # маркеры сносок: ¹, *, 12 — мелким шрифтом/надстрочно
                    if is_sup and re.fullmatch(r"\s*[\d*†‡]{1,3}\s*", t) and len(spans) > 1:
                        continue
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
            if key and len(pages_by_key[key]) >= min_repeat and (not ends_sentence(ln.text) or key == "#"):
                rep.removed_headers += 1
                continue
        out.append(ln)
    return out


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


def _mark_headings(lines: List[Line], body: float, use_font: bool, use_regex: bool) -> None:
    for i, ln in enumerate(lines):
        t = ln.text
        if SCENE_BREAK_RE.match(t):
            ln.kind = "scene"
            continue
        if len(t) > 120:
            continue
        big = ln.size >= body * 1.18
        if use_regex and looks_like_heading(t):
            ln.kind = "heading"
        elif use_font and big and len(t) <= 100 and not TOC_LEADER_RE.search(t):
            ln.kind = "heading"
        elif use_font and ROMAN_OR_NUM_RE.match(t) and (big or ln.bold):
            ln.kind = "heading"
        elif use_regex and ROMAN_OR_NUM_RE.match(t) and _isolated(lines, i):
            ln.kind = "heading"


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
    """Размечает начала глав по оглавлению PDF. Возвращает True при успехе."""
    toc = [e for e in doc.get_toc(simple=True) if e[0] <= level and e[2] >= 1]
    if len(toc) < 2:
        return False
    for ln in lines:
        if ln.kind == "heading":
            ln.kind = "text"
    by_page = defaultdict(list)
    for i, ln in enumerate(lines):
        by_page[ln.page].append(i)
    used = set()
    found = 0
    for _, title, page1 in toc:
        p = page1 - 1
        nt = _norm_title(title)
        if not nt:
            continue
        cand = by_page.get(p, [])[:12] + by_page.get(p + 1, [])[:4]
        for i in cand:
            if i in used:
                continue
            lt = _norm_title(lines[i].text)
            if lt and (lt == nt or nt.startswith(lt) or lt.startswith(nt)):
                lines[i].kind = "heading"
                used.add(i)
                found += 1
                # Многострочный заголовок: следующие строки, продолжающие название
                j = i + 1
                acc = lt
                while j < len(lines) and lines[j].page == lines[i].page and len(acc) < len(nt):
                    nxt = _norm_title(lines[j].text)
                    if nxt and nt.startswith((acc + " " + nxt).strip()):
                        lines[j].kind = "heading"
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


def _build(lines: List[Line], body: float) -> List[tuple]:
    """Возвращает список элементов ("heading"|"par"|"scene", text)."""
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
        if ln.kind == "heading":
            if items and items[-1][0] == "heading" and prev is not None and prev.kind == "heading" and prev.page == ln.page:
                items[-1][1] = items[-1][1].rstrip(".") + ". " + ln.text
            else:
                items.append(["heading", ln.text])
        elif ln.kind == "scene":
            items.append(["scene", "* * *"])
        else:
            if TOC_LEADER_RE.search(ln.text):  # строка оглавления «Глава 1 ....... 5»
                prev = ln
                continue
            if new_par(ln):
                items.append(["par", ln.text])
            else:
                items[-1][1] = join_lines(items[-1][1], ln.text)
        prev = ln
    return [tuple(x) for x in items]


def _to_book(items: List[tuple], split_chapters: bool) -> List[Chapter]:
    chapters: List[Chapter] = []
    for kind, text in items:
        if kind == "heading" and split_chapters:
            # Подряд идущие заголовки без текста («Часть I» + «Глава 1») — объединяем
            if chapters and not chapters[-1].paragraphs and chapters[-1].title:
                chapters[-1].title = chapters[-1].title.rstrip(".") + ". " + text
            else:
                chapters.append(Chapter(title=text))
            continue
        if not chapters:
            chapters.append(Chapter(title=""))
        if kind == "scene":
            if chapters[-1].paragraphs and chapters[-1].paragraphs[-1] != "* * *":
                chapters[-1].paragraphs.append("* * *")
        else:
            chapters[-1].paragraphs.append(text)
    for ch in chapters:
        while ch.paragraphs and ch.paragraphs[-1] == "* * *":
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
        if opts.remove_headers:
            lines = _remove_headers(lines, opts, rep)
        if opts.remove_footnotes:
            lines = _remove_footnotes(lines, body, rep)

        mode = opts.chapter_mode
        if mode in ("auto", "toc"):
            _mark_headings(lines, body, use_font=False, use_regex=False)  # только scene-break
            if _apply_toc(doc, lines, opts.toc_level):
                rep.chapter_source = "оглавление PDF"
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
        if mode == "auto" and len(chapters) > max(60, rep.pages // 2):
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
