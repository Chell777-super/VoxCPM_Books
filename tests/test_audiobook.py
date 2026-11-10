"""Тесты разбора PDF, нормализации и сегментации (без GPU и без модели)."""

import os
import re

import numpy as np
import pytest

from audiobook.book import Book, Chapter
from audiobook.normalize import NormalizeOptions, normalize_text, roman_to_int
from audiobook.pdf_extract import ExtractOptions, extract_pdf, join_lines, looks_like_heading
from audiobook.segmenter import SegmentOptions, segment_book, split_long, split_paragraph, split_sentences

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
FONT = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)


# --------------------------------------------------------------------------- #
# Синтетический «книжный» PDF
# --------------------------------------------------------------------------- #


def _make_book_pdf(path):
    pymupdf = pytest.importorskip("pymupdf")
    if FONT is None:
        pytest.skip("нет TTF-шрифта с кириллицей")
    doc = pymupdf.open()
    W, H = 420, 600
    L = 50
    body = 11
    lh = 15

    def page_with_furniture(num):
        pg = doc.new_page(width=W, height=H)
        pg.insert_font(fontname="F", fontfile=FONT)
        pg.insert_text((L, 30), "Тестовая книга. Автор Тестов", fontsize=8, fontname="F")
        pg.insert_text((W / 2 - 5, H - 20), str(num), fontsize=8, fontname="F")
        return pg

    def line(pg, y, text, x=L, size=body):
        pg.insert_text((x, y), text, fontsize=size, fontname="F")
        return y + lh

    p1 = page_with_furniture(1)
    y = 90
    y = line(p1, y, "Глава 1", x=150, size=18) + 20
    y = line(p1, y, "Солнце медленно поднималось над рекой, и туман", x=L + 15)
    y = line(p1, y, "стелился по воде. Старый рыбак сидел на берегу и сма-")
    y = line(p1, y, "тывал удочку. Кто-нибудь другой давно бы ушёл домой.")
    y = line(p1, y, "— Доброе утро! — сказал мальчик, подходя ближе.", x=L + 15)
    y = line(p1, y, "Рыбак кивнул и улыбнулся¹. Он помнил, как когда-", x=L + 15)
    y = line(p1, y, "то сам приходил сюда с отцом, и как они вместе ждали")
    y = line(p1, y, "первого клёва, и как отец рассказывал ему истории о")
    # сноска внизу страницы мелким шрифтом
    line(p1, H - 60, "¹ Это сноска, её не нужно читать вслух.", size=7)

    p2 = page_with_furniture(2)
    y = 90
    y = line(p2, y, "далёких морях и больших кораблях. Мальчик слушал")
    y = line(p2, y, "внимательно, т. е. не перебивая.")
    y = line(p2, y, "Прошло много лет; всё изменилось.", x=L + 15)
    y += 25
    y = line(p2, y, "Глава 2", x=150, size=18) + 20
    y = line(p2, y, "Вечером пошёл дождь. В XIX веке здесь стояла мельница.", x=L + 15)
    doc.save(path)
    doc.close()


@pytest.fixture()
def book_pdf(tmp_path):
    p = tmp_path / "book.pdf"
    _make_book_pdf(str(p))
    return str(p)


def test_extract_pdf_structure(book_pdf):
    book, rep = extract_pdf(book_pdf, ExtractOptions())
    titles = [c.title for c in book.chapters]
    assert titles == ["Глава 1", "Глава 2"], titles
    text = "\n".join(book.chapters[0].paragraphs)
    # колонтитулы и номера страниц удалены
    assert "Тестовая книга" not in text
    assert not re.search(r"(^|\n)\d+($|\n)", text)
    # сноска удалена, маркер сноски тоже
    assert "сноска" not in text
    assert "¹" not in text
    # перенос слова склеен, настоящие дефисы сохранены
    assert "сматывал" in text
    assert "Кто-нибудь" in text
    assert "когда-то" in text
    # абзац склеен через границу страниц
    assert any("истории о далёких морях" in p for p in book.chapters[0].paragraphs)
    # диалог — отдельный абзац
    assert any(p.startswith("— Доброе утро!") for p in book.chapters[0].paragraphs)
    assert rep.removed_headers >= 2


def test_full_pipeline_segments(book_pdf):
    book, _ = extract_pdf(book_pdf)
    segs = segment_book(book, SegmentOptions(target_chars=120, max_chars=200, min_chars=30))
    assert segs[0].kind == "heading" and segs[0].text == "Глава 1."
    for s in segs:
        assert len(s.text) <= 200
        assert not s.text.startswith("(")
    joined = " ".join(s.text for s in segs)
    assert "то есть" in joined  # сокращение раскрыто
    assert "19 веке" in joined  # римское число века
    # последний сегмент каждой главы имеет длинную паузу
    last_ch0 = [s for s in segs if s.chapter == 0][-1]
    assert last_ch0.pause_ms >= 2000


# --------------------------------------------------------------------------- #
# Склейка строк, заголовки
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("сма-", "тывал удочку", "сматывал удочку"),
        ("кто-", "то пришёл", "кто-то пришёл"),
        ("где-", "нибудь там", "где-нибудь там"),
        ("северо-", "Западный", "северо-Западный"),
        ("слово", "дальше", "слово дальше"),
        ("мяг\u00ad", "кий", "мягкий"),
    ],
)
def test_join_lines(a, b, expected):
    assert join_lines(a, b) == expected


@pytest.mark.parametrize(
    "s,ok",
    [
        ("Глава 1", True),
        ("ГЛАВА ПЕРВАЯ", True),
        ("Глава IV. Встреча", True),
        ("Часть вторая", True),
        ("Пролог", True),
        ("Chapter 12", True),
        ("Глава семьи сказал, что ужин будет поздно, и все разошлись по комнатам.", False),
        ("Обычная строка текста.", False),
    ],
)
def test_heading(s, ok):
    assert looks_like_heading(s) is ok


# --------------------------------------------------------------------------- #
# Нормализация
# --------------------------------------------------------------------------- #


def test_normalize_basic():
    t = normalize_text("Он пришёл (как обычно) поздно, т.е. в 10 ч. См. стр. 5[12].")
    assert "(" not in t and ")" not in t
    assert "то есть" in t
    assert "[12]" not in t
    assert "смотри" in t.lower()


def test_normalize_leading_bracket_is_not_voice_instruction():
    t = normalize_text("(Примечание автора) Текст продолжается.")
    assert not t.startswith("(")
    assert not t.startswith("—")


def test_normalize_caps():
    assert normalize_text("ГЛАВА ПЕРВАЯ. ВСТРЕЧА") == "Глава первая. Встреча"
    assert "СССР" in normalize_text("В СССР было много заводов.")


def test_roman():
    assert roman_to_int("XIX") == 19
    assert roman_to_int("IV") == 4
    assert roman_to_int("ABC") is None
    assert normalize_text("Глава XII") == "Глава 12"


def test_numbers_to_words_optional():
    pytest.importorskip("num2words")
    t = normalize_text("У него было 3 яблока.", NormalizeOptions(numbers_to_words=True))
    assert "три" in t


# --------------------------------------------------------------------------- #
# Предложения и сегменты
# --------------------------------------------------------------------------- #


def test_sentences_dialog_and_initials():
    sents = split_sentences("— Привет! — сказал он. А. С. Пушкин родился в Москве. Это было давно…")
    assert sents[0] == "— Привет! — сказал он."
    assert sents[1].startswith("А. С. Пушкин")
    assert len(sents) == 3


def test_split_long_prefers_strong_boundaries():
    s = (
        "Он долго шёл по дороге, которая вилась между холмами и полями; "
        "солнце уже садилось, и тени становились длиннее, а ветер приносил запах дыма."
    )
    parts = split_long(s, 90)
    assert parts[0][0].endswith(";")
    assert all(len(p) <= 90 for p, _ in parts)
    assert " ".join(p for p, _ in parts) == s


def test_split_paragraph_never_breaks_sentence_when_fits():
    sents = [f"Это предложение номер {i}, и в нём есть немного текста для проверки." for i in range(12)]
    par = " ".join(sents)
    opts = SegmentOptions(target_chars=200, max_chars=320, min_chars=40)
    parts = split_paragraph(par, opts)
    assert " ".join(parts) == par
    for p in parts:
        assert p.endswith(".")
        assert len(p) <= 320
    # без крошечных хвостов
    assert min(len(p) for p in parts) > 100


def test_short_dialog_paragraphs_merged():
    book = Book(chapters=[Chapter(title="", paragraphs=["— Да.", "— Нет.", "— Почему?", "Длинный абзац " * 10])])
    segs = segment_book(book, SegmentOptions(min_chars=40, target_chars=200, max_chars=300))
    assert segs[0].text.startswith("— Да. — Нет.")


def test_text_roundtrip():
    book = Book(title="Т", author="А", chapters=[Chapter("Глава 1", ["Абзац один.", "Абзац два."]), Chapter("Глава 2", ["Три."])])
    b2 = Book.from_text(book.to_text())
    assert b2.title == "Т" and b2.author == "А"
    assert [c.title for c in b2.chapters] == ["Глава 1", "Глава 2"]
    assert b2.chapters[0].paragraphs == ["Абзац один.", "Абзац два."]


def test_text_without_markup_detects_chapters():
    txt = "Глава 1\n\nПервый абзац\nпродолжение.\n\nВторой.\n\nГлава 2\n\nТретий."
    b = Book.from_text(txt)
    assert [c.title for c in b.chapters] == ["Глава 1", "Глава 2"]
    assert b.chapters[0].paragraphs == ["Первый абзац продолжение.", "Второй."]


# --------------------------------------------------------------------------- #
# Аудио
# --------------------------------------------------------------------------- #


def test_postprocess_and_assemble(tmp_path):
    sf = pytest.importorskip("soundfile")
    from audiobook.audio import assemble_chapter, postprocess, write_segment

    sr = 48000
    t = np.arange(sr) / sr
    tone = 0.05 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    wav = np.concatenate([np.zeros(sr // 2, np.float32), tone, np.zeros(sr // 2, np.float32)])
    out = postprocess(wav, sr)
    assert 0.9 * sr < len(out) < 1.3 * sr  # тишина по краям обрезана
    assert np.max(np.abs(out)) <= 10 ** (-1.5 / 20) + 1e-6
    p = tmp_path / "a.flac"
    write_segment(p, out, sr)
    dur, marks = assemble_chapter([(str(p), 500, "А"), (str(p), 0, "Б")], tmp_path / "ch.wav", sr, lead_in_ms=0)
    assert abs(dur - (2 * len(out) / sr + 0.5)) < 0.01
    assert marks[0] == ("А", 0.0) and abs(marks[1][1] - (len(out) / sr + 0.5)) < 0.01
    assert sf.info(str(tmp_path / "ch.wav")).samplerate == sr


def test_project_resume_keys(tmp_path, book_pdf):
    from audiobook.pipeline import parse_into_project
    from audiobook.project import Project

    proj = Project.create(tmp_path / "projects", book_pdf, name="Тест")
    info = parse_into_project(proj)
    assert info["total"] > 3
    s0 = proj.segments[1]
    proj.seg_path(s0).write_bytes(b"x")  # имитируем готовый сегмент
    proj.refresh_status()
    assert proj.segments[1].status == "done"
    # повторный разбор — готовый сегмент подхватился по хэшу
    parse_into_project(proj)
    assert any(s.status == "done" and s.text == s0.text for s in proj.segments)
    # смена голоса — всё нужно озвучивать заново
    proj.settings.voice.style = "другой стиль"
    proj.rekey_for_voice()
    assert all(s.status == "pending" for s in proj.segments)
    # перезагрузка проекта с диска
    p2 = Project(proj.root)
    assert len(p2.segments) == len(proj.segments)
    assert p2.settings.voice.style == "другой стиль"


@pytest.mark.parametrize(
    "src,expected",
    [
        ("Это случилось в 1812 г.", "Это случилось в 1812 году."),
        ("В XIX в. здесь жили люди.", "В 19 веке здесь жили люди."),
        ("Длина 5 см. См. рис. 3.", "Длина 5 сантиметров. Смотри рисунок 3."),
        ("Стоит 300 руб. и т.д. Потом", "Стоит 300 рублей и так далее. Потом"),
        ("Ул. Ленина, д. 5", "Улица Ленина, дом 5"),
        ("в 1941–1945 гг. шла война", "в 1941–1945 годы шла война"),
    ],
)
def test_normalize_units_and_abbr(src, expected):
    assert normalize_text(src) == expected


# --------------------------------------------------------------------------- #
# Учебник: Раздел → Глава → §, оглавление, колонтитулы с номерами страниц
# --------------------------------------------------------------------------- #

FONT_BOLD = next(
    (f for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "C:/Windows/Fonts/timesbd.ttf"] if os.path.exists(f)),
    None,
)

_BODY = (
    "Предпринимательство — одна из форм деятельности человека. Деятельность человека — форма "
    "проявления его активности, определенное поведение; в более широком плане — использование "
    "жизненных сил, человеческой энергии. Предпринимать — значит начать делать что-либо, "
    "приступить к чему-либо. Когда мы говорим «предприимчивый человек», то имеем в виду умеющего "
    "сделать что-либо в нужный момент, находчивого, изобретательного, практичного человека."
)


def _wrap(text, width=62):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines


def _make_textbook_pdf(path):
    pymupdf = pytest.importorskip("pymupdf")
    if FONT is None or FONT_BOLD is None:
        pytest.skip("нет TTF-шрифтов")
    doc = pymupdf.open()
    W, H, L = 420, 600, 40
    lh = 13

    def page(header=None):
        pg = doc.new_page(width=W, height=H)
        pg.insert_font(fontname="F", fontfile=FONT)
        pg.insert_font(fontname="B", fontfile=FONT_BOLD)
        if header:
            pg.insert_text((L, 25), header, fontsize=7, fontname="F")
        return pg

    def text(pg, y, s, size=9.5, bold=False, x=L, center=False):
        if center:
            x = (W - pymupdf.Font(fontfile=FONT_BOLD if bold else FONT).text_length(s, fontsize=size)) / 2
        pg.insert_text((x, y), s, fontsize=size, fontname="B" if bold else "F")
        return y + lh

    def para(pg, y, s):
        for k, ln in enumerate(_wrap(s)):
            y = text(pg, y, ln, x=L + (12 if k == 0 else 0))
        return y

    def definition(pg, y, term, rest):
        """Термин жирным, остальное обычным — как в глоссарии учебника."""
        x = L + 12
        pg.insert_text((x, y), term, fontsize=9.5, fontname="B")
        x += pymupdf.Font(fontfile=FONT_BOLD).text_length(term + " ", fontsize=9.5)
        lines = _wrap(rest, 50)
        pg.insert_text((x, y), lines[0], fontsize=9.5, fontname="F")
        y += lh
        for ln in _wrap(" ".join(lines[1:]), 62) if len(lines) > 1 else []:
            y = text(pg, y, ln)
        return y

    # 1. выходные данные
    p = page()
    text(p, 80, "УДК 351.824.1  ББК 67.404.91  ISBN 978-5-00156-")
    text(p, 100, "Учебник подготовлен на кафедре предпринимательского права.")
    # 2. оглавление (страница с точками-заполнителями)
    p = page()
    y = text(p, 60, "Оглавление", bold=True, center=True) + 10
    for s in [
        "Раздел I. Общие положения",
        "Глава 1. Предпринимательство и предпринимательское право ..... 18",
        "§ 1. Понятие предпринимательства и предпринимательской",
        "деятельности .................................................. 20",
        "§ 2. Предпринимательское право ............................ 40",
        "Глава 2. Источники предпринимательского права ........ 78",
        "Глава 3. Право на занятие предпринимательской деятельностью.",
        "Правовой статус предпринимателя ....................... 114",
    ]:
        y = text(p, y, s)
    # 3. Раздел I + Глава 1 (две строки) + Основные понятия
    p = page()
    y = text(p, 90, "Раздел I. Общие положения", size=11, bold=True, center=True) + 14
    y = text(p, y, "Глава 1. Предпринимательство", size=10.5, bold=True, center=True)
    y = text(p, y, "и предпринимательское право", size=10.5, bold=True, center=True) + 12
    y = text(p, y, "Основные понятия", bold=True, center=True) + 4
    y = definition(p, y, "Бизнес —", "деятельность, осуществляемая субъектами предпринимательства по извлечению благ.")
    y = definition(
        p, y, "Коммерческая деятельность", "(лат. commercium — торговля) — вид предпринимательской деятельности."
    )
    y = text(p, y + 10, "Основные нормативные акты", bold=True, center=True) + 4
    y = para(p, y, "Конституция РФ.")
    y = para(p, y, "Бюджетный кодекс РФ.")
    # 4. колонтитул «20  Раздел I. Общие положения» + § 1 (две строки) + сноска
    p = page("20                                   Раздел I. Общие положения")
    y = text(p, 60, "§ 1. Понятие предпринимательства", bold=True, center=True)
    y = text(p, y, "и предпринимательской деятельности", bold=True, center=True) + 8
    y = para(p, y, _BODY)
    y = para(p, y, "Глава 1 этого учебника посвящена основам, а предпринимательская деятельность в системе")
    text(p, H - 50, "¹ См.: Гинс Г. К. Предприниматель. М., 1992.", size=6.5)
    # 5. колонтитул «Глава 1. … право  21» + продолжение абзаца
    p = page("Глава 1. Предпринимательство и предпринимательское право                21")
    y = para(p, 60, "современного российского права анализируется и исследуется главным образом предпринимательским правом.")
    # 6. Глава 2 + § 1
    p = page("22                                   Раздел I. Общие положения")
    y = text(p, 90, "Глава 2. Источники", size=10.5, bold=True, center=True)
    y = text(p, y, "предпринимательского права", size=10.5, bold=True, center=True) + 12
    y = text(p, y, "§ 1. Понятие и виды источников", bold=True, center=True) + 8
    y = para(p, y, _BODY)
    doc.save(path)
    doc.close()


@pytest.fixture()
def textbook_pdf(tmp_path):
    p = tmp_path / "textbook.pdf"
    _make_textbook_pdf(str(p))
    return str(p)


def test_textbook_hierarchy(textbook_pdf):
    book, rep = extract_pdf(textbook_pdf)
    got = [(c.level, c.title) for c in book.chapters]
    assert got == [
        (1, "Раздел I. Общие положения"),
        (2, "Глава 1. Предпринимательство и предпринимательское право"),
        (3, "§ 1. Понятие предпринимательства и предпринимательской деятельности"),
        (2, "Глава 2. Источники предпринимательского права"),
        (3, "§ 1. Понятие и виды источников"),
    ], got
    assert rep.removed_pages == 2  # выходные данные + оглавление
    all_text = " ".join(p for c in book.chapters for p in c.paragraphs)
    assert "ISBN" not in all_text and "....." not in all_text
    assert "Раздел I. Общие положения" not in all_text  # колонтитулы
    assert "Гинс" not in all_text  # сноска
    # подзаголовок внутри главы и определения с жирным термином
    paras = book.chapters[1].paragraphs
    assert paras[0] == "#### Основные понятия"
    assert paras[1].startswith("**Бизнес** — деятельность")
    assert paras[2].startswith("**Коммерческая деятельность** (лат. commercium — торговля) — вид")
    assert "#### Основные нормативные акты" in paras
    # «Глава 1 этого учебника …» в тексте — не заголовок; абзац склеен через страницу
    assert any("в системе современного российского права" in p for p in book.chapters[2].paragraphs)


def test_textbook_text_roundtrip_levels(textbook_pdf):
    book, _ = extract_pdf(textbook_pdf)
    b2 = Book.from_text(book.to_text())
    assert [(c.level, c.title) for c in b2.chapters] == [(c.level, c.title) for c in book.chapters]
    assert b2.chapters[1].paragraphs[0] == "#### Основные понятия"


def test_file_groups_and_m4b_markers(tmp_path, textbook_pdf):
    from audiobook.pipeline import parse_into_project
    from audiobook.project import Project

    proj = Project.create(tmp_path / "projects", textbook_pdf, name="Учебник")
    parse_into_project(proj)
    by_chapter = [g[0] for g in proj.file_groups(2)]
    assert by_chapter == [
        "Глава 1. Предпринимательство и предпринимательское право",
        "Глава 2. Источники предпринимательского права",
    ]
    # раздел «приклеен» к первой главе
    assert proj.file_groups(2)[0][1] == [0, 1, 2]
    assert len(proj.file_groups(3)) == 3  # глава 1 (с глоссарием), § 1 главы 1, глава 2 + § 1
    assert len(proj.file_groups(1)) == 1
    # подзаголовок «Основные понятия» — отдельный сегмент-заголовок
    segs = proj.segments
    assert any(s.kind == "heading" and s.text == "Основные понятия." for s in segs)
    # определение: точка после термина, пауза после — «определительная»
    d = next(s for s in segs if s.text.startswith("Бизнес."))
    assert d.text.startswith("Бизнес. Деятельность, осуществляемая")
    assert d.pause_ms == 700
    c = next(s for s in segs if s.text.startswith("Коммерческая деятельность"))
    assert c.text.startswith("Коммерческая деятельность, от латинского commercium, торговля. Вид")
    # паузы по уровням: перед главой 2 — 2.5 с, после заголовка раздела — 1.5 с
    g2 = next(i for i, s in enumerate(segs) if s.text.startswith("Глава 2."))
    assert segs[g2 - 1].pause_ms == 2500
    assert segs[0].text.startswith("Раздел 1.") and segs[0].pause_ms == 1500
    p1 = next(i for i, s in enumerate(segs) if s.text.startswith("§ 1.") or s.text.startswith("Параграф 1."))
    assert segs[p1 - 1].pause_ms >= 1800


def test_export_mp3_folders(tmp_path, textbook_pdf):
    pytest.importorskip("imageio_ffmpeg")
    import soundfile as sf

    from audiobook.pipeline import export_book, parse_into_project
    from audiobook.project import Project

    proj = Project.create(tmp_path / "projects", textbook_pdf, name="Учебник")
    parse_into_project(proj)
    sr = 48000
    for s in proj.segments:  # «озвучиваем» тоном
        n = int(sr * 0.3)
        sf.write(str(proj.seg_path(s)), (0.1 * np.sin(np.arange(n) / sr * 2 * np.pi * 200)).astype(np.float32), sr)
    proj.refresh_status()
    proj.settings.export.formats = ("mp3",)
    outs = export_book(proj, log=lambda m: None)
    rel = [os.path.relpath(o, proj.output_dir / "mp3").replace("\\", "/") for o in outs]
    assert rel[0].startswith("01 Раздел I. Общие положения/01 Глава 1. Предпринимательство")
    assert rel[1].split("/")[-1].startswith("002 § 1. Понятие предпринимательства")
    assert rel[2].startswith("01 Раздел I. Общие положения/02 Глава 2.")


def test_voice_library(tmp_path):
    import soundfile as sf

    from audiobook.voices import check_reference, list_voices, load_voice, save_voice

    wav = tmp_path / "v.wav"
    sf.write(str(wav), (0.1 * np.sin(np.arange(16000 * 8) / 16000 * 2 * np.pi * 150)).astype(np.float32), 16000)
    v = save_voice("Диктор", str(wav), "Точный текст образца.", voices_dir=tmp_path / "voices")
    assert v.hifi and list_voices(tmp_path / "voices") == ["Диктор"]
    v2 = load_voice("Диктор", tmp_path / "voices")
    assert v2.transcript == "Точный текст образца." and os.path.exists(v2.audio)
    assert "8.0 с" in check_reference(v2.audio)


@pytest.mark.parametrize(
    "src,expected",
    [
        ("Как указано в ст. 34 Конституции РФ.", "Как указано в статье 34 Конституции РФ."),
        ("Согласно ч. 1 ст. 2 ГК РФ.", "Согласно части 1 статьи 2 ГК РФ."),
        ("См. п. 2 ст. 23 ГК РФ.", "Смотри пункт 2 статьи 23 ГК РФ."),
    ],
)
def test_legal_abbreviations(src, expected):
    assert normalize_text(src) == expected


def test_user_replacements():
    t = normalize_text("Нормы ГК РФ.", NormalizeOptions(replacements="ГК РФ=Гражданского кодекса"))
    assert t == "Нормы Гражданского кодекса."
