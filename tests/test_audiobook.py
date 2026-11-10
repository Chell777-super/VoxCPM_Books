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
    dur = assemble_chapter([(str(p), 500), (str(p), 0)], tmp_path / "ch.wav", sr, lead_in_ms=0)
    assert abs(dur - (2 * len(out) / sr + 0.5)) < 0.01
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
