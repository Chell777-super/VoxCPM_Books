"""Очистка и нормализация текста перед синтезом.

Важно для VoxCPM2: круглые скобки в *начале* текста модель воспринимает как
инструкцию голоса («(Молодая женщина, мягкий голос)Текст…»), поэтому скобки
в тексте книги всегда заменяются на тире.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Сокращения (русский)
# --------------------------------------------------------------------------- #
_RU_ABBR = [
    (r"\bт\.\s?е\.", "то есть"),
    (r"\bт\.\s?к\.", "так как"),
    (r"\bи\s+т\.\s?д\.", "и так далее"),
    (r"\bи\s+т\.\s?п\.", "и тому подобное"),
    (r"\bт\.\s?д\.", "так далее"),
    (r"\bт\.\s?п\.", "тому подобное"),
    (r"\bи\s+пр\.", "и прочее"),
    (r"\bи\s+др\.", "и другие"),
    (r"\bв\s+т\.\s?ч\.", "в том числе"),
    (r"\bт\.\s?н\.", "так называемый"),
    (r"\bт\.\s?о\.", "таким образом"),
    (r"\bнапр\.", "например"),
    (r"\bсм\.", "смотри"),
    (r"\bср\.", "сравни"),
    (r"\bрис\.", "рисунок"),
    (r"\bтабл\.", "таблица"),
    (r"\bстр\.", "страница"),
    (r"\bгл\.", "глава"),
    (r"\bприм\.\s?(?:ред|пер|авт)\.", "примечание"),
    (r"\bприм\.", "примечание"),
    (r"\bпер\.\s+с\b", "перевод с"),
    (r"\bдо\s+н\.\s?э\.", "до нашей эры"),
    (r"\bн\.\s?э\.", "нашей эры"),
    (r"(?<=[IVXLC])\s?вв\.", " века"),
    (r"(?<=[IVXLC])\s?в\.", " век"),
    (r"\bпроф\.", "профессор"),
    (r"\bакад\.", "академик"),
    (r"\bул\.", "улица"),
    (r"\bд\.\s?(?=\d)", "дом "),
    (r"№\s?", "номер "),
    (r"§\s?", "параграф "),
]
_RU_ABBR = [(re.compile(p, re.IGNORECASE), r) for p, r in _RU_ABBR]

# Единицы после числа. Точку сохраняем, только если это похоже на конец предложения.
_RU_UNITS = {
    "гг": "годы",
    "г": "год",
    "вв": "века",
    "в": "век",
    "км": "километров",
    "кг": "килограммов",
    "см": "сантиметров",
    "мм": "миллиметров",
    "руб": "рублей",
    "коп": "копеек",
    "тыс": "тысяч",
    "млн": "миллионов",
    "млрд": "миллиардов",
}
_UNIT_RE = re.compile(
    r"(?P<prep>\b(?i:во|в|к|по|с|со|до|после|около|от|из|между|начала|конца|середины)\s+)?"
    r"(?P<num>\d+(?:\s?[-–]\s?\d+)?|[IVXLC]{1,6})\s?(?P<unit>гг|г|вв|в|км|кг|см|мм|руб|коп|тыс|млн|млрд)"
    r"(?P<dot>\.)?(?![а-яёa-z])"
)
_YEAR_CASE = {"в": "году", "во": "году", "к": "году", "по": "год", "с": "года", "со": "года", "до": "года",
              "после": "года", "около": "года", "от": "года", "из": "года", "начала": "года", "конца": "года",
              "середины": "года", "между": "годами"}


def _sentence_end_follows(m) -> bool:
    rest = m.string[m.end():]
    return not rest.strip() or bool(re.match(r"\s+[«\"—–-]?\s?[А-ЯЁA-Z]", rest))


# Сокращения, которыми может заканчиваться предложение (точка одновременно — конец фразы).
# После «ул.», «проф.», «см.» обычно идёт имя собственное, там точку не сохраняем.
_CAN_END_SENTENCE = {
    "и так далее", "и тому подобное", "и прочее", "и другие", "так далее", "тому подобное",
    "нашей эры", "до нашей эры", "et cetera",
}


def _abbr_repl(m, rep: str) -> str:
    """Замена сокращения с сохранением заглавной буквы и точки конца предложения."""
    src = m.group(0)
    out = rep
    if src[:1].isupper() and out[:1].islower():
        out = out[:1].upper() + out[1:]
    if src.rstrip().endswith(".") and rep in _CAN_END_SENTENCE and _sentence_end_follows(m):
        out += "."
    if src.endswith(" ") and not out.endswith(" "):
        out += " "
    return out


def _expand_units(text: str) -> str:
    def repl(m):
        unit = m.group("unit")
        if m.group("dot") is None and unit in ("г", "в", "гг", "вв", "руб", "коп", "тыс"):
            return m.group(0)  # без точки это, скорее всего, не сокращение
        word = _RU_UNITS[unit]
        prep = (m.group("prep") or "")
        p = prep.strip().lower()
        if unit == "г" and p in _YEAR_CASE:
            word = _YEAR_CASE[p]
        elif unit == "в" and p in ("в", "во"):
            word = "веке"
        end_dot = "." if m.group("dot") and _sentence_end_follows(m) else ""
        return f"{prep}{m.group('num')} {word}{end_dot}"

    return _UNIT_RE.sub(repl, text)


_EN_ABBR = [
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\betc\.", "et cetera"),
    (r"\bvs\.", "versus"),
    (r"\bMr\.", "Mister"),
    (r"\bMrs\.", "Missis"),
    (r"\bDr\.", "Doctor"),
    (r"\bSt\.", "Saint"),
    (r"\bNo\.\s?(?=\d)", "number "),
]
# Единицы после числа. Точку сохраняем, только если это похоже на конец предложения.
_RU_UNITS = {
    "гг": "годы",
    "г": "год",
    "вв": "века",
    "в": "век",
    "км": "километров",
    "кг": "килограммов",
    "см": "сантиметров",
    "мм": "миллиметров",
    "руб": "рублей",
    "коп": "копеек",
    "тыс": "тысяч",
    "млн": "миллионов",
    "млрд": "миллиардов",
}
_UNIT_RE = re.compile(
    r"(?P<prep>\b(?i:во|в|к|по|с|со|до|после|около|от|из|между|начала|конца|середины)\s+)?"
    r"(?P<num>\d+(?:\s?[-–]\s?\d+)?|[IVXLC]{1,6})\s?(?P<unit>гг|г|вв|в|км|кг|см|мм|руб|коп|тыс|млн|млрд)"
    r"(?P<dot>\.)?(?![а-яёa-z])"
)
_YEAR_CASE = {"в": "году", "во": "году", "к": "году", "по": "год", "с": "года", "со": "года", "до": "года",
              "после": "года", "около": "года", "от": "года", "из": "года", "начала": "года", "конца": "года",
              "середины": "года", "между": "годами"}


def _sentence_end_follows(m) -> bool:
    rest = m.string[m.end():]
    return not rest.strip() or bool(re.match(r"\s+[«\"—–-]?\s?[А-ЯЁA-Z]", rest))


# Сокращения, которыми может заканчиваться предложение (точка одновременно — конец фразы).
# После «ул.», «проф.», «см.» обычно идёт имя собственное, там точку не сохраняем.
_CAN_END_SENTENCE = {
    "и так далее", "и тому подобное", "и прочее", "и другие", "так далее", "тому подобное",
    "нашей эры", "до нашей эры", "et cetera",
}


def _abbr_repl(m, rep: str) -> str:
    """Замена сокращения с сохранением заглавной буквы и точки конца предложения."""
    src = m.group(0)
    out = rep
    if src[:1].isupper() and out[:1].islower():
        out = out[:1].upper() + out[1:]
    if src.rstrip().endswith(".") and rep in _CAN_END_SENTENCE and _sentence_end_follows(m):
        out += "."
    if src.endswith(" ") and not out.endswith(" "):
        out += " "
    return out


def _expand_units(text: str) -> str:
    def repl(m):
        unit = m.group("unit")
        if m.group("dot") is None and unit in ("г", "в", "гг", "вв", "руб", "коп", "тыс"):
            return m.group(0)  # без точки это, скорее всего, не сокращение
        word = _RU_UNITS[unit]
        prep = (m.group("prep") or "")
        p = prep.strip().lower()
        if unit == "г" and p in _YEAR_CASE:
            word = _YEAR_CASE[p]
        elif unit == "в" and p in ("в", "во"):
            word = "веке"
        end_dot = "." if m.group("dot") and _sentence_end_follows(m) else ""
        return f"{prep}{m.group('num')} {word}{end_dot}"

    return _UNIT_RE.sub(repl, text)


_EN_ABBR = [(re.compile(p), r) for p, r in _EN_ABBR]

_ROMAN_KEYWORDS = r"(?:глава|главы|главе|часть|части|том|тома|книга|книги|раздел|chapter|part|book|volume)"
_ROMAN_RE = re.compile(rf"\b({_ROMAN_KEYWORDS})\s+([IVXLCDM]{{1,8}})\b", re.IGNORECASE)
_ROMAN_CENTURY_RE = re.compile(r"\b([IVXLC]{1,6})(?=\s*(?:век|века|веке|веков|столети))")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int | None:
    s = s.upper()
    if not s or any(c not in _ROMAN_VALUES for c in s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        v = _ROMAN_VALUES[c]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total if 0 < total < 4000 else None


@dataclass
class NormalizeOptions:
    lang: str = "ru"
    expand_abbreviations: bool = True
    numbers_to_words: bool = False  # VoxCPM2 обычно сам хорошо читает числа
    remove_footnote_refs: bool = True
    fix_caps: bool = True


_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b", re.I)
_FOOTREF_RE = re.compile(r"\[\s*\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\s*\]|[¹²³⁴⁵⁶⁷⁸⁹⁰]+|(?<=[\w.,;:!?»])\*{1,3}(?=[\s.,;:!?»]|$)")
_CAPS_RUN_RE = re.compile(r"\b(?:[A-ZА-ЯЁ]{2,}[\s,.:;!?—–-]+){1,}[A-ZА-ЯЁ]{2,}\b")
_CAPS_WORD_RE = re.compile(r"\b[A-ZА-ЯЁ]{7,}\b")


def _fix_caps(text: str) -> str:
    """«ГЛАВА ПЕРВАЯ. ВСТРЕЧА» → «Глава первая. Встреча». Аббревиатуры (СССР, НАТО) не трогаем."""

    def repl(m):
        s = m.group(0).lower()
        before = text[: m.start()].rstrip(" «\"—–-")
        if not before or before[-1] in ".!?…:":
            s = s[:1].upper() + s[1:]
        # первая буква после точки внутри фрагмента
        return re.sub(r"([.!?…]\s+)([a-zа-яё])", lambda x: x.group(1) + x.group(2).upper(), s)

    text = _CAPS_RUN_RE.sub(repl, text)
    return _CAPS_WORD_RE.sub(repl, text)


def _brackets_to_dashes(text: str) -> str:
    starts_with_bracket = text.lstrip()[:1] in "(["
    text = re.sub(r"\[([^\]]*)\]", r"(\1)", text)
    text = re.sub(r"\(\s*\)", " ", text)
    text = re.sub(r"\s*\(\s*", " — ", text)
    text = re.sub(r"\s*\)\s*", " — ", text)
    # чистка «— .», «— ,», двойных тире
    text = re.sub(r"\s*—\s*([.,;:!?…])", r"\1", text)
    text = re.sub(r"(?:\s*—\s*){2,}", " — ", text)
    text = text.strip()
    if starts_with_bracket:
        text = re.sub(r"^—\s*", "", text)
    return re.sub(r"\s*—$", "", text)


def _numbers_to_words(text: str, lang: str) -> str:
    try:
        from num2words import num2words
    except ImportError:
        return text

    def repl(m):
        s = m.group(0)
        try:
            if "," in s or "." in s:
                return num2words(float(s.replace(",", ".")), lang=lang)
            return num2words(int(s), lang=lang)
        except Exception:
            return s

    # не трогаем «1990-х», «5-й» — суффиксы после дефиса (модель читает их лучше в цифрах)
    return re.sub(r"(?<![\w-])\d+(?:[.,]\d+)?(?![\w-])", repl, text)


def normalize_text(text: str, opts: NormalizeOptions | None = None) -> str:
    opts = opts or NormalizeOptions()
    t = text

    t = _URL_RE.sub(" ", t)
    if opts.remove_footnote_refs:
        t = _FOOTREF_RE.sub("", t)

    # Кавычки, тире, многоточия
    t = t.replace("„", "«").replace("“", "»" if opts.lang == "ru" else "“")
    t = re.sub(r"\.\s?\.\s?\.", "…", t)
    t = re.sub(r"(?<=\s)--?(?=\s)", "—", t)
    t = re.sub(r"(?<=\s)–(?=\s)", "—", t)
    t = re.sub(r"^\s*[-–]\s+", "— ", t)

    # Римские цифры после «Глава/Часть/Том» и перед «век»
    t = _ROMAN_RE.sub(lambda m: f"{m.group(1)} {roman_to_int(m.group(2)) or m.group(2)}", t)
    t = _ROMAN_CENTURY_RE.sub(lambda m: str(roman_to_int(m.group(1)) or m.group(1)), t)

    if opts.expand_abbreviations:
        if opts.lang == "ru":
            t = _expand_units(t)
        table = _RU_ABBR if opts.lang == "ru" else _EN_ABBR
        for rx, rep in table:
            t = rx.sub(lambda m, rep=rep: _abbr_repl(m, rep), t)
        t = _ROMAN_CENTURY_RE.sub(lambda m: str(roman_to_int(m.group(1)) or m.group(1)), t)

    t = _brackets_to_dashes(t)

    # Символы, которые TTS не произносит или произносит плохо
    if opts.lang == "ru":
        t = re.sub(r"(?<=\d)\s?%", " процентов", t)
        t = t.replace("&", " и ")
    t = re.sub(r"[|_~^<>{}=#\\/]+", " ", t)
    t = re.sub(r"\*+", " ", t)

    if opts.fix_caps:
        t = _fix_caps(t)
    if opts.numbers_to_words:
        t = _numbers_to_words(t, "ru" if opts.lang == "ru" else "en")

    # Пробелы перед знаками препинания, повторы
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([.,;:!?…»)])", r"\1", t)
    t = re.sub(r"([«(])\s+", r"\1", t)
    t = re.sub(r"([,;:])(?=[^\s\d»])", r"\1 ", t)
    t = re.sub(r",{2,}", ",", t)
    return t.strip()


def speakable(text: str) -> bool:
    """Есть ли в тексте что произносить (буквы или цифры)."""
    return bool(re.search(r"[\w]", text, re.UNICODE)) and bool(re.search(r"[^\W_]", text))
