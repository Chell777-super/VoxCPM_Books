"""Веб-интерфейс VoxCPM Books (Gradio)."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf

from .audio import ExportOptions, fmt_duration, iphone_preset, postprocess
from .book import LEVEL_NAMES, Book, safe_filename
from .engine import EngineConfig, SynthConfig, TTSEngine, VoiceConfig
from .normalize import NormalizeOptions, normalize_text
from .pdf_extract import ExtractOptions, render_cover
from .pipeline import SynthesisJob, export_book, parse_into_project
from .project import Project, list_projects
from .segmenter import SegmentOptions, estimate_duration_sec
from .voices import check_reference, list_voices, load_voice, save_voice
from .textproc import process_file, process_markup, write_outputs

PRESETS = {"Быстро (черновик)": 6, "Баланс (рекомендуется)": 10, "Качество": 16}
CHAPTER_MODES = {
    "Авто (оглавление → шрифт → шаблоны)": "auto",
    "Оглавление PDF": "toc",
    "Крупный шрифт": "font",
    "Шаблоны «Глава N»": "regex",
    "Не делить на главы": "none",
}
VOICE_MODES = ["Клонирование по образцу (Hi-Fi с текстом образца)", "Создать по описанию", "Голос модели по умолчанию"]
DESIGN_SAMPLE = (
    "Добрый вечер. Сегодня я расскажу вам историю, которая началась холодным осенним утром, "
    "когда над городом ещё висел густой туман."
)
DESIGN_DEFAULT = "Мужчина средних лет, тёплый бархатный низкий голос, спокойная размеренная манера чтения аудиокниги"


class AppState:
    def __init__(self, projects_dir: str, model: str = "models/VoxCPM2", compile_model: bool = True, engine_cfg=None):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.messages: list[str] = []
        self.job = None
        from .engine_nano import make_engine

        cfg = engine_cfg or EngineConfig(model_path=model, compile=compile_model, backend="torch")
        self.engine = make_engine(cfg, log=self._log)
        self.engine_lock = threading.Lock()
        self.project: Optional[Project] = None
        self.voices_dir = Path(os.environ.get("VOXBOOKS_VOICES", "voices"))
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, m: str):
        print(m, flush=True)
        line = time.strftime("%H:%M:%S ") + m
        self.messages.append(line)
        del self.messages[:-300]
        if self.job is not None and self.job.state.running:
            self.job.state.log.append(line)  # сообщения движка (повторы и т.п.) — в журнал задачи

    @property
    def busy(self) -> bool:
        return self.job is not None and self.job.state.running


S: "AppState" = None  # type: ignore  # инициализируется в launch()


# --------------------------------------------------------------------------- #
# Помощники
# --------------------------------------------------------------------------- #


def _need_project() -> Project:
    if S.project is None:
        raise gr.Error("Сначала создайте или откройте проект (вкладка «Книга»).")
    return S.project


def _project_status_md() -> str:
    p = S.project
    gpu = S.engine.vram_info() if S.engine.model is not None else "модель ещё не загружена"
    if p is None:
        return f"**Проект не открыт.** · GPU: {gpu}"
    s = p.stats()
    pct = 100 * s["done_chars"] / max(1, s["chars"])
    est = estimate_duration_sec(s["chars"])
    return (
        f"**{p.book.title if p.book and p.book.title else p.name}** · глав {s['chapters']} · сегментов {s['total']} · "
        f"{s['chars']:,} симв. (≈ {fmt_duration(est)} аудио) · озвучено **{pct:.1f}%** "
        f"({fmt_duration(s['audio_sec'])}) · на проверку {s['flagged']} · ошибок {s['errors']}  \nGPU: {gpu}"
    ).replace(",", " ")


def _chapters_table():
    p = S.project
    if p is None:
        return []
    rows = []
    levels = p.chapter_levels or [2] * len(p.chapter_titles)
    indent = {1: "", 2: "   ", 3: "      "}
    for ci, title in enumerate(p.chapter_titles):
        segs = p.chapter_segments(ci)
        chars = sum(len(s.text) for s in segs)
        done = sum(1 for s in segs if s.status in ("done", "flagged"))
        lvl = levels[ci] if ci < len(levels) else 2
        rows.append(
            [ci + 1, LEVEL_NAMES.get(lvl, ""), indent.get(lvl, "") + title[:90], len(segs), chars,
             fmt_duration(estimate_duration_sec(chars)), f"{done}/{len(segs)}"]
        )
    return rows


def _segments_table(limit: int = 400, only_problems: bool = False):
    p = S.project
    if p is None:
        return []
    icon = {"done": "✅", "flagged": "⚑", "error": "✖", "pending": "·"}
    rows = []
    for s in p.segments:
        if only_problems and s.status not in ("flagged", "error"):
            continue
        rows.append([s.idx, s.chapter + 1, icon.get(s.status, s.status), len(s.text), s.text[:200], s.note])
        if len(rows) >= limit:
            break
    return rows


def _opts_from_values(
    first_page, last_page, chapter_mode, rm_headers, rm_foot, skip_toc, skip_service, lang, numbers, abbr, target, maxc, minc,
    titles, merge, mark_defs=True, replacements="", skip_front=True,
):
    extract = ExtractOptions(
        first_page=int(first_page or 1),
        last_page=int(last_page) if last_page else None,
        remove_headers=bool(rm_headers),
        remove_footnotes=bool(rm_foot),
        chapter_mode=CHAPTER_MODES.get(chapter_mode, "auto"),
        skip_toc_pages=bool(skip_toc),
        skip_service_pages=bool(skip_service),
        skip_front_matter=bool(skip_front),
    )
    normalize = NormalizeOptions(
        lang=lang, numbers_to_words=bool(numbers), expand_abbreviations=bool(abbr), replacements=replacements or ""
    )
    segment = SegmentOptions(
        target_chars=int(target),
        max_chars=max(int(maxc), int(target) + 20),
        min_chars=int(minc),
        read_chapter_titles=bool(titles),
        merge_short_paragraphs=bool(merge),
        mark_definitions=bool(mark_defs),
    )
    return extract, normalize, segment


def _settings_from_ui(*values):
    p = _need_project()
    p.settings.extract, p.settings.normalize, p.settings.segment = _opts_from_values(*values)


def _tp_outputs(res, stem: str):
    out_dir = Path(tempfile.mkdtemp(prefix="voxbooks_text_"))
    files = write_outputs(res, out_dir / f"{stem} — разметка.txt", out_dir / f"{stem} — озвучка.txt")
    summary = "\n".join(f"- {line}" for line in res.summary.splitlines())
    for w in res.warnings:
        summary += f"\n- ⚠ {w}"
    return summary, res.markup, res.speech, files


def on_text_process(file, *settings):
    if not file:
        raise gr.Error("Загрузите PDF или TXT.")
    extract, normalize, segment = _opts_from_values(*settings)
    t0 = time.time()
    res = process_file(file, extract, segment, normalize)
    summary, markup, speech, files = _tp_outputs(res, safe_filename(Path(file).stem, 60))
    return f"**Готово за {time.time() - t0:.1f} с**\n\n" + summary, markup, speech, files


def on_text_refresh(markup, file, *settings):
    if not (markup or "").strip():
        raise gr.Error("Нет текста — сначала обработайте файл.")
    _, normalize, segment = _opts_from_values(*settings)
    stem = safe_filename(Path(file).stem, 60) if file else "текст"
    res = process_markup(markup, segment, normalize, default_title=stem)
    summary, markup, speech, files = _tp_outputs(res, stem)
    return "**Предпросмотр обновлён по правленому тексту**\n\n" + summary, speech, files


def on_text_create_project(markup, file, name, *settings):
    if S.busy:
        raise gr.Error("Идёт озвучка — сначала остановите её.")
    if not (markup or "").strip():
        raise gr.Error("Нет текста — сначала обработайте файл.")
    stem = (name or "").strip() or (Path(file).stem if file else "Книга")
    tmp = Path(tempfile.mkdtemp(prefix="voxbooks_text_")) / f"{safe_filename(stem, 60)}.txt"
    tmp.write_text(markup, encoding="utf-8")
    S.project = Project.create(S.projects_dir, str(tmp), name=stem)
    _settings_from_ui(*settings)
    if file and str(file).lower().endswith(".pdf"):
        render_cover(str(file), str(S.project.cover_path))
    book = Book.from_text(markup, default_title=stem)
    info = parse_into_project(S.project, book=book, log=S._log)
    return _parse_outputs(
        f"✅ Проект «{S.project.name}» создан из размеченного текста: глав {info['chapters']}, сегментов {info['total']}. "
        "Дальше — вкладка «Голос»."
    )


def _parse_outputs(msg: str):
    p = S.project
    text = p.book.to_text() if p and p.book else ""
    return (
        msg,
        _chapters_table(),
        _segments_table(),
        text,
        _project_status_md(),
        gr.update(choices=list_projects(S.projects_dir), value=p.name if p else None),
    )


# --------------------------------------------------------------------------- #
# Обработчики: Книга
# --------------------------------------------------------------------------- #


def on_create(file, name, *settings):
    if S.busy:
        raise gr.Error("Идёт озвучка — сначала остановите её.")
    if not file:
        raise gr.Error("Загрузите PDF или TXT.")
    path = file if isinstance(file, str) else file.name
    S.project = Project.create(S.projects_dir, path, name=name or Path(path).stem)
    _settings_from_ui(*settings)
    t0 = time.time()
    info = parse_into_project(S.project, log=S._log)
    rep = info.get("report")
    msg = f"✅ Разобрано за {time.time() - t0:.1f} с: глав {info['chapters']}, сегментов {info['total']}."
    if rep:
        msg += (
            f"  \nСтраниц: {rep.pages}, удалено колонтитулов: {rep.removed_headers}, сносок: {rep.removed_footnotes}. "
            f"Главы определены: {rep.chapter_source}."
        )
        for w in rep.warnings:
            msg += f"  \n⚠ {w}"
    msg += "  \nПроверьте главы и текст ниже. Правки можно внести в блоке «Редактировать текст»."
    return _parse_outputs(msg)


def on_resegment(*settings):
    """Заново разбить на сегменты текущий текст (с правками) — например, после смены длины сегмента."""
    if S.busy:
        raise gr.Error("Идёт озвучка — сначала остановите её.")
    _settings_from_ui(*settings)
    p = S.project
    info = parse_into_project(p, book=p.book, log=S._log)
    return _parse_outputs(f"✅ Перераспределено: сегментов {info['total']} (озвученные неизменённые сегменты сохранены).")


def on_reextract(*settings):
    """Извлечь текст из исходного файла заново (ручные правки текста будут потеряны)."""
    if S.busy:
        raise gr.Error("Идёт озвучка — сначала остановите её.")
    _settings_from_ui(*settings)
    info = parse_into_project(S.project, log=S._log)
    rep = info.get("report")
    msg = f"✅ Текст извлечён заново: глав {info['chapters']}, сегментов {info['total']}."
    if rep:
        msg += f"  \nГлавы определены: {rep.chapter_source}."
        for w in rep.warnings:
            msg += f"  \n⚠ {w}"
    return _parse_outputs(msg)


def on_apply_text(text, *settings):
    if S.busy:
        raise gr.Error("Идёт озвучка — сначала остановите её.")
    p = _need_project()
    _settings_from_ui(*settings)
    book = Book.from_text(text, default_title=p.book.title if p.book else "")
    if p.book and not book.author:
        book.author = p.book.author
    info = parse_into_project(p, book=book, log=S._log)
    return _parse_outputs(f"✅ Правки применены: глав {info['chapters']}, сегментов {info['total']}.")


def on_open(name):
    if S.busy:
        raise gr.Error("Идёт озвучка — сначала остановите её.")
    if not name:
        raise gr.Error("Выберите проект")
    S.project = Project(S.projects_dir / name)
    out = list(_parse_outputs(f"Открыт проект «{name}»."))
    return out + list(_voice_ui_values()) + _parse_ui_values()


def _parse_ui_values():
    st = S.project.settings
    e, n, g = st.extract, st.normalize, st.segment
    mode = next((k for k, v in CHAPTER_MODES.items() if v == e.chapter_mode), list(CHAPTER_MODES)[0])
    return [
        e.first_page, e.last_page or 0, mode, e.remove_headers, e.remove_footnotes, e.skip_toc_pages,
        e.skip_service_pages, n.lang, n.numbers_to_words, n.expand_abbreviations, g.target_chars, g.max_chars,
        g.min_chars, g.read_chapter_titles, g.merge_short_paragraphs, g.mark_definitions, n.replacements,
        e.skip_front_matter,
    ]


def _voice_ui_values():
    p = S.project
    v = p.settings.voice if p else VoiceConfig()
    sc = p.settings.synth if p else SynthConfig()
    mode = {"clone": VOICE_MODES[0], "design": VOICE_MODES[1]}.get(v.mode, VOICE_MODES[2])
    preset = next((k for k, t in PRESETS.items() if t == sc.inference_timesteps), "Баланс (рекомендуется)")
    ref = v.reference_wav if v.reference_wav and os.path.exists(v.reference_wav) else None
    return mode, ref, v.prompt_text or "", v.style or "", v.denoise, preset, sc.cfg_value, sc.seed


# --------------------------------------------------------------------------- #
# Обработчики: Голос
# --------------------------------------------------------------------------- #


def _apply_voice(mode, ref_audio, ref_text, style, denoise, preset, cfg, seed) -> bool:
    p = _need_project()
    v = p.settings.voice
    before = (v.mode, v.reference_wav, v.prompt_text, v.style)
    if mode == VOICE_MODES[0]:
        if not ref_audio:
            raise gr.Error("Загрузите образец голоса (10–20 секунд чистой речи).")
        p.voice_dir.mkdir(parents=True, exist_ok=True)
        src = Path(ref_audio)
        dst = p.voice_dir / ("reference" + (src.suffix.lower() or ".wav"))
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        v.mode, v.reference_wav = "clone", str(dst)
        v.prompt_text = (ref_text or "").strip() or None
    elif mode == VOICE_MODES[1]:
        v.mode, v.reference_wav, v.prompt_text = "design", None, None
    else:
        v.mode, v.reference_wav, v.prompt_text = "default", None, None
    v.style = (style or "").strip()
    v.denoise = bool(denoise)
    p.settings.synth.inference_timesteps = PRESETS.get(preset, 10)
    p.settings.synth.cfg_value = float(cfg)
    p.settings.synth.seed = int(seed)
    changed = before != (v.mode, v.reference_wav, v.prompt_text, v.style)
    if changed:
        p.rekey_for_voice()
    p.save()
    return changed


def on_save_voice(*args):
    if S.busy:
        raise gr.Error("Идёт озвучка — сначала остановите её.")
    changed = _apply_voice(*args)
    msg = "✅ Голос сохранён."
    if changed:
        msg += " Голос изменился — ранее озвученные сегменты будут переозвучены этим голосом."
    if S.project.settings.voice.mode != "clone":
        msg += (
            "  \n⚠ Без образца тембр может немного меняться от сегмента к сегменту. "
            "Для книги лучше: «Создать по описанию» → «Сгенерировать образец» → «Использовать этот голос»."
        )
    return msg, _project_status_md()


def _synth_preview(text: str, sc: SynthConfig):
    with S.engine_lock:
        res = S.engine.synthesize(text, sc)
    wav = postprocess(res.wav, res.sr)
    out = Path(tempfile.gettempdir()) / f"voxbooks_preview_{int(time.time() * 1000)}.wav"
    sf.write(out, wav, res.sr)
    return str(out), res


def on_test_voice(test_text, *voice_args):
    if S.busy:
        raise gr.Error("Идёт озвучка — прослушивание недоступно, пока модель занята.")
    _apply_voice(*voice_args)
    p = S.project
    text = (test_text or "").strip() or (p.segments[1].text if len(p.segments) > 1 else DESIGN_SAMPLE)
    text = normalize_text(text, p.settings.normalize)
    S.engine.load()
    if S.project.settings.voice.mode == "design" and not S.project.settings.voice.style:
        raise gr.Error("Введите описание голоса.")
    S.engine.set_voice(p.settings.voice)
    path, res = _synth_preview(text, p.settings.synth)
    return path, f"Сгенерировано {res.seconds:.1f} с аудио за {res.gen_time:.1f} с (RTF {res.gen_time / max(res.seconds, 0.01):.2f}), темп {res.cps:.1f} симв./с"


def on_design_sample(description, sample_text, cfg, preset, seed):
    if S.busy:
        raise gr.Error("Идёт озвучка — модель занята.")
    if not (description or "").strip():
        raise gr.Error("Опишите голос: пол, возраст, тембр, манеру.")
    S.engine.load()
    S.engine.set_voice(VoiceConfig(mode="design", style=description.strip()))
    sc = SynthConfig(cfg_value=float(cfg), inference_timesteps=PRESETS.get(preset, 10), seed=int(seed) + int(time.time()) % 1000)
    path, res = _synth_preview((sample_text or DESIGN_SAMPLE).strip(), sc)
    return path, f"Образец: {res.seconds:.1f} с. Не понравилось — нажмите ещё раз (каждый раз новый вариант)."


def on_ref_change(path):
    return check_reference(path) if path else ""


def on_lib_save(name, audio, transcript, style):
    if not audio:
        raise gr.Error("Сначала загрузите образец голоса.")
    v = save_voice(name, audio, transcript or "", style or "", S.voices_dir)
    kind = "Hi-Fi (с транскрипцией)" if v.hifi else "без транскрипции — будет обычное клонирование"
    return f"✅ Голос «{v.name}» сохранён в библиотеку: {kind}.", gr.update(choices=list_voices(S.voices_dir), value=v.name)


def on_lib_load(name):
    if not name:
        raise gr.Error("Выберите голос из библиотеки.")
    v = load_voice(name, S.voices_dir)
    return VOICE_MODES[0], v.audio, v.transcript, v.style, name, f"Голос «{name}» загружен. Нажмите «Сохранить голос в проект»."


def on_use_design(sample_path, sample_text):
    p = _need_project()
    if not sample_path:
        raise gr.Error("Сначала сгенерируйте образец.")
    p.voice_dir.mkdir(parents=True, exist_ok=True)
    dst = p.voice_dir / f"designed_{int(time.time())}.wav"
    shutil.copyfile(sample_path, dst)
    return VOICE_MODES[0], str(dst), (sample_text or DESIGN_SAMPLE).strip(), "", (
        "✅ Образец выбран как голос книги (режим клонирования + транскрипт). Нажмите «Сохранить голос»."
    )


# --------------------------------------------------------------------------- #
# Обработчики: Озвучка
# --------------------------------------------------------------------------- #


def _start_job(indices=None, redo=False, reseed=False):
    p = _need_project()
    if S.busy:
        raise gr.Error("Озвучка уже идёт.")
    if not p.segments:
        raise gr.Error("В проекте нет сегментов — сначала разберите книгу.")
    S.job = SynthesisJob(p, S.engine, indices=indices, redo=redo, reseed=reseed, log=None).start()
    return "▶ Запущено…"


def on_start():
    return _start_job()


def on_redo_flagged():
    p = _need_project()
    idx = [s.idx for s in p.segments if s.status in ("flagged", "error")]
    if not idx:
        raise gr.Error("Нет помеченных сегментов.")
    return _start_job(indices=idx, redo=True, reseed=True)


def on_stop():
    if S.job and S.job.state.running:
        S.job.stop()
        return "⏹ Остановка после текущего сегмента…"
    return "Ничего не запущено."


def _progress_html() -> str:
    p = S.project
    if p is None:
        return ""
    s = p.stats()
    pct = 100 * s["done_chars"] / max(1, s["chars"])
    extra = ""
    j = S.job
    if j is not None:
        st = j.state
        remaining = s["chars"] - s["done_chars"]
        cps_vals = [x.cps for x in p.segments if x.status == "done" and x.cps > 0]
        cps = float(np.median(cps_vals)) if cps_vals else 14.0
        eta = st.eta(remaining, cps) if st.running else 0
        state = "идёт озвучка" if st.running else (st.finished_msg or "остановлено")
        extra = (
            f"<div>Статус: <b>{state}</b> · за сессию {st.done}/{st.total} · RTF <b>{st.rtf:.2f}</b>"
            + (f" · осталось ≈ <b>{fmt_duration(eta)}</b>" if eta else "")
            + "</div>"
        )
        if st.current:
            extra += f"<div style='opacity:.75;margin-top:4px'>{st.current}</div>"
    return (
        f"<div style='font-size:15px'><div style='background:#8884;border-radius:6px;height:18px;overflow:hidden'>"
        f"<div style='background:#4c8bf5;height:18px;width:{pct:.2f}%'></div></div>"
        f"<div style='margin-top:6px'>Озвучено <b>{pct:.1f}%</b> · сегментов {s['done']}/{s['total']} · "
        f"аудио {fmt_duration(s['audio_sec'])} · на проверку {s['flagged']} · ошибок {s['errors']}</div>{extra}</div>"
    )


def on_tick():
    logs = "\n".join((S.job.state.log if S.job else [])[-60:] or S.messages[-60:])
    return _progress_html(), logs, _project_status_md()


def on_show_segment(idx):
    p = _need_project()
    idx = int(idx or 0)
    if not 0 <= idx < len(p.segments):
        raise gr.Error(f"Номер сегмента от 0 до {len(p.segments) - 1}")
    s = p.segments[idx]
    path = p.seg_path(s)
    info = (
        f"Глава {s.chapter + 1} · {s.kind} · статус **{s.status}** · {len(s.text)} симв. · "
        f"{s.seconds:.1f} с · {s.cps:.1f} симв./с · попыток {s.attempts} · пауза после {s.pause_ms} мс"
        + (f"  \nЗаметка: {s.note}" if s.note else "")
    )
    return s.text, (str(path) if path.exists() else None), info


def on_save_segment_text(idx, text):
    p = _need_project()
    if S.busy:
        raise gr.Error("Идёт озвучка — сначала остановите её.")
    p.update_segment_text(int(idx), text)
    return "✅ Текст сегмента сохранён — он будет озвучен заново.", _segments_table(only_problems=False)


def on_redo_segment(idx):
    p = _need_project()
    i = int(idx)
    if not 0 <= i < len(p.segments):
        raise gr.Error("Неверный номер сегмента")
    return _start_job(indices=[i], redo=True, reseed=True)


def on_mark_ok(idx):
    p = _need_project()
    s = p.segments[int(idx)]
    if s.status == "flagged":
        s.status, s.note = "done", "проверено вручную"
        p.save_segments()
    return "✅ Отмечено как проверенное.", _segments_table(only_problems=True)


# --------------------------------------------------------------------------- #
# Обработчики: Сборка
# --------------------------------------------------------------------------- #


SPLIT_LEVELS = {"Разделам": 1, "Главам": 2, "Параграфам (§)": 3}


FMT_M4B = "M4B (одним файлом, с главами)"
IPHONE_HINT = (
    "📱 **Как закинуть на iPhone.** Самое простое — **BookPlayer** (бесплатный, App Store):\n"
    "1. Отправьте `.m4b` себе в Telegram → «Избранное» (или положите в iCloud Drive).\n"
    "2. На iPhone откройте файл → «Поделиться» → **BookPlayer**.\n\n"
    "Главы (§) и обложка видны в плеере; позиция запоминается. Если частей несколько — "
    "закиньте все, BookPlayer покажет их по порядку."
)


def on_export_iphone(allow_missing, progress=gr.Progress()):
    """Пресет «Для iPhone»: M4B AAC 96k моно 44.1 кГц, метки по §, части до 1.9 ГБ (лимит Telegram)."""
    opts = iphone_preset(S.project.settings.export if S.project else None)
    outs, log, msg = on_export(
        [FMT_M4B], "Параграфам (§)", "Параграфам (§)", "320k", opts.aac_bitrate, str(opts.sample_rate),
        True, allow_missing, opts.max_part_mb, progress=progress,
    )
    ui = (
        gr.update(value=[FMT_M4B]),
        gr.update(value="Параграфам (§)"),
        gr.update(value=opts.aac_bitrate),
        gr.update(value=str(opts.sample_rate)),
        gr.update(value=opts.max_part_mb),
    )
    return (outs, log, msg + "\n\n" + IPHONE_HINT) + ui


def on_export(formats, split_by, marks_by, mp3_br, aac_br, sr, folders, allow_missing, max_part_mb=0, progress=gr.Progress()):
    p = _need_project()
    if S.busy:
        raise gr.Error("Дождитесь окончания озвучки или остановите её.")
    fm = []
    if "MP3 (отдельные файлы)" in formats:
        fm.append("mp3")
    if FMT_M4B in formats:
        fm.append("m4b")
    if "WAV (отдельные файлы)" in formats:
        fm.append("wav")
    if not fm:
        raise gr.Error("Выберите хотя бы один формат.")
    p.settings.export = ExportOptions(
        formats=tuple(fm),
        mp3_bitrate=mp3_br,
        aac_bitrate=aac_br,
        sample_rate=int(sr),
        split_level=SPLIT_LEVELS.get(split_by, 2),
        marker_level=SPLIT_LEVELS.get(marks_by, 3),
        folders=bool(folders),
        max_part_mb=int(max_part_mb or 0),
    )
    p.save()
    logs = []
    outs = export_book(p, allow_missing=bool(allow_missing), log=logs.append, progress=lambda f, m: progress(f, desc=m))
    return outs, "\n".join(logs), f"Папка результата: `{p.output_dir.resolve()}`"


# --------------------------------------------------------------------------- #
# Интерфейс
# --------------------------------------------------------------------------- #


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VoxCPM Books") as demo:
        gr.Markdown("# 📚 VoxCPM Books — аудиокниги из PDF\nVoxCPM2 · 48 кГц · клонирование и дизайн голоса")
        with gr.Row():
            _projects = list_projects(S.projects_dir)
            proj_dd = gr.Dropdown(label="Проекты", choices=_projects, value=_projects[0] if _projects else None, scale=3)
            open_btn = gr.Button("📂 Открыть", scale=1)
            refresh_btn = gr.Button("🔄", scale=0, min_width=50)
        status_md = gr.Markdown(_project_status_md())
        with gr.Accordion("⚙ Настройки разбора текста (для вкладок «Текст из PDF» и «Книга»)", open=False):
            with gr.Row():
                first_page = gr.Number(label="С страницы", value=1, precision=0)
                last_page = gr.Number(label="По страницу (0 = до конца)", value=0, precision=0)
            chapter_mode = gr.Dropdown(label="Главы", choices=list(CHAPTER_MODES), value=list(CHAPTER_MODES)[0])
            rm_headers = gr.Checkbox(label="Удалять колонтитулы и номера страниц", value=True)
            rm_foot = gr.Checkbox(label="Удалять сноски", value=True)
            skip_toc = gr.Checkbox(label="Пропускать страницы оглавления", value=True)
            skip_service = gr.Checkbox(label="Пропускать выходные данные (УДК, ББК, ISBN, ©)", value=True)
            skip_front = gr.Checkbox(
                label="Пропускать начало книги до «Предисловия / Введения / Раздела / Главы» (титул, авторы, сокращения)",
                value=True,
            )
            lang = gr.Radio(label="Язык", choices=["ru", "en"], value="ru")
            numbers = gr.Checkbox(label="Числа → слова (обычно не нужно, модель читает цифры сама)", value=False)
            abbr = gr.Checkbox(label="Раскрывать сокращения (т. е., и т. д., см. …)", value=True)
            target = gr.Slider(120, 400, value=230, step=10, label="Целевая длина сегмента, симв.")
            maxc = gr.Slider(200, 600, value=380, step=10, label="Максимальная длина сегмента, симв.")
            minc = gr.Slider(10, 120, value=50, step=5, label="Минимальная длина (короткие реплики объединяются)")
            titles = gr.Checkbox(label="Читать названия глав", value=True)
            merge = gr.Checkbox(label="Объединять короткие абзацы-реплики", value=True)
            mark_defs = gr.Checkbox(
                label="Определения «**Термин** — текст»: пауза после термина", value=True
            )
            replacements = gr.Textbox(
                label="Словарь замен (строка «что=на что»)",
                lines=4,
                placeholder="ГК РФ=Гражданский кодекс Российской Федерации\nНК РФ=Налоговый кодекс Российской Федерации",
            )


        # ---------------------------------------------------- Текст из PDF
        with gr.Tab("0 · Текст из PDF"):
            gr.Markdown(
                "Только обработка текста — без модели и видеокарты. Загрузите PDF/TXT и получите **разметку** "
                "(`#` раздел, `##` глава, `###` §, `####` подзаголовок, `**термин**` — определение) и "
                "**предпросмотр озвучки**: текст ровно так, как его прочитает модель, с фрагментами и паузами. "
                "Разметку можно поправить прямо здесь и создать из неё проект."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    tp_file = gr.File(label="PDF или TXT", file_types=[".pdf", ".txt", ".md"], type="filepath")
                    tp_btn = gr.Button("🔎 Обработать", variant="primary")
                    tp_summary = gr.Markdown()
                    tp_files = gr.File(label="Скачать", file_count="multiple", interactive=False)
                    tp_name = gr.Textbox(label="Название проекта", placeholder="по умолчанию — имя файла")
                    tp_create_btn = gr.Button("📥 Создать проект из этого текста")
                with gr.Column(scale=2):
                    with gr.Tab("Разметка (можно править)"):
                        tp_markup = gr.Textbox(lines=26, max_lines=40, show_label=False)
                        tp_refresh_btn = gr.Button("↻ Обновить предпросмотр по правленому тексту")
                    with gr.Tab("Как прочитает модель"):
                        tp_speech = gr.Textbox(lines=28, max_lines=40, show_label=False, interactive=False)

        # ------------------------------------------------------------ Книга
        with gr.Tab("1 · Книга"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="PDF или TXT", file_types=[".pdf", ".txt", ".md"], type="filepath")
                    name_in = gr.Textbox(label="Название проекта (необязательно)")
                    create_btn = gr.Button("📥 Создать проект и разобрать", variant="primary")
                    with gr.Row():
                        reparse_btn = gr.Button("♻ Перераспределить сегменты")
                        reextract_btn = gr.Button("📄 Извлечь из файла заново")
                with gr.Column(scale=2):
                    parse_msg = gr.Markdown()
                    chapters_df = gr.Dataframe(
                        headers=["№", "Уровень", "Название", "Сегм.", "Симв.", "≈ Длит.", "Озвучено"], value=_chapters_table(), interactive=False, wrap=True
                    )
            with gr.Accordion("✏ Редактировать текст книги (# Заголовок главы, пустая строка между абзацами)", open=False):
                book_text = gr.Textbox(lines=24, max_lines=40, show_label=False)
                apply_text_btn = gr.Button("Применить правки")
            with gr.Accordion("Сегменты (первые 400)", open=False):
                segs_df = gr.Dataframe(headers=["#", "Гл.", "", "Симв.", "Текст", "Заметка"], value=_segments_table(), interactive=False, wrap=True)

        parse_settings = [first_page, last_page, chapter_mode, rm_headers, rm_foot, skip_toc, skip_service, lang, numbers, abbr, target, maxc, minc, titles, merge, mark_defs, replacements, skip_front]
        parse_out = [parse_msg, chapters_df, segs_df, book_text, status_md, proj_dd]

        # ------------------------------------------------------------ Голос
        with gr.Tab("2 · Голос"):
            voice_mode = gr.Radio(label="Источник голоса", choices=VOICE_MODES, value=VOICE_MODES[0])
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        "**Клонирование.** Образец 5–30 с (лучше 10–20 с) чистой речи без музыки и эха. "
                        "**Hi-Fi:** впишите *точный* текст образца — модель продолжит речь из образца "
                        "(reference + prompt + transcript), похожесть максимальная; инструкция стиля тогда игнорируется."
                    )
                    with gr.Row():
                        lib_dd = gr.Dropdown(label="📚 Библиотека голосов", choices=list_voices(S.voices_dir), scale=3)
                        lib_load_btn = gr.Button("Взять", scale=1)
                        lib_refresh_btn = gr.Button("🔄", scale=0, min_width=50)
                    ref_audio = gr.Audio(label="Образец голоса", type="filepath", sources=["upload", "microphone"])
                    ref_info = gr.Markdown()
                    ref_text = gr.Textbox(
                        label="Точный текст образца → Hi-Fi клонирование (дословно, со знаками препинания)", lines=3
                    )
                    with gr.Row():
                        lib_name = gr.Textbox(label="Имя голоса", placeholder="Например: Диктор_мужской", scale=3)
                        lib_save_btn = gr.Button("💾 В библиотеку", scale=1)
                    lib_msg = gr.Markdown()
                    denoise = gr.Checkbox(label="Шумоподавление образца (ZipEnhancer, нужен modelscope)", value=False)
                    style = gr.Textbox(
                        label="Инструкция стиля (не работает вместе с текстом образца)",
                        placeholder="спокойно, размеренно, с лёгкой теплотой",
                    )
                with gr.Column():
                    gr.Markdown(
                        "**Дизайн голоса по описанию.** Сгенерируйте образец, послушайте, "
                        "и если понравился — сделайте его голосом книги. Так тембр будет одинаковым во всей книге."
                    )
                    design_desc = gr.Textbox(label="Описание голоса", value=DESIGN_DEFAULT, lines=2)
                    design_text = gr.Textbox(label="Текст образца", value=DESIGN_SAMPLE, lines=2)
                    design_btn = gr.Button("🎲 Сгенерировать образец")
                    design_audio = gr.Audio(label="Образец", type="filepath", interactive=False)
                    design_msg = gr.Markdown()
                    use_design_btn = gr.Button("✔ Использовать этот голос для книги")
            with gr.Row():
                preset = gr.Radio(label="Качество / скорость", choices=list(PRESETS), value="Баланс (рекомендуется)")
                cfg = gr.Slider(1.0, 3.5, value=2.0, step=0.1, label="CFG (выразительность ↔ стабильность)")
                seed = gr.Number(label="Seed", value=42, precision=0)
            gr.Markdown("CFG: если на длинной книге появляется шум/«жужжание» или ускорение — уменьшите до 1.5–1.6.")
            save_voice_btn = gr.Button("💾 Сохранить голос в проект", variant="primary")
            voice_msg = gr.Markdown()
            with gr.Row():
                test_text = gr.Textbox(label="Фраза для проверки (пусто = первый абзац книги)", lines=2, scale=3)
                test_btn = gr.Button("🔊 Прослушать", scale=1)
            test_audio = gr.Audio(label="Проверка", type="filepath", interactive=False)
            test_msg = gr.Markdown()

        voice_inputs = [voice_mode, ref_audio, ref_text, style, denoise, preset, cfg, seed]

        # ----------------------------------------------------------- Озвучка
        with gr.Tab("3 · Озвучка"):
            with gr.Row():
                start_btn = gr.Button("▶ Начать / продолжить", variant="primary")
                stop_btn = gr.Button("⏹ Стоп")
                redo_flag_btn = gr.Button("↻ Переозвучить помеченные")
            job_msg = gr.Markdown()
            progress_html = gr.HTML(_progress_html())
            log_box = gr.Textbox(label="Журнал", lines=12, max_lines=12, autoscroll=True)
            with gr.Accordion("🔍 Проверка сегментов", open=True):
                with gr.Row():
                    seg_idx = gr.Number(label="Сегмент №", value=0, precision=0)
                    show_btn = gr.Button("Показать")
                    redo_btn = gr.Button("↻ Переозвучить (новый seed)")
                    ok_btn = gr.Button("✔ Нормально")
                seg_info = gr.Markdown()
                seg_text = gr.Textbox(label="Текст сегмента", lines=3)
                save_seg_btn = gr.Button("Сохранить текст сегмента")
                seg_audio = gr.Audio(label="Аудио сегмента", type="filepath", interactive=False)
                problems_btn = gr.Button("Показать помеченные / ошибки")
                problems_df = gr.Dataframe(headers=["#", "Гл.", "", "Симв.", "Текст", "Заметка"], interactive=False, wrap=True)

        # ------------------------------------------------------------ Сборка
        with gr.Tab("4 · Сборка"):
            formats = gr.CheckboxGroup(
                label="Форматы",
                choices=["MP3 (отдельные файлы)", "M4B (одним файлом, с главами)", "WAV (отдельные файлы)"],
                value=["MP3 (отдельные файлы)", "M4B (одним файлом, с главами)"],
            )
            with gr.Row():
                split_by = gr.Radio(
                    label="Отдельные файлы по", choices=["Разделам", "Главам", "Параграфам (§)"], value="Параграфам (§)"
                )
                marks_by = gr.Radio(
                    label="Метки глав внутри M4B по", choices=["Разделам", "Главам", "Параграфам (§)"], value="Параграфам (§)"
                )
            with gr.Row():
                mp3_br = gr.Dropdown(label="Битрейт MP3", choices=["128k", "192k", "256k", "320k"], value="320k")
                aac_br = gr.Dropdown(label="Битрейт AAC (M4B)", choices=["96k", "128k", "192k", "256k"], value="256k")
                out_sr = gr.Dropdown(label="Частота", choices=["48000", "44100", "24000"], value="48000")
            folders = gr.Checkbox(label="Раскладывать MP3 по папкам: Раздел / Глава / файлы §", value=True)
            max_part = gr.Number(
                label="Делить M4B на части до, МБ (0 — одним файлом; Telegram: 1900, Telegram Premium: 3900)",
                value=0,
                precision=0,
            )
            allow_missing = gr.Checkbox(label="Собрать, даже если не все сегменты озвучены", value=False)
            with gr.Row():
                export_btn = gr.Button("📦 Собрать аудиокнигу", variant="primary", scale=2)
                iphone_btn = gr.Button("📱 Для iPhone (M4B 96k, части ≤ 1.9 ГБ)", scale=1)
            export_files = gr.File(label="Результат", file_count="multiple")
            export_log = gr.Textbox(label="Журнал сборки", lines=8)
            export_msg = gr.Markdown()

        # ----------------------------------------------------------- События
        refresh_btn.click(lambda: gr.update(choices=list_projects(S.projects_dir)), outputs=proj_dd)
        tp_btn.click(on_text_process, inputs=[tp_file] + parse_settings, outputs=[tp_summary, tp_markup, tp_speech, tp_files])
        tp_refresh_btn.click(on_text_refresh, inputs=[tp_markup, tp_file] + parse_settings, outputs=[tp_summary, tp_speech, tp_files])
        tp_create_btn.click(on_text_create_project, inputs=[tp_markup, tp_file, tp_name] + parse_settings, outputs=parse_out)
        open_btn.click(on_open, inputs=proj_dd, outputs=parse_out + voice_inputs + parse_settings)
        create_btn.click(on_create, inputs=[file_in, name_in] + parse_settings, outputs=parse_out)
        reparse_btn.click(on_resegment, inputs=parse_settings, outputs=parse_out)
        reextract_btn.click(on_reextract, inputs=parse_settings, outputs=parse_out)
        apply_text_btn.click(on_apply_text, inputs=[book_text] + parse_settings, outputs=parse_out)

        ref_audio.change(on_ref_change, inputs=ref_audio, outputs=ref_info)
        lib_save_btn.click(on_lib_save, inputs=[lib_name, ref_audio, ref_text, style], outputs=[lib_msg, lib_dd])
        lib_load_btn.click(on_lib_load, inputs=lib_dd, outputs=[voice_mode, ref_audio, ref_text, style, lib_name, lib_msg])
        lib_refresh_btn.click(lambda: gr.update(choices=list_voices(S.voices_dir)), outputs=lib_dd)
        save_voice_btn.click(on_save_voice, inputs=voice_inputs, outputs=[voice_msg, status_md])
        test_btn.click(on_test_voice, inputs=[test_text] + voice_inputs, outputs=[test_audio, test_msg])
        design_btn.click(on_design_sample, inputs=[design_desc, design_text, cfg, preset, seed], outputs=[design_audio, design_msg])
        use_design_btn.click(
            on_use_design, inputs=[design_audio, design_text], outputs=[voice_mode, ref_audio, ref_text, style, design_msg]
        )

        start_btn.click(on_start, outputs=job_msg)
        stop_btn.click(on_stop, outputs=job_msg)
        redo_flag_btn.click(on_redo_flagged, outputs=job_msg)
        show_btn.click(on_show_segment, inputs=seg_idx, outputs=[seg_text, seg_audio, seg_info])
        redo_btn.click(on_redo_segment, inputs=seg_idx, outputs=job_msg)
        ok_btn.click(on_mark_ok, inputs=seg_idx, outputs=[job_msg, problems_df])
        save_seg_btn.click(on_save_segment_text, inputs=[seg_idx, seg_text], outputs=[job_msg, segs_df])
        problems_btn.click(lambda: _segments_table(limit=1000, only_problems=True), outputs=problems_df)

        export_btn.click(
            on_export,
            inputs=[formats, split_by, marks_by, mp3_br, aac_br, out_sr, folders, allow_missing, max_part],
            outputs=[export_files, export_log, export_msg],
        )
        iphone_btn.click(
            on_export_iphone,
            inputs=[allow_missing],
            outputs=[export_files, export_log, export_msg, formats, marks_by, aac_br, out_sr, max_part],
        )

        # При запуске открываем последний проект (если есть)
        def _autoload(name):
            if name and S.project is None and not S.busy:
                try:
                    return on_open(name)
                except Exception:
                    pass
            return [gr.update()] * (len(parse_out) + len(voice_inputs) + len(parse_settings))

        demo.load(_autoload, inputs=proj_dd, outputs=parse_out + voice_inputs + parse_settings)

        timer = gr.Timer(2.0)
        timer.tick(on_tick, outputs=[progress_html, log_box, status_md])
    return demo


def launch(
    host: str = "127.0.0.1",
    port: int = 7861,
    share: bool = False,
    model: str = "models/VoxCPM2",
    projects_dir: str = "projects",
    compile_model: bool = True,
    engine_cfg=None,
):
    global S
    S = AppState(projects_dir, model, compile_model, engine_cfg=engine_cfg)
    demo = build_ui()
    demo.queue(default_concurrency_limit=4)
    print(f"Открой в браузере: http://{host}:{port}", flush=True)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=True,
        allowed_paths=[str(S.projects_dir.resolve()), str(S.voices_dir.resolve()), tempfile.gettempdir()],
        theme=gr.themes.Soft(),
    )
