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

from .audio import ExportOptions, fmt_duration, postprocess
from .book import Book
from .engine import EngineConfig, SynthConfig, TTSEngine, VoiceConfig
from .normalize import NormalizeOptions, normalize_text
from .pdf_extract import ExtractOptions
from .pipeline import SynthesisJob, export_book, parse_into_project
from .project import Project, list_projects
from .segmenter import SegmentOptions, estimate_duration_sec

PRESETS = {"Быстро (черновик)": 6, "Баланс (рекомендуется)": 10, "Качество": 16}
CHAPTER_MODES = {
    "Авто (оглавление → шрифт → шаблоны)": "auto",
    "Оглавление PDF": "toc",
    "Крупный шрифт": "font",
    "Шаблоны «Глава N»": "regex",
    "Не делить на главы": "none",
}
VOICE_MODES = ["Клонировать по образцу", "Создать по описанию", "Голос модели по умолчанию"]
DESIGN_SAMPLE = (
    "Добрый вечер. Сегодня я расскажу вам историю, которая началась холодным осенним утром, "
    "когда над городом ещё висел густой туман."
)
DESIGN_DEFAULT = "Мужчина средних лет, тёплый бархатный низкий голос, спокойная размеренная манера чтения аудиокниги"


class AppState:
    def __init__(self, projects_dir: str, model: str, compile_model: bool):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.engine = TTSEngine(EngineConfig(model_path=model, compile=compile_model), log=self._log)
        self.engine_lock = threading.Lock()
        self.project: Optional[Project] = None
        self.job: Optional[SynthesisJob] = None
        self.messages: list[str] = []

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


S: AppState  # инициализируется в launch()


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
    for ci, title in enumerate(p.chapter_titles):
        segs = p.chapter_segments(ci)
        chars = sum(len(s.text) for s in segs)
        done = sum(1 for s in segs if s.status in ("done", "flagged"))
        rows.append([ci + 1, title[:80], len(segs), chars, fmt_duration(estimate_duration_sec(chars)), f"{done}/{len(segs)}"])
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


def _settings_from_ui(first_page, last_page, chapter_mode, rm_headers, rm_foot, lang, numbers, abbr, target, maxc, minc, titles, merge):
    p = _need_project()
    st = p.settings
    st.extract = ExtractOptions(
        first_page=int(first_page or 1),
        last_page=int(last_page) if last_page else None,
        remove_headers=bool(rm_headers),
        remove_footnotes=bool(rm_foot),
        chapter_mode=CHAPTER_MODES.get(chapter_mode, "auto"),
    )
    st.normalize = NormalizeOptions(lang=lang, numbers_to_words=bool(numbers), expand_abbreviations=bool(abbr))
    maxc = max(int(maxc), int(target) + 20)
    st.segment = SegmentOptions(
        target_chars=int(target),
        max_chars=maxc,
        min_chars=int(minc),
        read_chapter_titles=bool(titles),
        merge_short_paragraphs=bool(merge),
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
    return out + list(_voice_ui_values())


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


def on_export(formats, mp3_br, aac_br, sr, allow_missing, progress=gr.Progress()):
    p = _need_project()
    if S.busy:
        raise gr.Error("Дождитесь окончания озвучки или остановите её.")
    fm = []
    if "MP3 по главам" in formats:
        fm.append("mp3")
    if "M4B (одним файлом, с главами)" in formats:
        fm.append("m4b")
    if "WAV по главам" in formats:
        fm.append("wav")
    if not fm:
        raise gr.Error("Выберите хотя бы один формат.")
    p.settings.export = ExportOptions(formats=tuple(fm), mp3_bitrate=mp3_br, aac_bitrate=aac_br, sample_rate=int(sr))
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
            proj_dd = gr.Dropdown(label="Проекты", choices=list_projects(S.projects_dir), scale=3)
            open_btn = gr.Button("📂 Открыть", scale=1)
            refresh_btn = gr.Button("🔄", scale=0, min_width=50)
        status_md = gr.Markdown(_project_status_md())

        # ------------------------------------------------------------ Книга
        with gr.Tab("1 · Книга"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="PDF или TXT", file_types=[".pdf", ".txt", ".md"], type="filepath")
                    name_in = gr.Textbox(label="Название проекта (необязательно)")
                    with gr.Accordion("Настройки разбора", open=False):
                        with gr.Row():
                            first_page = gr.Number(label="С страницы", value=1, precision=0)
                            last_page = gr.Number(label="По страницу (0 = до конца)", value=0, precision=0)
                        chapter_mode = gr.Dropdown(label="Главы", choices=list(CHAPTER_MODES), value=list(CHAPTER_MODES)[0])
                        rm_headers = gr.Checkbox(label="Удалять колонтитулы и номера страниц", value=True)
                        rm_foot = gr.Checkbox(label="Удалять сноски", value=True)
                        lang = gr.Radio(label="Язык", choices=["ru", "en"], value="ru")
                        numbers = gr.Checkbox(label="Числа → слова (обычно не нужно, модель читает цифры сама)", value=False)
                        abbr = gr.Checkbox(label="Раскрывать сокращения (т. е., и т. д., см. …)", value=True)
                        target = gr.Slider(120, 400, value=230, step=10, label="Целевая длина сегмента, симв.")
                        maxc = gr.Slider(200, 600, value=380, step=10, label="Максимальная длина сегмента, симв.")
                        minc = gr.Slider(10, 120, value=50, step=5, label="Минимальная длина (короткие реплики объединяются)")
                        titles = gr.Checkbox(label="Читать названия глав", value=True)
                        merge = gr.Checkbox(label="Объединять короткие абзацы-реплики", value=True)
                    create_btn = gr.Button("📥 Создать проект и разобрать", variant="primary")
                    with gr.Row():
                        reparse_btn = gr.Button("♻ Перераспределить сегменты")
                        reextract_btn = gr.Button("📄 Извлечь из файла заново")
                with gr.Column(scale=2):
                    parse_msg = gr.Markdown()
                    chapters_df = gr.Dataframe(
                        headers=["№", "Глава", "Сегм.", "Симв.", "≈ Длит.", "Озвучено"], value=_chapters_table(), interactive=False, wrap=True
                    )
            with gr.Accordion("✏ Редактировать текст книги (# Заголовок главы, пустая строка между абзацами)", open=False):
                book_text = gr.Textbox(lines=24, max_lines=40, show_label=False)
                apply_text_btn = gr.Button("Применить правки")
            with gr.Accordion("Сегменты (первые 400)", open=False):
                segs_df = gr.Dataframe(headers=["#", "Гл.", "", "Симв.", "Текст", "Заметка"], value=_segments_table(), interactive=False, wrap=True)

        parse_settings = [first_page, last_page, chapter_mode, rm_headers, rm_foot, lang, numbers, abbr, target, maxc, minc, titles, merge]
        parse_out = [parse_msg, chapters_df, segs_df, book_text, status_md, proj_dd]

        # ------------------------------------------------------------ Голос
        with gr.Tab("2 · Голос"):
            voice_mode = gr.Radio(label="Источник голоса", choices=VOICE_MODES, value=VOICE_MODES[0])
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        "**Клонирование.** Образец 10–20 с чистой речи без музыки. "
                        "Если указать точный текст образца — похожесть максимальная (режим продолжения)."
                    )
                    ref_audio = gr.Audio(label="Образец голоса", type="filepath", sources=["upload", "microphone"])
                    ref_text = gr.Textbox(label="Точный текст образца (необязательно, но рекомендуется)", lines=2)
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
                choices=["MP3 по главам", "M4B (одним файлом, с главами)", "WAV по главам"],
                value=["MP3 по главам", "M4B (одним файлом, с главами)"],
            )
            with gr.Row():
                mp3_br = gr.Dropdown(label="Битрейт MP3", choices=["64k", "96k", "128k", "192k"], value="128k")
                aac_br = gr.Dropdown(label="Битрейт AAC (M4B)", choices=["48k", "64k", "96k", "128k"], value="96k")
                out_sr = gr.Dropdown(label="Частота", choices=["48000", "44100", "24000"], value="48000")
            allow_missing = gr.Checkbox(label="Собрать, даже если не все сегменты озвучены", value=False)
            export_btn = gr.Button("📦 Собрать аудиокнигу", variant="primary")
            export_files = gr.File(label="Результат", file_count="multiple")
            export_log = gr.Textbox(label="Журнал сборки", lines=8)
            export_msg = gr.Markdown()

        # ----------------------------------------------------------- События
        refresh_btn.click(lambda: gr.update(choices=list_projects(S.projects_dir)), outputs=proj_dd)
        open_btn.click(on_open, inputs=proj_dd, outputs=parse_out + voice_inputs)
        create_btn.click(on_create, inputs=[file_in, name_in] + parse_settings, outputs=parse_out)
        reparse_btn.click(on_resegment, inputs=parse_settings, outputs=parse_out)
        reextract_btn.click(on_reextract, inputs=parse_settings, outputs=parse_out)
        apply_text_btn.click(on_apply_text, inputs=[book_text] + parse_settings, outputs=parse_out)

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

        export_btn.click(on_export, inputs=[formats, mp3_br, aac_br, out_sr, allow_missing], outputs=[export_files, export_log, export_msg])

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
):
    global S
    S = AppState(projects_dir, model, compile_model)
    demo = build_ui()
    demo.queue(default_concurrency_limit=4)
    print(f"Открой в браузере: http://{host}:{port}", flush=True)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=True,
        allowed_paths=[str(S.projects_dir.resolve()), tempfile.gettempdir()],
        theme=gr.themes.Soft(),
    )
