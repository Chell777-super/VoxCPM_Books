"""Командная строка.

Примеры::

    python -m audiobook make книга.pdf --ref голос.wav --ref-text "Текст образца." --formats mp3,m4b
    python -m audiobook new книга.pdf              # создать проект и разбить текст
    python -m audiobook text  книга --apply book.txt  # применить отредактированный текст
    python -m audiobook synth книга --ref голос.wav  # озвучка (продолжает с места остановки)
    python -m audiobook export книга --formats m4b
    python -m audiobook ui                          # веб-интерфейс
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from .audio import fmt_duration
from .book import Book
from .engine import EngineConfig, TTSEngine
from .pipeline import SynthesisJob, export_book, parse_into_project
from .project import Project

DEFAULT_PROJECTS = os.environ.get("VOXBOOKS_PROJECTS", "projects")
DEFAULT_MODEL = os.environ.get("VOXBOOKS_MODEL", "models/VoxCPM2")


def _open_project(name: str, projects_dir: str) -> Project:
    p = Path(name)
    if not (p / "project.json").exists():
        p = Path(projects_dir) / name
    if not (p / "project.json").exists():
        sys.exit(f"Проект не найден: {name}")
    return Project(p)


def _print_stats(proj: Project) -> None:
    s = proj.stats()
    print(
        f"Проект «{proj.name}»: глав {s['chapters']}, сегментов {s['total']}, символов {s['chars']:,}. "
        f"Озвучено {s['done']}/{s['total']} ({fmt_duration(s['audio_sec'])}), на проверку {s['flagged']}, ошибок {s['errors']}."
    )


def _apply_parse_args(proj: Project, a) -> None:
    st = proj.settings
    if getattr(a, "first_page", None):
        st.extract.first_page = a.first_page
    if getattr(a, "last_page", None):
        st.extract.last_page = a.last_page
    if getattr(a, "chapters", None):
        st.extract.chapter_mode = a.chapters
    if getattr(a, "target_chars", None):
        st.segment.target_chars = a.target_chars
    if getattr(a, "max_chars", None):
        st.segment.max_chars = a.max_chars
    if getattr(a, "numbers", False):
        st.normalize.numbers_to_words = True
    if getattr(a, "lang", None):
        st.normalize.lang = a.lang


def _apply_voice_args(proj: Project, a) -> bool:
    """Возвращает True, если голос изменился."""
    v = proj.settings.voice
    before = (v.mode, v.reference_wav, v.prompt_text, v.style)
    if getattr(a, "ref", None):
        proj.voice_dir.mkdir(parents=True, exist_ok=True)
        dst = proj.voice_dir / ("reference" + Path(a.ref).suffix.lower())
        if Path(a.ref).resolve() != dst.resolve():
            shutil.copyfile(a.ref, dst)
        v.mode, v.reference_wav = "clone", str(dst)
    if getattr(a, "ref_text", None) is not None:
        v.prompt_text = a.ref_text or None
    if getattr(a, "style", None) is not None:
        v.style = a.style
    if getattr(a, "denoise", False):
        v.denoise = True
    if not v.reference_wav:
        v.mode = "default"
    s = proj.settings.synth
    if getattr(a, "timesteps", None):
        s.inference_timesteps = a.timesteps
    if getattr(a, "cfg", None):
        s.cfg_value = a.cfg
    if getattr(a, "seed", None) is not None:
        s.seed = a.seed
    changed = before != (v.mode, v.reference_wav, v.prompt_text, v.style)
    if changed:
        proj.rekey_for_voice()
    proj.save()
    return changed


def _run_synth(proj: Project, a) -> None:
    if proj.settings.voice.mode != "clone":
        print(
            "⚠ Образец голоса не задан (--ref). Без образца тембр будет немного меняться между сегментами. "
            "Рекомендуется: --ref voice.wav (10–20 с чистой речи)."
        )
    engine = TTSEngine(EngineConfig(model_path=a.model, compile=not a.no_compile), log=lambda m: print(m, flush=True))
    indices = None
    redo = False
    if getattr(a, "redo_flagged", False):
        indices = [s.idx for s in proj.segments if s.status in ("flagged", "error")]
        redo = True
    job = SynthesisJob(proj, engine, indices=indices, redo=redo, reseed=redo, log=lambda m: print(m, flush=True))

    def on_sigint(*_):
        print("\nОстанавливаю после текущего сегмента… (повторное Ctrl+C — немедленный выход)", flush=True)
        job.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, on_sigint)
    job.start()
    while job.thread.is_alive():
        time.sleep(0.5)
    _print_stats(proj)


def cmd_new(a):
    proj = Project.create(a.projects_dir, a.file, name=a.name)
    _apply_parse_args(proj, a)
    parse_into_project(proj)
    _print_stats(proj)
    print(f"Текст для проверки/правки: {proj.root / 'book.txt'}")
    return proj


def cmd_text(a):
    proj = _open_project(a.project, a.projects_dir)
    _apply_parse_args(proj, a)
    if a.apply:
        raw = Path(a.apply).read_text(encoding="utf-8")
        book = Book.from_text(raw, default_title=proj.book.title if proj.book else "")
        if proj.book and not book.author:
            book.author = proj.book.author
        parse_into_project(proj, book=book)
    else:
        parse_into_project(proj)
    _print_stats(proj)


def cmd_synth(a):
    proj = _open_project(a.project, a.projects_dir)
    _apply_voice_args(proj, a)
    _run_synth(proj, a)


def cmd_export(a):
    proj = _open_project(a.project, a.projects_dir)
    if a.formats:
        proj.settings.export.formats = tuple(x.strip() for x in a.formats.split(",") if x.strip())
    if a.bitrate:
        proj.settings.export.mp3_bitrate = a.bitrate
    levels = {"part": 1, "chapter": 2, "section": 3}
    if getattr(a, "split_by", None):
        proj.settings.export.split_level = levels[a.split_by]
    if getattr(a, "marks_by", None):
        proj.settings.export.marker_level = levels[a.marks_by]
    proj.save()
    outs = export_book(proj, allow_missing=a.allow_missing)
    for o in outs:
        print(o)


def cmd_make(a):
    proj = cmd_new(a)
    _apply_voice_args(proj, a)
    _run_synth(proj, a)
    a.project = str(proj.root)
    cmd_export(a)


def cmd_status(a):
    proj = _open_project(a.project, a.projects_dir)
    _print_stats(proj)
    for s in proj.segments:
        if s.status in ("flagged", "error"):
            print(f"  #{s.idx} [{s.status}] {s.note}: {s.text[:80]}")


def cmd_ui(a):
    from .webui import launch

    launch(host=a.host, port=a.port, share=a.share, model=a.model, projects_dir=a.projects_dir, compile_model=not a.no_compile)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="audiobook", description="VoxCPM Books — аудиокниги из PDF")
    ap.add_argument("--projects-dir", default=DEFAULT_PROJECTS)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="папка модели VoxCPM2 или HF id")
    ap.add_argument("--no-compile", action="store_true", help="не использовать torch.compile")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def parse_opts(p):
        p.add_argument("--first-page", type=int)
        p.add_argument("--last-page", type=int)
        p.add_argument("--chapters", choices=["auto", "toc", "font", "regex", "none"])
        p.add_argument("--target-chars", type=int)
        p.add_argument("--max-chars", type=int)
        p.add_argument("--numbers", action="store_true", help="числа → слова")
        p.add_argument("--lang", choices=["ru", "en"])

    def voice_opts(p):
        p.add_argument("--ref", help="образец голоса (wav/mp3/flac), 10–20 с")
        p.add_argument("--ref-text", help="точный текст образца → максимальная похожесть")
        p.add_argument("--style", help="инструкция стиля, напр. «спокойно, размеренно»")
        p.add_argument("--denoise", action="store_true")
        p.add_argument("--timesteps", type=int, help="шаги диффузии: 6 быстро / 10 баланс / 16 качество")
        p.add_argument("--cfg", type=float)
        p.add_argument("--seed", type=int)

    def export_opts(p):
        p.add_argument("--formats", help="mp3,m4b,wav")
        p.add_argument("--bitrate", help="битрейт MP3, напр. 128k")
        p.add_argument("--allow-missing", action="store_true")
        p.add_argument("--split-by", choices=["part", "chapter", "section"], help="файлы по разделам/главам/§ (по умолч. главам)")
        p.add_argument("--marks-by", choices=["part", "chapter", "section"], help="метки в M4B (по умолч. §)")

    p = sub.add_parser("new", help="создать проект из PDF/TXT")
    p.add_argument("file")
    p.add_argument("--name")
    parse_opts(p)
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("text", help="перераспределить текст (или применить правленый book.txt)")
    p.add_argument("project")
    p.add_argument("--apply", help="путь к отредактированному book.txt")
    parse_opts(p)
    p.set_defaults(func=cmd_text)

    p = sub.add_parser("synth", help="озвучить (продолжает с места остановки)")
    p.add_argument("project")
    p.add_argument("--redo-flagged", action="store_true", help="переозвучить помеченные/ошибочные")
    voice_opts(p)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("export", help="собрать MP3/M4B")
    p.add_argument("project")
    export_opts(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("make", help="всё сразу: PDF → аудиокнига")
    p.add_argument("file")
    p.add_argument("--name")
    parse_opts(p)
    voice_opts(p)
    export_opts(p)
    p.set_defaults(func=cmd_make)

    p = sub.add_parser("status", help="состояние проекта")
    p.add_argument("project")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("ui", help="веб-интерфейс")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true")
    p.set_defaults(func=cmd_ui)
    return ap


def main(argv=None):
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
