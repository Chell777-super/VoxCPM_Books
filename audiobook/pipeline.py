"""Конвейер: разбор книги → синтез (GPU) → запись (CPU-поток) → сборка.

GPU никогда не ждёт диск: синтез идёт в одном потоке, а пост-обработка
(обрезка тишины, нормализация громкости, FLAC-кодирование) и сохранение
манифеста — в отдельном потоке-писателе через очередь.
"""

from __future__ import annotations

import queue
import re
import statistics
import threading
import time
import traceback
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .audio import ExportOptions, assemble_chapter, build_m4b, encode_mp3, fmt_duration, postprocess, write_segment
from .book import Book, safe_filename
from .engine import SynthConfig, TTSEngine
from .pdf_extract import extract_any, render_cover
from .project import Project, SegState
from .segmenter import segment_book

LogFn = Callable[[str], None]


# --------------------------------------------------------------------------- #
# Разбор
# --------------------------------------------------------------------------- #


def parse_into_project(project: Project, book: Optional[Book] = None, log: LogFn = print) -> dict:
    """Извлекает текст (если book не передан) и разбивает на сегменты."""
    st = project.settings
    report = None
    if book is None:
        src = project.source_file
        if src is None:
            raise FileNotFoundError("В проекте нет исходного файла")
        book, report = extract_any(str(src), st.extract)
        if src.suffix == ".pdf" and not project.cover_path.exists():
            render_cover(str(src), str(project.cover_path))
        if not book.title:
            book.title = project.name.replace("_", " ")
    segs = segment_book(book, st.segment, st.normalize)
    project.set_segments(book, segs)
    info = project.stats()
    if report:
        info["report"] = report
        for w in report.warnings:
            log("⚠ " + w)
    return info


# --------------------------------------------------------------------------- #
# Синтез
# --------------------------------------------------------------------------- #


@dataclass
class JobState:
    running: bool = False
    total: int = 0
    done: int = 0
    skipped: int = 0
    flagged: int = 0
    errors: int = 0
    started: float = 0.0
    gen_time: float = 0.0
    audio_time: float = 0.0
    current: str = ""
    finished_msg: str = ""
    log: List[str] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        return self.gen_time / self.audio_time if self.audio_time else 0.0

    def eta(self, remaining_chars: int, cps: float) -> float:
        if not self.audio_time or not cps:
            return 0.0
        return remaining_chars / cps * self.rtf


class SynthesisJob:
    def __init__(
        self,
        project: Project,
        engine: TTSEngine,
        synth: Optional[SynthConfig] = None,
        indices: Optional[List[int]] = None,
        redo: bool = False,
        reseed: bool = False,
        log: Optional[LogFn] = None,
    ):
        self.project = project
        self.engine = engine
        self.synth = synth or project.settings.synth
        self.indices = indices
        self.redo = redo
        self.reseed = reseed
        self.state = JobState()
        self._stop = threading.Event()
        # большой буфер: в пакетном режиме (Nano-vLLM) результаты приходят пачками, а поток
        # событий движка нельзя блокировать — иначе встанет видеокарта
        self._q: "queue.Queue" = queue.Queue(maxsize=512)
        self._ext_log = log
        self.thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- control
    def log(self, msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        self.state.log.append(line)
        del self.state.log[:-400]
        if self._ext_log:
            self._ext_log(msg)

    def start(self) -> "SynthesisJob":
        self.thread = threading.Thread(target=self.run, daemon=True, name="synthesis")
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    # ---------------------------------------------------------------- work
    def _todo(self) -> List[SegState]:
        segs = self.project.segments
        if self.indices is not None:
            chosen = [segs[i] for i in self.indices if 0 <= i < len(segs)]
            return chosen if self.redo else [s for s in chosen if s.status not in ("done", "flagged")]
        return [s for s in segs if s.status not in ("done", "flagged")]

    def _seed_for(self, s: SegState) -> int:
        base = self.synth.seed + (zlib.crc32(s.text.encode("utf-8")) % 100000)
        if self.reseed:
            base += int(time.time()) % 9973 + 1
        return base

    def _writer(self):
        last_save = time.time()
        while True:
            item = self._q.get()
            if item is None:
                break
            seg, res = item
            try:
                wav = postprocess(res.wav, res.sr, target_db=self.project.settings.export.target_db)
                self.project.audio_dir.mkdir(parents=True, exist_ok=True)
                write_segment(self.project.seg_path(seg), wav, res.sr)
                with self.project.lock:
                    seg.status = "done" if res.ok else "flagged"
                    seg.seconds = len(wav) / res.sr
                    seg.cps = round(res.cps, 2)
                    seg.attempts = res.attempts
                    seg.seed = res.seed
                    seg.note = res.note
                self.state.audio_time += res.seconds
                if not res.ok:
                    self.state.flagged += 1
                    self.log(f"⚑ #{seg.idx} помечен для проверки: {res.note}")
            except Exception as e:
                seg.status, seg.note = "error", f"запись: {e}"
                self.log(f"✖ ошибка записи #{seg.idx}: {e}")
            if time.time() - last_save > 5:
                self.project.save_segments()
                last_save = time.time()
        self.project.save_segments()

    def run(self) -> None:
        st = self.state
        st.running = True
        st.started = time.time()
        writer = threading.Thread(target=self._writer, daemon=True, name="writer")
        writer.start()
        try:
            self.log("Загрузка модели…" if self.engine.model is None else "Модель уже загружена.")
            batch = getattr(self.engine, "batch_capable", False)
            if batch:
                self.engine.set_voice(self.project.settings.voice)  # латенты посчитаются при запуске
                self.engine.load(self.synth.inference_timesteps)
            else:
                self.engine.load()
                self.engine.set_voice(self.project.settings.voice)
            todo = self._todo()
            st.total = len(todo)
            if not todo:
                self.log("Все сегменты уже озвучены.")
            self.log(f"К синтезу: {len(todo)} сегм. ({sum(len(s.text) for s in todo)} симв.)")

            cps_hist = [s.cps for s in self.project.segments if s.status == "done" and s.cps > 0 and len(s.text) >= 40]

            def expected():
                return statistics.median(cps_hist[-200:]) if len(cps_hist) >= 8 else None

            def accept(s, res):
                if res.ok and len(s.text) >= 40:
                    cps_hist.append(res.cps)
                self._q.put((s, res))
                st.done += 1
                if st.done % (50 if batch else 10) == 0 or st.done == st.total:
                    self.log(
                        f"{st.done}/{st.total} · RTF {st.rtf:.3f} · "
                        f"аудио {fmt_duration(st.audio_time)} за {fmt_duration(time.time() - st.started)}"
                    )

            def fail(s, e):
                if "out of memory" in str(e).lower():
                    self._recover_oom()
                s.status, s.note = "error", str(e)[:300]
                st.errors += 1
                self.log(f"✖ #{s.idx}: {e}")

            if batch:
                t_start = time.time()
                self.log(f"Пакетный режим: до {getattr(self.engine.cfg, 'concurrency', '?')} фрагментов одновременно")

                def on_result(s, res, err):
                    if err is not None:
                        fail(s, err)
                        return
                    st.gen_time = time.time() - t_start  # в пакетном режиме RTF — по общему времени
                    st.current = f"#{s.idx}: {s.text[:120]}"
                    accept(s, res)

                self.engine.synthesize_many(
                    [(s, s.text, self._seed_for(s)) for s in todo],
                    self.synth,
                    on_result,
                    expected_cps=expected,
                    should_stop=self._stop.is_set,
                )
                if self._stop.is_set():
                    self.log("Остановлено пользователем. Прогресс сохранён — можно продолжить позже.")
            else:
                for s in todo:
                    if self._stop.is_set():
                        self.log("Остановлено пользователем. Прогресс сохранён — можно продолжить позже.")
                        break
                    st.current = f"#{s.idx}: {s.text[:120]}"
                    try:
                        res = self.engine.synthesize(s.text, self.synth, seed=self._seed_for(s), expected_cps=expected())
                    except Exception as e:
                        fail(s, e)
                        self.log(traceback.format_exc(limit=2))
                        continue
                    st.gen_time += res.gen_time
                    accept(s, res)
        except Exception as e:
            self.log(f"✖ Критическая ошибка: {e}")
            self.log(traceback.format_exc(limit=4))
            st.finished_msg = f"Ошибка: {e}"
        finally:
            self._q.put(None)
            writer.join()
            st.running = False
            st.current = ""
            if not st.finished_msg:
                st.finished_msg = (
                    f"Готово: {st.done} сегм., помечено {st.flagged}, ошибок {st.errors}. "
                    f"RTF {st.rtf:.2f}"
                )
            self.log(st.finished_msg)

    def _recover_oom(self):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Сборка
# --------------------------------------------------------------------------- #


def export_book(
    project: Project,
    opts: Optional[ExportOptions] = None,
    allow_missing: bool = False,
    log: LogFn = print,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[str]:
    opts = opts or project.settings.export
    project.refresh_status()
    segs = project.segments
    missing = [s.idx for s in segs if s.status not in ("done", "flagged")]
    if missing and not allow_missing:
        raise RuntimeError(
            f"Не озвучено {len(missing)} сегм. (первые: {missing[:10]}). Допишите синтез или разрешите сборку с пропусками."
        )
    if not segs:
        raise RuntimeError("Нет сегментов")

    sr = None
    for s in segs:
        if project.seg_path(s).exists():
            import soundfile as sf

            sr = sf.info(str(project.seg_path(s))).samplerate
            break
    if sr is None:
        raise RuntimeError("Нет ни одного озвученного сегмента")

    book = project.book or Book(title=project.name)
    title = book.title or project.name
    artist = book.author or ""
    cover = str(project.cover_path) if project.cover_path.exists() else None
    out_dir = project.output_dir
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    groups = project.file_groups(opts.split_level)
    levels = project.chapter_levels or [2] * len(project.chapter_titles)
    chapter_wavs, chapter_titles, durations, outputs = [], [], [], []
    all_markers: List[tuple] = []
    offset = 0.0
    parent = ""  # «Глава 2» — префикс для меток параграфов, чтобы «§ 1» разных глав различались
    hierarchical = len(set(levels)) > 1
    # Папки: mp3/01 Раздел …/02 Глава …/003 § ….mp3 (уровни выше split_level)
    folder = {1: None, 2: None}  # текущие папки раздела и главы
    counters = {1: 0, 2: 0, 3: 0}
    for gi, (name, cis) in enumerate(groups):
        items = []
        for ci in cis:
            lvl = levels[ci] if ci < len(levels) else 2
            name_ci = project.chapter_titles[ci]
            if hierarchical and lvl in (1, 2) and lvl < opts.split_level:
                counters[lvl] += 1
                folder[lvl] = f"{counters[lvl]:02d} {safe_filename(name_ci, 40)}"
                if lvl == 1:
                    folder[2], counters[2] = None, 0
            if lvl == 2:
                parent = re.split(r"[.:]", name_ci, maxsplit=1)[0].strip()[:20]
            # метка: начало каждой главы уровня <= marker_level (и самой группы)
            mark = None
            if lvl <= opts.marker_level or ci == cis[0]:
                mark = f"{parent} · {name_ci}" if lvl == 3 and parent else name_ci
            for s in project.chapter_segments(ci):
                if project.seg_path(s).exists():
                    items.append((str(project.seg_path(s)), s.pause_ms, mark))
                    mark = None
        if not items:
            continue
        wav = work / f"{len(chapter_wavs) + 1:03d}.wav"
        if progress:
            progress(gi / max(1, len(groups)), f"Файл {gi + 1}/{len(groups)}: {name}")
        dur, markers = assemble_chapter(items, wav, sr)
        chapter_wavs.append(str(wav))
        chapter_titles.append(name)
        durations.append(dur)
        all_markers += [(m, offset + t) for m, t in markers]
        offset += dur
        log(f"{len(chapter_wavs):03d}. {name} — {fmt_duration(dur)}")
        if "mp3" in opts.formats:
            mp3_dir = out_dir / "mp3"
            if opts.folders and hierarchical:
                for lvl in (1, 2):
                    if folder[lvl] and lvl < opts.split_level:
                        mp3_dir = mp3_dir / folder[lvl]
            mp3_dir.mkdir(parents=True, exist_ok=True)
            # сквозной номер в имени — чтобы плеер играл по порядку даже без папок
            mp3 = mp3_dir / f"{len(chapter_wavs):03d} {safe_filename(name, 60)}.mp3"
            encode_mp3(str(wav), str(mp3), opts, title=name, album=title, artist=artist, track=len(chapter_wavs), cover=cover)
            outputs.append(str(mp3))

    if "m4b" in opts.formats and chapter_wavs:
        if progress:
            progress(0.95, "Сборка M4B с главами…")
        m4b = out_dir / f"{safe_filename(title)}.m4b"
        build_m4b(chapter_wavs, all_markers, offset, str(m4b), opts, title=title, artist=artist, cover=cover, work_dir=str(work))
        outputs.append(str(m4b))
        log(f"M4B: {m4b.name} ({fmt_duration(offset)}, меток глав: {len(all_markers)})")

    if "wav" in opts.formats:
        wav_dir = out_dir / "wav"
        wav_dir.mkdir(exist_ok=True)
        for i, (p, name) in enumerate(zip(chapter_wavs, chapter_titles), 1):
            dst = wav_dir / f"{i:03d} - {safe_filename(name, 50)}.wav"
            Path(p).replace(dst)
            outputs.append(str(dst))
    for p in chapter_wavs:
        Path(p).unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass
    if progress:
        progress(1.0, "Готово")
    log(f"Итого: {fmt_duration(sum(durations))}, файлов: {len(outputs)}")
    return outputs
