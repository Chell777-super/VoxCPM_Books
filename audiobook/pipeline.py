"""Конвейер: разбор книги → синтез (GPU) → запись (CPU-поток) → сборка.

GPU никогда не ждёт диск: синтез идёт в одном потоке, а пост-обработка
(обрезка тишины, нормализация громкости, FLAC-кодирование) и сохранение
манифеста — в отдельном потоке-писателе через очередь.
"""

from __future__ import annotations

import os
import queue
import re
import statistics
import threading
import time
import traceback
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .audio import (
    ExportOptions,
    assemble_chapter,
    build_m4b_from_aac,
    encode_aac,
    encode_mp3,
    encode_timeout,
    estimate_m4b_mb,
    fmt_duration,
    media_duration,
    postprocess,
    write_segment,
)
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


def plan_m4b_parts(durations: List[float], sections: List[str], max_mb: int, aac_bitrate: str) -> List[List[int]]:
    """Делит файлы-группы на части M4B не больше max_mb (по оценке размера).

    Режем только между группами (§/главами — середину § не разрываем). Если в набираемой части
    во второй её половине начался новый раздел — режем по границе раздела: так части осмысленнее.
    Группа, которая одна больше лимита, становится отдельной частью.
    """
    idx = list(range(len(durations)))
    if not max_mb or max_mb <= 0 or estimate_m4b_mb(sum(durations), aac_bitrate) <= max_mb:
        return [idx] if idx else []
    parts: List[List[int]] = []
    cur: List[int] = []
    for i in idx:
        if cur and estimate_m4b_mb(sum(durations[j] for j in cur + [i]), aac_bitrate) > max_mb:
            # граница раздела во второй половине текущей части?
            cut = None
            half = sum(durations[j] for j in cur) / 2
            acc = 0.0
            for k, j in enumerate(cur):
                if k > 0 and sections[j] != sections[cur[k - 1]] and acc >= half:
                    cut = k
                acc += durations[j]
            if cut:
                parts.append(cur[:cut])
                cur = cur[cut:]
                if estimate_m4b_mb(sum(durations[j] for j in cur + [i]), aac_bitrate) > max_mb:
                    parts.append(cur)
                    cur = []
            else:
                parts.append(cur)
                cur = []
        cur.append(i)
    if cur:
        parts.append(cur)
    return parts


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
    want_mp3, want_m4b, want_wav = "mp3" in opts.formats, "m4b" in opts.formats, "wav" in opts.formats
    tasks_per_group = int(want_mp3) + int(want_m4b)
    # Кодирование — самое долгое (AAC ≈ 55×, MP3 320k q0 ≈ 13× реального времени на ядро),
    # поэтому каждая группа кодируется отдельным процессом ffmpeg, параллельно на всех ядрах.
    n_workers = max(1, min(8, (os.cpu_count() or 2) // 2))
    max_inflight = n_workers * 3  # ограничиваем очередь, чтобы промежуточные WAV не забили диск
    t_start = time.time()

    G: List[dict] = []  # сведения о файлах-группах
    outputs: List[str] = []
    parent = ""  # «Глава 2» — префикс для меток параграфов, чтобы «§ 1» разных глав различались
    hierarchical = len(set(levels)) > 1
    # Папки: mp3/01 Раздел …/02 Глава …/003 § ….mp3 (уровни выше split_level)
    folder = {1: None, 2: None}  # текущие папки раздела и главы
    counters = {1: 0, 2: 0, 3: 0}
    section = ""  # текущий раздел — для деления M4B на части по границам разделов

    total_units = max(1, len(groups) * (1 + tasks_per_group))
    done_units = [0]
    pending: dict = {}  # future -> (индекс группы, вид)
    left_tasks: dict = {}  # индекс группы -> сколько задач кодирования ещё осталось
    pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="encode") if tasks_per_group else None

    def report(msg: str) -> None:
        if not progress:
            return
        frac = done_units[0] / total_units
        el = time.time() - t_start
        eta = f" · ≈ {fmt_duration(el / frac * (1 - frac))} осталось" if frac > 0.03 else ""
        progress(min(0.97, frac * 0.97), msg + eta)

    def finish(fut) -> None:
        gi, kind = pending.pop(fut)
        fut.result()  # пробрасываем ошибку ffmpeg (в т.ч. таймаут)
        done_units[0] += 1
        left_tasks[gi] -= 1
        g = G[gi]
        if kind == "m4a":
            g["m4a_dur"] = media_duration(g["m4a"]) or g["dur"]
        if left_tasks[gi] == 0 and not want_wav:
            Path(g["wav"]).unlink(missing_ok=True)  # промежуточный WAV больше не нужен
        ready = len(G) * tasks_per_group - sum(left_tasks.values())
        report(f"Кодирую {ready}/{len(groups) * tasks_per_group} (×{n_workers} параллельно)")

    def drain(block: bool) -> None:
        if not pending:
            return
        done, _ = wait(list(pending), timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for fut in done:
            finish(fut)

    try:
        for gi, (name, cis) in enumerate(groups):
            items = []
            for ci in cis:
                lvl = levels[ci] if ci < len(levels) else 2
                name_ci = project.chapter_titles[ci]
                if lvl == 1:
                    section = name_ci
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
            done_units[0] += 1
            if not items:
                done_units[0] += tasks_per_group
                continue
            while len(pending) >= max_inflight:
                drain(block=True)
            drain(block=False)
            k = len(G) + 1
            wav = work / f"{k:03d}.wav"
            report(f"Готовлю {gi + 1}/{len(groups)}: {name}")
            dur, markers = assemble_chapter(items, wav, sr)
            g = dict(name=name, wav=str(wav), dur=dur, marks=markers, section=section, m4a=None, m4a_dur=dur)
            G.append(g)
            gidx = len(G) - 1
            left_tasks[gidx] = 0
            log(f"{k:03d}. {name} — {fmt_duration(dur)}")
            tmo = encode_timeout(dur)
            if want_mp3:
                mp3_dir = out_dir / "mp3"
                if opts.folders and hierarchical:
                    for lvl in (1, 2):
                        if folder[lvl] and lvl < opts.split_level:
                            mp3_dir = mp3_dir / folder[lvl]
                mp3_dir.mkdir(parents=True, exist_ok=True)
                # сквозной номер в имени — чтобы плеер играл по порядку даже без папок
                mp3 = mp3_dir / f"{k:03d} {safe_filename(name, 60)}.mp3"
                fut = pool.submit(
                    encode_mp3, str(wav), str(mp3), opts, title=name, album=title, artist=artist, track=k, cover=cover,
                    timeout=tmo,
                )
                pending[fut] = (gidx, "mp3")
                left_tasks[gidx] += 1
                outputs.append(str(mp3))
            if want_m4b:
                g["m4a"] = str(work / f"{k:03d}.m4a")
                fut = pool.submit(encode_aac, str(wav), g["m4a"], opts, timeout=tmo)
                pending[fut] = (gidx, "m4a")
                left_tasks[gidx] += 1
        while pending:
            drain(block=True)
    except BaseException:
        for fut in pending:
            fut.cancel()
        raise
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)

    durations = [g["m4a_dur"] if want_m4b else g["dur"] for g in G]
    if want_m4b and G:
        if progress:
            progress(0.97, "Склейка M4B с главами (без перекодирования)…")
        # метки — по фактическим длительностям закодированных файлов (кодер добавляет миллисекунды)
        starts = [sum(durations[:i]) for i in range(len(durations) + 1)]
        all_markers = [(m, starts[i] + t) for i, g in enumerate(G) for m, t in g["marks"]]
        parts = plan_m4b_parts(durations, [g["section"] for g in G], opts.max_part_mb, opts.aac_bitrate)
        n = len(parts)
        for pi, idx in enumerate(parts, 1):
            p0, p1 = starts[idx[0]], starts[idx[-1] + 1]
            # метки этой части, со сдвигом на её начало
            marks = [(m, t - p0) for m, t in all_markers if p0 - 1e-6 <= t < p1 - 1e-6]
            if not marks or marks[0][1] > 0.05:  # часть начинается с середины главы — метка в начале
                marks.insert(0, (G[idx[0]]["name"], 0.0))
            if n == 1:
                m4b = out_dir / f"{safe_filename(title)}.m4b"
                part_title, track = title, None
            else:
                m4b = out_dir / f"{safe_filename(title, 50)} ({pi} из {n}).m4b"
                part_title, track = f"{title}, часть {pi} из {n}", f"{pi}/{n}"
            build_m4b_from_aac(
                [G[i]["m4a"] for i in idx], marks, p1 - p0, str(m4b),
                title=part_title, album=title, artist=artist, cover=cover, work_dir=str(work), track=track,
            )
            outputs.append(str(m4b))
            size = m4b.stat().st_size / 1e6
            log(f"M4B: {m4b.name} ({fmt_duration(p1 - p0)}, {size:.0f} МБ, меток: {len(marks)})")
        for g in G:
            Path(g["m4a"]).unlink(missing_ok=True)

    if want_wav:
        wav_dir = out_dir / "wav"
        wav_dir.mkdir(exist_ok=True)
        for i, g in enumerate(G, 1):
            dst = wav_dir / f"{i:03d} - {safe_filename(g['name'], 50)}.wav"
            Path(g["wav"]).replace(dst)
            outputs.append(str(dst))
    for g in G:
        Path(g["wav"]).unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass
    if progress:
        progress(1.0, "Готово")
    log(f"Сборка заняла {fmt_duration(time.time() - t_start)} (кодирование ×{n_workers} параллельно)")
    log(f"Итого: {fmt_duration(sum(durations))}, файлов: {len(outputs)}")
    return outputs
