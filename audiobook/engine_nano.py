"""Движок на Nano-vLLM-VoxCPM: параллельный синтез многих фрагментов на одной видеокарте.

Почему это быстро для аудиокниги:

* фрагменты книги независимы → их можно генерировать пачкой (continuous batching):
  по замерам авторов на RTX 4090 при 16 одновременных запросах каждый идёт с RTF ~0.18,
  т.е. суммарно в ~90 раз быстрее реального времени (у обычного PyTorch — ~3×);
* кэш префиксов: образец голоса (reference/prompt) одинаков у всех фрагментов и
  считается один раз;
* CUDA Graphs + FlashAttention + Triton.

Ограничения: Linux (на Windows — только с вручную установленным flash-attn), torch с
готовым flash-attn (см. install_ubuntu.sh). Число шагов диффузии задаётся при запуске
сервера — при его смене сервер перезапускается.

Интерфейс совместим с :class:`audiobook.engine.TTSEngine` (load / set_voice / synthesize /
sample_rate / vram_info / model) + метод :meth:`synthesize_many` для пакетного режима.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import time
from typing import Callable, Iterable, Optional

import numpy as np

from .engine import (
    EngineConfig,
    SynthConfig,
    SynthResult,
    VoiceConfig,
    _resolve_model_dir,
    evaluate_take,
    max_patches_for,
    pick_better,
)


def nano_available() -> tuple[bool, str]:
    if sys.platform == "win32" and not os.environ.get("VOXBOOKS_FORCE_NANO"):
        return False, "Nano-vLLM рассчитан на Linux (на Windows — только с вручную собранным flash-attn)"
    try:
        import flash_attn  # noqa: F401
        import nanovllm_voxcpm  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"nanovllm_voxcpm/flash-attn не установлены ({e})"
    return True, ""


class NanoEngine:
    batch_capable = True

    def __init__(self, cfg: EngineConfig, log=print):
        self.cfg = cfg
        self.log = log
        self.pool = None
        self.voice: Optional[VoiceConfig] = None
        self.compiled = True
        self._sr = 48000
        self._timesteps: Optional[int] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ref_latents: Optional[bytes] = None
        self._prompt_latents: Optional[bytes] = None
        self._prompt_text: str = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------- event loop
    def _ensure_loop(self):
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="nanovllm-loop")
        self._thread.start()

    def _run(self, coro, timeout: Optional[float] = None):
        self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # ------------------------------------------------------------------ load
    @property
    def model(self):  # совместимость с TTSEngine (webui проверяет engine.model)
        return self.pool

    @property
    def sample_rate(self) -> int:
        return self._sr

    def load(self, timesteps: Optional[int] = None):
        timesteps = int(timesteps or self._timesteps or 10)
        with self._lock:
            if self.pool is not None and timesteps == self._timesteps:
                return self
            if self.pool is not None:
                self.log(f"Число шагов изменилось ({self._timesteps} → {timesteps}) — перезапускаю Nano-vLLM…")
                self.unload()
            path = _resolve_model_dir(self.cfg.model_path)
            t0 = time.time()
            self.log(
                f"Запуск Nano-vLLM (одновременно до {self.cfg.concurrency} фрагментов, шагов {timesteps}, "
                f"память GPU {int(self.cfg.gpu_memory_utilization * 100)} %)… первый раз 1–3 мин"
            )

            async def start():
                from nanovllm_voxcpm import VoxCPM

                pool = VoxCPM.from_pretrained(
                    model=path,
                    inference_timesteps=timesteps,
                    max_num_batched_tokens=16384,
                    max_num_seqs=max(1, int(self.cfg.concurrency)),
                    max_model_len=4096,
                    gpu_memory_utilization=float(self.cfg.gpu_memory_utilization),
                    enforce_eager=not self.cfg.compile,
                    devices=[0],
                )
                await pool.wait_for_ready()
                info = await pool.get_model_info()
                return pool, int(info.get("output_sample_rate") or info.get("sample_rate") or 48000)

            self.pool, self._sr = self._run(start())
            self._timesteps = timesteps
            if not getattr(self, "_atexit", False):
                import atexit

                atexit.register(self.unload)  # корректно гасим дочерние процессы сервера
                self._atexit = True
            self.log(f"Nano-vLLM готов за {time.time() - t0:.0f} с, {self._sr} Гц")
            if self.voice is not None:
                self._encode_voice(self.voice)
        return self

    def unload(self):
        if self.pool is not None:
            try:
                self._run(self.pool.stop(), timeout=60)
            except Exception as e:  # noqa: BLE001
                self.log(f"⚠ остановка Nano-vLLM: {e}")
        self.pool = None
        self._ref_latents = self._prompt_latents = None

    # ----------------------------------------------------------------- voice
    def set_voice(self, voice: VoiceConfig):
        """Запоминает голос; латенты образца считаются при (пере)запуске сервера или сразу, если он уже идёт."""
        self.voice = voice
        if self.pool is not None:
            self._encode_voice(voice)

    def _encode_voice(self, voice: VoiceConfig):
        self._ref_latents = self._prompt_latents = None
        self._prompt_text = ""
        if voice.mode == "clone" and voice.reference_wav:
            wav_bytes = _trimmed_wav_bytes(voice.reference_wav)
            lat = self._run(self.pool.encode_latents(wav_bytes, "wav"))
            self._ref_latents = lat
            text = (voice.prompt_text or "").strip()
            if text:
                self._prompt_latents, self._prompt_text = lat, text
            self.log("Голос: клонирование — " + ("Hi-Fi (образец + транскрипт)" if text else "образец"))
        else:
            self.log("Голос: " + ("по описанию" if voice.mode == "design" else "по умолчанию"))

    def _full_text(self, text: str) -> str:
        style = (self.voice.style or "").strip() if self.voice else ""
        if style and not self._prompt_latents:
            return f"({style}){text}"
        return text

    # ----------------------------------------------------------------- synth
    async def _generate(self, text: str, sc: SynthConfig, seed: int) -> np.ndarray:
        chunks = []
        async for ch in self.pool.generate(
            target_text=self._full_text(text),
            prompt_latents=self._prompt_latents,
            prompt_text=self._prompt_text if self._prompt_latents else "",
            ref_audio_latents=self._ref_latents,
            max_generate_length=max_patches_for(text),
            cfg_value=float(sc.cfg_value),
            seed=int(seed) % (2**31 - 1),
        ):
            chunks.append(np.asarray(ch, dtype=np.float32).reshape(-1))
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    async def _synth_qc(self, text: str, sc: SynthConfig, seed: int, expected_cps: Optional[float]) -> SynthResult:
        best: Optional[SynthResult] = None
        t_all = time.time()
        attempt = 0
        for attempt in range(1, sc.max_retries + 1):
            cur_seed = seed + (attempt - 1) * 1009
            t0 = time.time()
            wav = await self._generate(text, sc, cur_seed)
            ok, cps, note = evaluate_take(text, wav, self._sr, sc, expected_cps)
            res = SynthResult(wav, self._sr, len(wav) / self._sr, time.time() - t0, attempt, cps, ok, cur_seed, note)
            if ok:
                best = res
                break
            best = pick_better(best, res, expected_cps)
            self.log(f"  повтор ({attempt}/{sc.max_retries}): {note}, {cps:.1f} симв./с")
        best.attempts = attempt
        best.gen_time = time.time() - t_all
        return best

    def synthesize(self, text: str, sc: SynthConfig, seed: Optional[int] = None, expected_cps: Optional[float] = None) -> SynthResult:
        self.load(sc.inference_timesteps)
        return self._run(self._synth_qc(text, sc, sc.seed if seed is None else seed, expected_cps))

    def synthesize_many(
        self,
        items: Iterable[tuple],
        sc: SynthConfig,
        on_result: Callable[[object, Optional[SynthResult], Optional[BaseException]], None],
        expected_cps: Callable[[], Optional[float]] = lambda: None,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        """Параллельный синтез: items = [(ключ, текст, seed)]. on_result вызывается по мере готовности
        (из потока event loop). Возвращается, когда всё готово или запрошена остановка."""
        self.load(sc.inference_timesteps)
        queue_items = list(items)

        async def run():
            it = iter(queue_items)
            lock = asyncio.Lock()

            async def worker():
                while not should_stop():
                    async with lock:
                        nxt = next(it, None)
                    if nxt is None:
                        return
                    key, text, seed = nxt
                    try:
                        res = await self._synth_qc(text, sc, seed, expected_cps())
                        on_result(key, res, None)
                    except Exception as e:  # noqa: BLE001
                        on_result(key, None, e)

            n = max(1, int(self.cfg.concurrency))
            await asyncio.gather(*[worker() for _ in range(n)])

        self._run(run())

    def vram_info(self) -> str:
        try:
            import torch

            if not torch.cuda.is_available():
                return "CUDA недоступна"
            free, total = torch.cuda.mem_get_info()
            return (
                f"{torch.cuda.get_device_name(0)}: свободно {free / 2**30:.1f} из {total / 2**30:.1f} ГБ · "
                f"Nano-vLLM ×{self.cfg.concurrency}"
            )
        except Exception as e:  # noqa: BLE001
            return str(e)


def _trimmed_wav_bytes(path: str) -> bytes:
    """Образец без тишины по краям, моно 16 кГц, WAV в памяти (VAE кодирует 16 кГц)."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(path, sr=16000, mono=True)
    if y.size:
        yt, _ = librosa.effects.trim(y, top_db=35)
        if yt.size > sr:  # не обрезаем до пустоты
            y = yt
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def make_engine(cfg: EngineConfig, log=print):
    """Выбор движка: nano (если доступен или явно запрошен) или обычный PyTorch."""
    from .engine import TTSEngine

    backend = (cfg.backend or "auto").lower()
    if backend in ("auto", "nano"):
        ok, why = nano_available()
        if ok:
            log("Движок: Nano-vLLM (параллельный синтез)")
            return NanoEngine(cfg, log)
        if backend == "nano":
            log(f"⚠ Nano-vLLM недоступен: {why}. Использую обычный PyTorch.")
    log("Движок: PyTorch (VoxCPM)")
    return TTSEngine(cfg, log)
