"""Движок синтеза на VoxCPM2 с оптимизациями под одну CUDA-видеокарту.

Что сделано для RTX 3090 (24 ГБ, Ampere) + 16 ГБ ОЗУ:

* bfloat16 (нативно на Ampere) + TF32 для оставшихся fp32-операций;
* ``torch.compile(mode="reduce-overhead")`` (CUDA Graphs) — на Windows через
  пакет ``triton-windows``; если компиляция не удалась — автоматический откат
  на eager-режим без падения;
* модель строится *сразу на GPU* и веса грузятся из safetensors прямо в VRAM —
  пиковое потребление ОЗУ ~2–3 ГБ вместо ~12 ГБ при стандартной загрузке;
* кэш промпта (опорного голоса) строится один раз на всю книгу;
* контроль качества: проверка скорости речи (симв./с) и повтор с другим seed,
  если сегмент «заговорился» или оборвался.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class VoiceConfig:
    """Какой голос использовать.

    mode:
        ``clone``   — клон по образцу (reference_wav), опционально + транскрипт
                      (prompt_text) → «ультимативное» клонирование (продолжение);
        ``design``  — голос по текстовому описанию. ВАЖНО: без образца голос
                      будет «плавать» между сегментами, поэтому в UI сначала
                      генерируется образец, который затем используется как clone;
        ``default`` — голос модели по умолчанию (тоже нестабилен между сегментами).
    """

    mode: str = "clone"
    reference_wav: Optional[str] = None
    prompt_text: Optional[str] = None  # точный текст reference_wav
    style: str = ""  # инструкция стиля, напр. «спокойно, размеренно, тёплый тембр»
    denoise: bool = False


@dataclass
class SynthConfig:
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    seed: int = 42
    max_retries: int = 3
    min_cps: float = 5.0  # симв./с — медленнее = «заговорился»/тянет
    max_cps: float = 32.0  # симв./с — быстрее = обрыв / проглоченные слова


@dataclass
class EngineConfig:
    model_path: str = "models/VoxCPM2"
    device: str = "cuda"
    compile: bool = True
    low_ram_load: bool = True
    max_length: Optional[int] = None  # размер KV-кэша; None = из конфига модели


@dataclass
class SynthResult:
    wav: np.ndarray
    sr: int
    seconds: float
    gen_time: float
    attempts: int
    cps: float
    ok: bool
    seed: int
    note: str = ""


def _setup_torch():
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    # i5-11400F: 6 ядер / 12 потоков. Больше потоков CPU-операциям не помогает,
    # а оставшиеся ядра нужны потоку записи и ffmpeg.
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 6)))
    try:
        import torch._dynamo as dynamo

        dynamo.config.cache_size_limit = max(getattr(dynamo.config, "cache_size_limit", 8), 64)
        dynamo.config.suppress_errors = True
    except Exception:
        pass


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def next_and_close(gen):
    try:
        return next(gen)
    finally:
        gen.close()


def _resolve_model_dir(path: str, repo_id: str = "openbmb/VoxCPM2") -> str:
    """Локальная папка модели; если её нет — скачиваем VoxCPM2 туда же."""
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    if not os.path.exists(path) and path.count("/") == 1 and not path.startswith(("models", ".", "/")):
        repo_id, path = path, os.path.join("models", path.split("/")[-1])  # передан HF id
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    from huggingface_hub import snapshot_download

    print(f"Скачиваю {repo_id} в {path} (~5 ГБ, один раз)…", file=sys.stderr)
    snapshot_download(repo_id=repo_id, local_dir=path)
    return path


def _load_low_ram(path: str, device: str, optimize: bool, max_length: Optional[int]):
    """Копия ``VoxCPM2Model.from_local`` с построением модели прямо на GPU.

    Стандартный путь создаёт 2B-модель в fp32 в ОЗУ (~9 ГБ), затем ещё грузит
    веса (~4–5 ГБ) — на 16 ГБ ОЗУ под Windows это на грани. Здесь параметры
    сразу выделяются в VRAM, а safetensors читаются прямо на устройство.
    """
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaTokenizerFast

    from voxcpm.model.utils import get_dtype
    from voxcpm.model.voxcpm2 import AudioVAEV2, VoxCPM2Model, VoxCPMConfig

    with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
        config = VoxCPMConfig.model_validate_json(f.read())
    if max_length:
        config.max_length = int(max_length)
    tokenizer = LlamaTokenizerFast.from_pretrained(path)

    vae_st = os.path.join(path, "audiovae.safetensors")
    model_st = os.path.join(path, "model.safetensors")
    if not (os.path.exists(vae_st) and os.path.exists(model_st)):
        raise FileNotFoundError("нет safetensors-весов — нужен стандартный загрузчик")

    with torch.device(device):
        avc = getattr(config, "audio_vae_config", None)
        audio_vae = AudioVAEV2(config=avc) if avc else AudioVAEV2()
        model = VoxCPM2Model(config, tokenizer, audio_vae, None, device=device)
    model = model.to(get_dtype(model.config.dtype))
    model.audio_vae = model.audio_vae.to(torch.float32)

    state = load_file(model_st, device=device)
    vae_state = load_file(vae_st, device=device)
    for k, v in vae_state.items():
        state[f"audio_vae.{k}"] = v
    missing, unexpected = model.load_state_dict(state, strict=False)
    del state, vae_state
    real_missing = [k for k in missing if "lora_" not in k]
    if real_missing:
        raise RuntimeError(f"не найдены веса: {real_missing[:5]}…")
    gc.collect()
    return model.to(device).eval().optimize(disable=not optimize)


class TTSEngine:
    def __init__(self, cfg: EngineConfig, log=print):
        self.cfg = cfg
        self.log = log
        self.model = None
        self.prompt_cache = None
        self.voice: Optional[VoiceConfig] = None
        self.compiled = False
        self._denoiser = None

    # ------------------------------------------------------------------ load
    @property
    def sample_rate(self) -> int:
        return int(self.model.sample_rate) if self.model is not None else 48000

    def load(self):
        if self.model is not None:
            return self
        import torch

        _setup_torch()
        device = self.cfg.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.log("⚠ CUDA недоступна — работаю на CPU (будет ОЧЕНЬ медленно).")
            device = "cpu"
        path = _resolve_model_dir(self.cfg.model_path)
        want_compile = self.cfg.compile and device.startswith("cuda") and _has_triton()
        if self.cfg.compile and device.startswith("cuda") and not want_compile:
            self.log("⚠ triton не установлен — torch.compile отключён (на Windows: pip install triton-windows).")
        t0 = time.time()

        for attempt_compile in ([True, False] if want_compile else [False]):
            try:
                self.model = self._load_model(path, device, attempt_compile)
                if attempt_compile:
                    self.log("Прогрев и компиляция (torch.compile + CUDA Graphs), 1–3 мин при первом запуске…")
                    self._warmup()
                self.compiled = attempt_compile
                break
            except Exception as e:  # откат на eager
                self.log(f"⚠ {'Компиляция' if attempt_compile else 'Загрузка'} не удалась: {e}")
                self.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not attempt_compile:
                    raise
                try:
                    import torch._dynamo

                    torch._dynamo.reset()
                except Exception:
                    pass
                self.log("Перезагружаю модель без torch.compile…")
        if self.model is None:
            raise RuntimeError("Не удалось загрузить модель")

        if not self.compiled:
            self._warmup()
        mem = ""
        if torch.cuda.is_available():
            mem = f", VRAM занято {torch.cuda.memory_allocated() / 2**30:.1f} ГБ"
        self.log(
            f"Модель загружена за {time.time() - t0:.0f} с "
            f"(compile={'да' if self.compiled else 'нет'}, dtype={self.model.config.dtype}{mem})"
        )
        return self

    def _load_model(self, path: str, device: str, optimize: bool):
        if self.cfg.low_ram_load and device.startswith("cuda"):
            try:
                return _load_low_ram(path, device, optimize, self.cfg.max_length)
            except Exception as e:
                self.log(f"Экономная загрузка недоступна ({e}), использую стандартную.")
        from voxcpm.model.voxcpm2 import VoxCPM2Model

        return VoxCPM2Model.from_local(path, optimize=optimize, device=device)

    def _warmup(self):
        # Два коротких прогона: первый компилирует, второй захватывает CUDA-графы
        for txt in ("Проверка связи.", "Это тестовое предложение для прогрева."):
            next_and_close(
                self.model._generate_with_prompt_cache(
                    target_text=txt, prompt_cache=None, max_len=12, inference_timesteps=4, streaming=False
                )
            )

    def unload(self):
        import torch

        self.model = None
        self.prompt_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------- voice
    def set_voice(self, voice: VoiceConfig):
        self.load()
        self.voice = voice
        if voice.mode == "clone" and voice.reference_wav:
            ref = voice.reference_wav
            if voice.denoise:
                ref = self._denoise(ref)
            prompt_text = (voice.prompt_text or "").strip() or None
            self.prompt_cache = self.model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=ref if prompt_text else None,
                reference_wav_path=ref,
                trim_silence_vad=True,
            )
            mode = "образец + транскрипт (максимальная похожесть)" if prompt_text else "образец"
            self.log(f"Голос: клонирование — {mode}")
        else:
            self.prompt_cache = None
            self.log("Голос: " + ("по описанию" if voice.mode == "design" else "по умолчанию"))

    def _denoise(self, path: str) -> str:
        try:
            if self._denoiser is None:
                from voxcpm.zipenhancer import ZipEnhancer

                self._denoiser = ZipEnhancer("iic/speech_zipenhancer_ans_multiloss_16k_base")
            out = os.path.splitext(path)[0] + "_denoised.wav"
            self._denoiser.enhance(path, output_path=out)
            return out
        except Exception as e:
            self.log(f"⚠ Шумоподавление недоступно: {e}")
            return path

    def _full_text(self, text: str) -> str:
        v = self.voice
        style = (v.style or "").strip() if v else ""
        # В режиме продолжения (с транскриптом) инструкция стиля игнорируется моделью
        in_continuation = bool(self.prompt_cache and self.prompt_cache.get("mode") in ("continuation", "ref_continuation"))
        if style and not in_continuation:
            return f"({style}){text}"
        return text

    # ----------------------------------------------------------------- synth
    def synthesize(self, text: str, sc: SynthConfig, seed: Optional[int] = None, expected_cps: Optional[float] = None) -> SynthResult:
        """Синтез одного сегмента с контролем качества и повторами."""
        import torch

        self.load()
        seed = sc.seed if seed is None else seed
        full = self._full_text(text)
        n_chars = max(1, len(text))
        best: Optional[SynthResult] = None
        t_all = time.time()

        for attempt in range(1, sc.max_retries + 1):
            cur_seed = seed + (attempt - 1) * 1009
            t0 = time.time()
            wav, _, _ = next_and_close(
                self.model._generate_with_prompt_cache(
                    target_text=full,
                    prompt_cache=self.prompt_cache,
                    min_len=2,
                    max_len=4096,
                    inference_timesteps=sc.inference_timesteps,
                    cfg_value=sc.cfg_value,
                    retry_badcase=True,
                    retry_badcase_max_times=2,
                    retry_badcase_ratio_threshold=6.0,
                    streaming=False,
                    seed=cur_seed,
                )
            )
            wav = wav.squeeze(0).float().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
            gen_time = time.time() - t0
            sec = speech_seconds(wav, self.sample_rate)
            cps = n_chars / max(sec, 0.05)
            ok = sc.min_cps <= cps <= sc.max_cps
            note = ""
            if not ok:
                note = "слишком медленно/лишнее" if cps < sc.min_cps else "слишком быстро/обрыв"
            # Если знаем типичную скорость диктора — проверяем отклонение строже
            if ok and expected_cps and n_chars >= 40:
                ratio = cps / expected_cps
                if ratio < 0.55 or ratio > 1.8:
                    ok, note = False, f"темп ×{ratio:.2f} от среднего"
            res = SynthResult(wav, self.sample_rate, len(wav) / self.sample_rate, gen_time, attempt, cps, ok, cur_seed, note)
            if ok:
                best = res
                break
            # запоминаем самый правдоподобный из неудачных
            target = expected_cps or 14.0
            if best is None or abs(np.log(res.cps / target)) < abs(np.log(best.cps / target)):
                best = res
            self.log(f"  повтор ({attempt}/{sc.max_retries}): {note}, {cps:.1f} симв./с")
        best.attempts = attempt
        best.gen_time = time.time() - t_all
        return best

    def vram_info(self) -> str:
        try:
            import torch

            if not torch.cuda.is_available():
                return "CUDA недоступна"
            free, total = torch.cuda.mem_get_info()
            name = torch.cuda.get_device_name(0)
            return f"{name}: свободно {free / 2**30:.1f} из {total / 2**30:.1f} ГБ"
        except Exception as e:
            return str(e)


def speech_seconds(wav: np.ndarray, sr: int, rel_db: float = -40.0) -> float:
    """Длительность без тишины по краям — для оценки темпа речи."""
    from .audio import trim_silence

    return len(trim_silence(np.asarray(wav, dtype=np.float32), sr, rel_db=rel_db, keep_ms=0)) / sr


@dataclass
class Presets:
    """Готовые профили качество/скорость (inference_timesteps)."""

    table: dict = field(
        default_factory=lambda: {
            "Быстро (черновик)": 6,
            "Баланс (рекомендуется)": 10,
            "Качество": 16,
        }
    )
