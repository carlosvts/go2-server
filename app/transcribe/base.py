"""O que qualquer motor de transcrição entrega."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.audio import SAMPLE_RATE


@dataclass(frozen=True)
class Transcription:
    text: str
    # Preenchido quando a transcrição não deve ser interpretada (silêncio,
    # ruído, alucinação conhecida).
    reject_reason: str | None = None
    details: dict = field(default_factory=dict)  # scores do motor, para o log


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> Transcription:
        """Áudio float32 mono 16 kHz → texto livre em pt-BR. Bloqueante."""


class TranscriptionPool:
    """Roda a transcrição fora do event loop, com uma faixa para áudio curto.

    A parada falada do modo thin chega como um áudio curto. Com uma thread
    reservada, ela não espera atrás de uma frase longa em transcrição.
    """

    def __init__(self, transcriber: Transcriber, workers: int, short_audio_s: float) -> None:
        self._transcriber = transcriber
        self._short_audio_s = short_audio_s
        self._shared = ThreadPoolExecutor(workers, thread_name_prefix="stt")
        self._reserved = ThreadPoolExecutor(1, thread_name_prefix="stt-short")
        self._reserved_busy = False

    async def transcribe(self, audio: np.ndarray) -> tuple[Transcription, str]:
        """Devolve a transcrição e a faixa usada ("short" ou "shared")."""
        loop = asyncio.get_running_loop()
        short = len(audio) / SAMPLE_RATE <= self._short_audio_s
        if not (short and not self._reserved_busy):
            call = self._transcriber.transcribe
            return await loop.run_in_executor(self._shared, call, audio), "shared"
        self._reserved_busy = True
        try:
            result = await loop.run_in_executor(self._reserved, self._transcriber.transcribe, audio)
        finally:
            self._reserved_busy = False
        return result, "short"

    def close(self) -> None:
        self._shared.shutdown(wait=False, cancel_futures=True)
        self._reserved.shutdown(wait=False, cancel_futures=True)
