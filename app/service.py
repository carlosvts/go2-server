"""O caminho de uma frase: áudio → transcrição → interpretação → fila."""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from app.audio import SAMPLE_RATE
from app.catalog import Catalog
from app.config import Settings
from app.dispatch import Dispatcher
from app.interpret.interpreter import Interpretation, Interpreter, Kind
from app.metrics import KNOWN_REASONS, Stats
from app.transcribe.base import TranscriptionPool
from app.wake_word import strip_wake_word

log = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class CommandRef(BaseModel):
    id: str


class UtteranceResponse(BaseModel):
    utterance_id: str
    status: str  # "enqueued", "rejected", "stopped" ou "ignored"
    transcript: str
    commands: list[CommandRef] = []
    confidence: float | None = None
    reject_reason: str | None = None


@dataclass(frozen=True)
class Meta:
    utterance_id: str
    edge_id: str
    reason: str
    local_hypothesis: Any = None
    local_confidence: Any = None


class Service:
    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        interpreter: Interpreter,
        pool: TranscriptionPool,
        dispatcher: Dispatcher,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.interpreter = interpreter
        self.pool = pool
        self.dispatcher = dispatcher
        self.stats = Stats(settings.stats_window)
        # utterance_id → (quando expira, resposta). A resposta é um Future
        # para que a repetição que chega no meio do processamento espere a
        # mesma resposta em vez de processar de novo.
        self._seen: dict[str, tuple[float, asyncio.Future[UtteranceResponse]]] = {}

    async def handle(self, meta: Meta, audio: np.ndarray, wav: bytes) -> UtteranceResponse:
        """Idempotente por `utterance_id`: a repetição não executa nada."""
        received = time.monotonic()
        self._forget_expired(received)
        known = self._seen.get(meta.utterance_id)
        if known is not None:
            log.info(
                "utterance repetida",
                extra={"event": "duplicate", "utterance_id": meta.utterance_id},
            )
            return await asyncio.shield(known[1])

        future: asyncio.Future[UtteranceResponse] = asyncio.get_running_loop().create_future()
        self._seen[meta.utterance_id] = (received + self.settings.idempotency_ttl_s, future)
        try:
            response = await self._process(meta, audio, wav, received)
        except Exception as error:
            # Não guarda falha: uma nova tentativa tem de poder processar.
            del self._seen[meta.utterance_id]
            future.set_exception(error)
            future.exception()  # marca como lida, para o caso de ninguém esperar
            raise
        except BaseException:
            del self._seen[meta.utterance_id]
            future.cancel()
            raise
        future.set_result(response)
        return response

    def _forget_expired(self, now: float) -> None:
        for utterance_id in [key for key, (expires, _) in self._seen.items() if expires < now]:
            del self._seen[utterance_id]

    async def _process(
        self, meta: Meta, audio: np.ndarray, wav: bytes, received: float
    ) -> UtteranceResponse:
        """`received` é o relógio do servidor: a idade da fala nunca vem do cliente."""
        if meta.reason not in KNOWN_REASONS:
            log.warning("reason desconhecido aceito: %r", meta.reason)

        transcription, lane = await self.pool.transcribe(audio)
        transcribed = time.monotonic()
        raw = transcription.text
        text, wake_word_removed = strip_wake_word(raw, self.settings.wake_words)

        status = "rejected"
        commands: list[str] = []
        confidence: float | None = None
        reject_reason: str | None = None
        interpretation: Interpretation | None = None
        interpreted = transcribed

        if self.interpreter.is_stop_only(text):
            # Caminho curto da parada: sem embedding e sem troca de thread.
            # Vale até para a transcrição que o motor marcou como duvidosa:
            # parar por engano é seguro, deixar de parar não é.
            status, commands, confidence = "stop", [self.catalog.stop_id], 1.0
        elif transcription.reject_reason is not None:
            reject_reason = transcription.reject_reason
        else:
            interpretation = await asyncio.to_thread(self.interpreter.interpret, text)
            interpreted = time.monotonic()
            confidence = interpretation.confidence
            if interpretation.kind is Kind.EMPTY:
                # Só a wake word, ou nada que seja palavra.
                status = "ignored"
            elif interpretation.kind is Kind.STOP:
                status, commands = "stop", interpretation.commands
            elif interpretation.kind is Kind.REJECTED:
                reject_reason = interpretation.reject_reason
            else:
                commands = interpretation.commands

        if status == "stop":
            cancellation = await self.dispatcher.cancel("stop", meta.edge_id)
            if cancellation.stop_sent:
                status = "stopped"
            else:
                status, reject_reason, commands = "rejected", "stop_not_delivered", []
        elif commands:
            status, reject_reason = self._enqueue(commands, meta, received)
            if status != "enqueued":
                commands = []
        done = time.monotonic()

        response = UtteranceResponse(
            utterance_id=meta.utterance_id,
            status=status,
            transcript=raw,
            commands=[CommandRef(id=command_id) for command_id in commands],
            confidence=None if confidence is None else round(confidence, 4),
            reject_reason=reject_reason,
        )
        total_ms = (done - received) * 1000
        self.stats.record(meta.reason, status, total_ms)
        record = {
            "event": "utterance",
            "utterance_id": meta.utterance_id,
            "edge_id": meta.edge_id,
            "reason": meta.reason,
            "local_hypothesis": meta.local_hypothesis,
            "local_confidence": meta.local_confidence,
            "audio_s": round(len(audio) / SAMPLE_RATE, 2),
            "transcript_raw": raw,
            "transcript": text,
            "wake_word_removed": wake_word_removed,
            "normalized": interpretation.normalized if interpretation else None,
            "status": status,
            "commands": commands,
            "confidence": response.confidence,
            "reject_reason": reject_reason,
            "clauses": [clause.log() for clause in interpretation.clauses]
            if interpretation
            else [],
            "stt": transcription.details,
            "stt_lane": lane,
            "stt_ms": round((transcribed - received) * 1000, 1),
            "interpret_ms": round((interpreted - transcribed) * 1000, 1),
            "dispatch_ms": round((done - interpreted) * 1000, 1),
            "total_ms": round(total_ms, 1),
        }
        log.info("utterance %s", status, extra=record)
        if self.settings.save_utterances:
            await asyncio.to_thread(self._save, wav, record)
        return response

    def _enqueue(self, commands: list[str], meta: Meta, received: float) -> tuple[str, str | None]:
        # A saída só pode conter IDs do catálogo, e a parada nunca entra na fila.
        if any(c not in self.catalog.ids or c == self.catalog.stop_id for c in commands):
            return "rejected", "unknown_command"
        if time.monotonic() - received > self.settings.max_utterance_age_s:
            return "rejected", "stale"
        submission = self.dispatcher.submit(commands, meta.edge_id, meta.utterance_id)
        if not submission.accepted:
            return "rejected", submission.reason
        return "enqueued", None

    def _save(self, wav: bytes, record: dict) -> None:
        """Guarda o áudio e o resultado, para montar o conjunto de avaliação."""
        try:
            name = record["utterance_id"]
            if not _SAFE_NAME.match(name):
                # O ID vem do cliente: não vira caminho de arquivo sem conferir.
                name = hashlib.sha256(name.encode()).hexdigest()[:32]
            folder = Path(self.settings.save_dir) / time.strftime("%Y-%m-%d")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{name}.wav").write_bytes(wav)
            (folder / f"{name}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
        except OSError as error:
            log.warning("não foi possível salvar a frase: %s", error)
