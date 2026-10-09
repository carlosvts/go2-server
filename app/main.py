"""App factory e rotas HTTP.

uv run uvicorn app.main:create_app --factory --host 0.0.0.0 --port 9000
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from app.audio import InvalidAudioError, decode_wav
from app.catalog import load_catalog
from app.config import Settings
from app.dispatch import Dispatcher, Go2, Limits
from app.go2_client import Go2Client
from app.interpret.embed import Embedder
from app.interpret.interpreter import Interpreter, Thresholds
from app.logging_setup import setup_logging
from app.service import Meta, Service, UtteranceResponse
from app.transcribe.base import Transcriber, TranscriptionPool

log = logging.getLogger(__name__)

STATS_LOG_INTERVAL_S = 60.0
_UNSET = object()


class CancelRequest(BaseModel):
    reason: Literal["stop", "local_command"]
    # O edge atual não manda `edge_id` no cancel: é opcional.
    edge_id: str | None = None


class CancelResponse(BaseModel):
    cleared: int
    stop_sent: bool | None = None  # só com reason="stop"


def _load_transcriber(settings: Settings) -> Transcriber:
    if settings.stt_engine == "vosk":
        from app.transcribe.vosk import VoskTranscriber

        return VoskTranscriber(settings.vosk_model_path)
    from app.transcribe.whisper import WhisperTranscriber

    return WhisperTranscriber(
        model=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        cpu_threads=settings.whisper_cpu_threads,
        num_workers=settings.stt_workers + 1,
        beam_size=settings.whisper_beam_size,
        no_speech_threshold=settings.no_speech_threshold,
        min_avg_logprob=settings.min_avg_logprob,
        max_compression_ratio=settings.max_compression_ratio,
        hallucinations=settings.hallucinations,
    )


def _load_embedder(settings: Settings) -> Embedder:
    from app.interpret.embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(settings.embed_model, settings.embed_threads)


def _parse_meta(raw: object) -> Meta:
    if raw is None:
        raise HTTPException(422, "campo `meta` ausente")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise HTTPException(422, "`meta` não é JSON válido") from error
    if not isinstance(data, dict):
        raise HTTPException(422, "`meta` tem de ser um objeto JSON")
    utterance_id = data.get("utterance_id")
    if not isinstance(utterance_id, str) or not utterance_id.strip():
        raise HTTPException(422, "`utterance_id` ausente")
    # Daqui para baixo nada é motivo de recusa: valor estranho é aceito e logado.
    return Meta(
        utterance_id=utterance_id,
        edge_id=str(data.get("edge_id") or "unknown")[:64],
        reason=str(data.get("reason") or "unknown")[:64],
        local_hypothesis=data.get("local_hypothesis"),
        local_confidence=data.get("local_confidence"),
    )


def create_app(
    settings: Settings | None = None,
    *,
    transcriber: Transcriber | None = None,
    embedder: Embedder | None | object = _UNSET,
    go2: Go2 | None = None,
) -> FastAPI:
    """Monta o app. Os parâmetros além de `settings` são para os testes."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        catalog = load_catalog(settings.catalog_path)
        client = go2 or Go2Client(
            settings.go2_api_url,
            settings.go2_api_timeout_s,
            settings.obstacle_avoidance_timeout_s,
        )
        # Modelos carregados e catálogo embutido antes de aceitar a primeira frase.
        stt = transcriber or await asyncio.to_thread(_load_transcriber, settings)
        encoder = (
            await asyncio.to_thread(_load_embedder, settings) if embedder is _UNSET else embedder
        )
        interpreter = await asyncio.to_thread(
            Interpreter,
            catalog,
            encoder,
            Thresholds(
                accept=settings.tau_accept,
                margin=settings.tau_margin,
                slot_accept=settings.tau_slot_accept,
                slot_margin=settings.tau_slot_margin,
                max_clauses=settings.max_clauses,
                stop_accept=settings.tau_stop_accept,
            ),
        )
        dispatcher = Dispatcher(
            client,
            catalog,
            Limits(
                max_commands=settings.queue_max_commands,
                max_total_s=settings.queue_max_total_s,
                replace=settings.replace_queue,
                cross_edge_policy=settings.cross_edge_policy,
                ensure_obstacle_avoidance=settings.ensure_obstacle_avoidance,
                obstacle_avoidance_ttl_s=settings.obstacle_avoidance_ttl_s,
            ),
        )
        dispatcher.start()
        pool = TranscriptionPool(stt, settings.stt_workers, settings.short_audio_s)
        service = Service(settings, catalog, interpreter, pool, dispatcher)
        app.state.service = service
        app.state.go2 = client
        stats_task = asyncio.create_task(_log_stats(service))
        log.info(
            "go2-server pronto",
            extra={
                "event": "startup",
                "stt_engine": settings.stt_engine if transcriber is None else "injetado",
                "embed_model": settings.embed_model,
                "catalog_ids": sorted(catalog.ids),
                "go2_api_url": settings.go2_api_url,
                "save_utterances": settings.save_utterances,
            },
        )
        try:
            yield
        finally:
            stats_task.cancel()
            with suppress(asyncio.CancelledError):
                await stats_task
            await dispatcher.close()
            pool.close()
            if isinstance(client, Go2Client):
                await client.aclose()

    app = FastAPI(title="go2-server", lifespan=lifespan)

    @app.post("/v1/utterance", response_model=UtteranceResponse)
    async def utterance(request: Request) -> UtteranceResponse:
        form = await request.form()
        audio_part, meta_part = form.get("audio"), form.get("meta")
        if isinstance(meta_part, UploadFile):
            meta_part = await meta_part.read()
        meta = _parse_meta(meta_part)
        if not isinstance(audio_part, UploadFile):
            raise HTTPException(422, "campo `audio` ausente")
        wav = await audio_part.read()
        try:
            audio = decode_wav(wav)
        except InvalidAudioError as error:
            raise HTTPException(422, f"`audio` inválido: {error}") from error
        service: Service = request.app.state.service
        return await service.handle(meta, audio, wav)

    @app.post("/v1/cancel", response_model=CancelResponse)
    async def cancel(body: CancelRequest, request: Request) -> CancelResponse:
        service: Service = request.app.state.service
        result = await service.dispatcher.cancel(body.reason, body.edge_id)
        return CancelResponse(cleared=result.cleared, stop_sent=result.stop_sent)

    @app.get("/health")
    async def health(request: Request) -> dict:
        service: Service = request.app.state.service
        client = request.app.state.go2
        api = await client.status() if isinstance(client, Go2Client) else None
        return {
            "status": "ok",
            "catalog_commands": len(service.catalog.ids),
            "queue_pending": service.dispatcher.pending,
            "last_dispatch_error": service.dispatcher.last_error,
            "go2_api": {
                "reachable": api is not None,
                "robot_connected": api.get("connected") if api else None,
            },
        }

    @app.get("/v1/stats")
    async def stats(request: Request) -> dict:
        """Contagem por `status` e latência média/p95, separadas por `reason`."""
        return request.app.state.service.stats.summary()

    return app


async def _log_stats(service: Service) -> None:
    while True:
        await asyncio.sleep(STATS_LOG_INTERVAL_S)
        summary = service.stats.summary()
        if summary:
            log.info("resumo por reason", extra={"event": "stats", "by_reason": summary})
