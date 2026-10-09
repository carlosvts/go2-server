"""Endpoints, com a go2-api e a transcrição falsas."""

import asyncio
import json
import time

import httpx
import pytest

from app import main
from app.config import Settings
from tests.conftest import FakeGo2, FakeTranscriber, StubEmbedder, wav_bytes


@pytest.fixture
def stt() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, save_dir=tmp_path / "captures")


@pytest.fixture
async def make_client(monkeypatch, fast_catalog, go2, stt):
    """Sobe o app com catálogo de esperas curtas e dependências falsas."""
    monkeypatch.setattr(main, "load_catalog", lambda path: fast_catalog)
    stack = []

    async def factory(settings: Settings, go2_api: FakeGo2 = go2) -> httpx.AsyncClient:
        app = main.create_app(settings, transcriber=stt, embedder=StubEmbedder(), go2=go2_api)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        client.app = app
        stack.append((client, lifespan))
        return client

    yield factory
    for client, lifespan in stack:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.fixture
async def client(make_client, settings):
    return await make_client(settings)


def meta(utterance_id="u1", reason="unk", **extra) -> str:
    base = {
        "utterance_id": utterance_id,
        "edge_id": "tvbox-sala",
        "reason": reason,
        "local_hypothesis": "[unk]",
        "local_confidence": 0.4,
    }
    return json.dumps({**base, **extra})


async def post(client, audio: bytes | None, meta_json: str | None):
    files = {} if audio is None else {"audio": ("utterance.wav", audio, "audio/wav")}
    data = {} if meta_json is None else {"meta": meta_json}
    if not files:
        files = {"other": ("x.txt", b"x", "text/plain")}  # mantém o corpo multipart
    return await client.post("/v1/utterance", files=files, data=data)


async def drain(client) -> None:
    await asyncio.wait_for(client.app.state.service.dispatcher.wait_idle(), timeout=5)


# ─── Contrato ──────────────────────────────────────────────────────────────


async def test_health(client):
    body = (await client.get("/health")).json()
    assert body["status"] == "ok" and body["queue_pending"] == 0


async def test_command_is_enqueued_and_dispatched(client, stt, go2):
    response = await post(client, stt.say(1.0, "Senta."), meta())
    assert response.status_code == 200
    assert response.json() == {
        "utterance_id": "u1",
        "status": "enqueued",
        "transcript": "Senta.",
        "commands": [{"id": "sit"}],
        "confidence": 1.0,
        "reject_reason": None,
    }
    await drain(client)
    assert go2.calls == ["sit"]


async def test_compound_sentence_with_movement(client, stt, go2):
    audio = stt.say(3.0, "Levanta e anda para frente")
    body = (await post(client, audio, meta())).json()
    assert body["status"] == "enqueued"
    assert body["commands"] == [{"id": "stand_up"}, {"id": "move_forward"}]
    await drain(client)
    assert go2.calls == ["stand_up", "balance_stand", "move:0.3,0,0"]


async def test_thin_mode_wake_word_and_null_local_fields(client, stt, go2):
    """Modo thin: reason="wake_word", `local_*` nulos e a wake word no áudio."""
    audio = stt.say(2.0, "Hey, Jarvis. Senta!")
    body = (
        await post(
            client, audio, meta(reason="wake_word", local_hypothesis=None, local_confidence=None)
        )
    ).json()
    assert body["status"] == "enqueued"
    assert body["commands"] == [{"id": "sit"}]
    assert body["transcript"] == "Hey, Jarvis. Senta!"
    await drain(client)
    assert go2.calls == ["sit"]


async def test_thin_mode_without_local_fields_at_all(client, stt):
    audio = stt.say(2.0, "hey jarvis levanta")
    meta_json = json.dumps({"utterance_id": "u9", "edge_id": "thin-1", "reason": "wake_word"})
    assert (await post(client, audio, meta_json)).json()["status"] == "enqueued"


async def test_only_the_wake_word_is_ignored(client, stt, go2):
    body = (await post(client, stt.say(0.8, "Hey Jarvis."), meta(reason="wake_word"))).json()
    assert body["status"] == "ignored"
    assert body["commands"] == [] and body["reject_reason"] is None
    assert go2.calls == []


async def test_spoken_stop_clears_the_queue_and_stops(client, stt, go2):
    await post(client, stt.say(3.0, "senta, levanta e deita"), meta("u1"))
    body = (await post(client, stt.say(1.0, "Hey Jarvis, para!"), meta("u2", "wake_word"))).json()
    assert body["status"] == "stopped"
    assert body["commands"] == [{"id": "stop"}]
    await drain(client)
    assert go2.calls == ["sit", "stop"]


async def test_stop_not_delivered_is_not_reported_as_stopped(client, stt, go2):
    go2.fail_on = {"stop"}
    body = (await post(client, stt.say(1.0, "para"), meta())).json()
    assert (body["status"], body["reject_reason"]) == ("rejected", "stop_not_delivered")


async def test_rejections(client, stt, go2):
    cases = [
        (1.1, "não senta", "negation"),
        (1.2, "toca uma música", "low_similarity"),
        (1.3, "senta e toca uma música", "low_similarity"),
        (1.4, "anda", "movement_missing_direction"),
    ]
    for number, (seconds, text, reason) in enumerate(cases):
        body = (await post(client, stt.say(seconds, text), meta(f"u{number}"))).json()
        assert (body["status"], body["reject_reason"]) == ("rejected", reason), text
        assert body["commands"] == []
    assert go2.calls == []


async def test_transcription_rejection_is_not_interpreted(client, stt, go2):
    """Silêncio/alucinação: mesmo que o texto pareça comando, nada executa."""
    audio = stt.say(1.0, "senta", reject="no_speech")
    body = (await post(client, audio, meta())).json()
    assert (body["status"], body["reject_reason"]) == ("rejected", "no_speech")
    assert go2.calls == []


async def test_doubtful_transcription_of_a_stop_still_stops(client, stt, go2):
    """Parar por engano é seguro; deixar de parar não é."""
    audio = stt.say(1.0, "Pare!", reject="low_confidence")
    assert (await post(client, audio, meta())).json()["status"] == "stopped"
    assert go2.calls == ["stop"]


async def test_unknown_reason_is_accepted(client, stt):
    response = await post(client, stt.say(1.0, "senta"), meta(reason="algo_novo"))
    assert response.status_code == 200 and response.json()["status"] == "enqueued"
    assert "algo_novo" in (await client.get("/v1/stats")).json()


@pytest.mark.parametrize(
    ("audio", "meta_json"),
    [
        (None, meta()),  # áudio ausente
        (b"isto nao e um wav", meta()),
        (b"", meta()),
        (wav_bytes(0.5, rate=8000), meta()),  # taxa errada
        (wav_bytes(0.5, channels=2), meta()),  # estéreo
        (wav_bytes(0.5, width=1), meta()),  # 8 bits
        (wav_bytes(0.5), None),  # meta ausente
        (wav_bytes(0.5), "{nao e json"),
        (wav_bytes(0.5), json.dumps({"edge_id": "x", "reason": "unk"})),  # sem utterance_id
        (wav_bytes(0.5), json.dumps({"utterance_id": "", "reason": "unk"})),
    ],
)
async def test_invalid_requests_get_422(client, go2, audio, meta_json):
    assert (await post(client, audio, meta_json)).status_code == 422
    assert go2.calls == []


async def test_meta_sent_as_a_json_part_like_the_tvbox_does(client, stt):
    """O edge manda `meta` como parte com Content-Type application/json."""
    audio = stt.say(1.0, "senta")
    boundary = "abc123"
    parts = (
        ('name="audio"; filename="utterance.wav"', "audio/wav", audio),
        ('name="meta"', "application/json", meta().encode()),
    )
    body = (
        b"".join(
            f"--{boundary}\r\nContent-Disposition: form-data; {disposition}\r\n"
            f"Content-Type: {content_type}\r\n\r\n".encode()
            + content
            + b"\r\n"
            for disposition, content_type, content in parts
        )
        + f"--{boundary}--\r\n".encode()
    )
    response = await client.post(
        "/v1/utterance",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.json()["status"] == "enqueued"


# ─── Idempotência ──────────────────────────────────────────────────────────


async def test_same_utterance_id_runs_once(client, stt, go2):
    audio = stt.say(1.0, "senta")
    first = (await post(client, audio, meta("same"))).json()
    await drain(client)
    second = (await post(client, audio, meta("same"))).json()
    await drain(client)
    assert first == second
    assert go2.calls == ["sit"] and stt.calls == 1


async def test_same_utterance_id_at_the_same_time_runs_once(client, stt, go2):
    audio = stt.say(1.0, "senta", delay=0.05)
    responses = await asyncio.gather(*(post(client, audio, meta("same")) for _ in range(4)))
    await drain(client)
    assert len({response.text for response in responses}) == 1
    assert go2.calls == ["sit"] and stt.calls == 1


# ─── Cancel ────────────────────────────────────────────────────────────────


async def test_cancel_stop(client, stt, go2):
    await post(client, stt.say(2.0, "senta e levanta"), meta())
    await asyncio.sleep(0.01)  # o consumidor envia o primeiro comando
    response = await client.post("/v1/cancel", json={"edge_id": "tvbox-sala", "reason": "stop"})
    assert response.json() == {"cleared": 2, "stop_sent": True}
    again = await client.post("/v1/cancel", json={"edge_id": "tvbox-sala", "reason": "stop"})
    assert again.json() == {"cleared": 0, "stop_sent": True}
    await drain(client)
    assert go2.calls == ["sit", "stop", "stop"]


async def test_cancel_local_command_without_edge_id(client, stt, go2):
    """O edge atual manda só `{"reason": ...}`."""
    await post(client, stt.say(2.0, "senta e levanta"), meta())
    await asyncio.sleep(0.01)  # o consumidor envia o primeiro comando
    response = await client.post("/v1/cancel", json={"reason": "local_command"})
    assert response.json() == {"cleared": 2, "stop_sent": None}
    await drain(client)
    assert go2.calls == ["sit"]


async def test_cancel_with_invalid_reason(client):
    assert (await client.post("/v1/cancel", json={"reason": "outro"})).status_code == 422


# ─── Vários clientes e prioridade da parada ────────────────────────────────


async def test_two_edges_at_the_same_time(client, stt, go2):
    """Dois edge_ids ao mesmo tempo: as duas respondem, e só uma frase executa."""
    first = stt.say(2.0, "senta e cumprimente", delay=0.03)
    second = stt.say(2.2, "levanta e alonga", delay=0.03)
    one, two = await asyncio.gather(
        post(client, first, meta("a1", edge_id="tv1")),
        post(client, second, meta("b1", "wake_word", edge_id="tv2")),
    )
    assert one.json()["status"] == two.json()["status"] == "enqueued"
    await drain(client)
    # A frase que chegou por último executa inteira; da outra, no máximo o
    # primeiro comando já tinha saído.
    assert go2.calls in (
        ["sit", "hello"],
        ["stand_up", "stretch"],
        ["sit", "stand_up", "stretch"],
        ["stand_up", "sit", "hello"],
    )


async def test_stop_from_another_edge_stops_everything(client, stt, go2):
    await post(client, stt.say(2.0, "senta e levanta"), meta("a1", edge_id="tv1"))
    body = (await post(client, stt.say(0.7, "pare"), meta("b1", "wake_word", edge_id="tv2"))).json()
    assert body["status"] == "stopped"
    await drain(client)
    assert go2.calls == ["sit", "stop"]


async def test_spoken_stop_does_not_wait_behind_a_long_transcription(client, stt, go2):
    """A parada (áudio curto) usa a faixa reservada e responde antes da frase longa."""
    long_audio = stt.say(6.0, "senta", delay=0.4)
    stop_audio = stt.say(0.9, "para")
    long_request = asyncio.create_task(post(client, long_audio, meta("long")))
    await asyncio.sleep(0.05)
    started = time.monotonic()
    body = (await post(client, stop_audio, meta("stop", "wake_word"))).json()
    assert body["status"] == "stopped"
    assert time.monotonic() - started < 0.2
    assert not long_request.done()
    await long_request


async def test_stale_utterance_is_not_executed(make_client, settings, stt, go2):
    """Idade medida no servidor: transcrição lenta demais não vira movimento."""
    client = await make_client(replace_settings(settings, max_utterance_age_s=0.05))
    body = (await post(client, stt.say(1.0, "senta", delay=0.15), meta())).json()
    assert (body["status"], body["reject_reason"]) == ("rejected", "stale")
    assert go2.calls == []


async def test_stale_stop_still_stops(make_client, settings, stt, go2):
    client = await make_client(replace_settings(settings, max_utterance_age_s=0.05))
    body = (await post(client, stt.say(1.0, "para", delay=0.15), meta())).json()
    assert body["status"] == "stopped"


def replace_settings(settings: Settings, **changes) -> Settings:
    return settings.model_copy(update=changes)


# ─── Observabilidade e captura ─────────────────────────────────────────────


async def test_stats_by_reason(client, stt):
    await post(client, stt.say(1.0, "senta"), meta("u1", "unk"))
    await post(client, stt.say(1.1, "toca uma música"), meta("u2", "unk"))
    await post(client, stt.say(1.2, "hey jarvis para"), meta("u3", "wake_word"))
    await post(client, stt.say(1.3, "hey jarvis"), meta("u4", "wake_word"))
    stats = (await client.get("/v1/stats")).json()
    assert stats["unk"]["by_status"] == {"enqueued": 1, "rejected": 1}
    assert stats["wake_word"]["by_status"] == {"stopped": 1, "ignored": 1}
    assert stats["wake_word"]["stop_latency"]["count"] == 1
    assert stats["unk"]["latency"]["p95_ms"] >= stats["unk"]["latency"]["mean_ms"] > 0
    assert stats["unk"]["stop_latency"] is None


async def test_structured_log_has_the_whole_story(client, stt, caplog):
    with caplog.at_level("INFO", logger="app.service"):
        await post(client, stt.say(1.0, "Hey Jarvis, senta"), meta(reason="wake_word"))
    record = next(r for r in caplog.records if getattr(r, "event", "") == "utterance")
    assert record.utterance_id == "u1" and record.edge_id == "tvbox-sala"
    assert record.reason == "wake_word"
    assert record.transcript_raw == "Hey Jarvis, senta" and record.transcript == "senta"
    assert record.wake_word_removed is True
    assert record.commands == ["sit"] and record.status == "enqueued"
    assert record.clauses[0]["method"] == "lexical" and record.clauses[0]["score"] == 1.0
    for field in ("stt_ms", "interpret_ms", "dispatch_ms", "total_ms"):
        assert getattr(record, field) >= 0


async def test_save_flag_stores_audio_and_transcript(make_client, settings, stt):
    client = await make_client(replace_settings(settings, save_utterances=True))
    audio = stt.say(1.0, "senta")
    await post(client, audio, meta("abc-123"))
    await post(client, audio, meta("../../etc/passwd"))  # o ID não vira caminho
    saved = sorted(path.name for path in settings.save_dir.rglob("*") if path.is_file())
    assert "abc-123.wav" in saved and "abc-123.json" in saved
    assert len(saved) == 4 and all(".." not in name and "/" not in name for name in saved)
    record = json.loads(next(settings.save_dir.rglob("abc-123.json")).read_text())
    assert record["transcript"] == "senta" and record["commands"] == ["sit"]
    assert next(settings.save_dir.rglob("abc-123.wav")).read_bytes() == audio


async def test_nothing_is_saved_by_default(client, settings, stt):
    await post(client, stt.say(1.0, "senta"), meta())
    assert not settings.save_dir.exists()
