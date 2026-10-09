"""Fila e despacho, com a go2-api falsa e as esperas do catálogo encolhidas."""

import asyncio
import random

import pytest

from app.dispatch import Dispatcher, Limits

SIT_S = 3.0 * 0.02  # espera de `sit` no catálogo de teste


@pytest.fixture
async def make(fast_catalog, go2):
    created: list[Dispatcher] = []

    def factory(**limits) -> Dispatcher:
        limits.setdefault("max_total_s", 1.0)
        dispatcher = Dispatcher(go2, fast_catalog, Limits(**limits))
        dispatcher.start()
        created.append(dispatcher)
        return dispatcher

    yield factory
    for dispatcher in created:
        await dispatcher.close()


async def idle(dispatcher: Dispatcher) -> None:
    await asyncio.wait_for(dispatcher.wait_idle(), timeout=5)


async def test_runs_in_order_and_waits_between_commands(make, go2):
    dispatcher = make()
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert dispatcher.submit(["sit", "hello"], "tv1", "u1").accepted
    await idle(dispatcher)
    assert go2.calls == ["sit", "hello"]
    # sit (3 s) + hello (5 s), na escala do teste.
    assert loop.time() - started >= (3.0 + 5.0) * 0.02 * 0.9


async def test_new_sentence_replaces_the_queue(make, go2):
    dispatcher = make()
    dispatcher.submit(["sit", "hello", "stretch"], "tv1", "u1")
    await asyncio.sleep(SIT_S / 3)  # `sit` saiu e está na espera
    submission = dispatcher.submit(["stand_down"], "tv1", "u2")
    assert submission.accepted and submission.replaced == 3
    await idle(dispatcher)
    assert go2.calls == ["sit", "stand_down"]


async def test_append_when_replacement_is_off(make, go2):
    dispatcher = make(replace=False)
    dispatcher.submit(["sit"], "tv1", "u1")
    assert dispatcher.submit(["hello"], "tv1", "u2").replaced == 0
    await idle(dispatcher)
    assert go2.calls == ["sit", "hello"]


async def test_queue_full_when_replacement_is_off(make, go2):
    dispatcher = make(replace=False, max_commands=2)
    go2.hold = "sit"
    dispatcher.submit(["sit"], "tv1", "u1")
    await go2.holding.wait()
    assert dispatcher.submit(["hello", "stretch"], "tv1", "u2").accepted
    refused = dispatcher.submit(["stand_up"], "tv1", "u3")
    assert (refused.accepted, refused.reason) == (False, "queue_full")
    go2.release.set()
    await idle(dispatcher)
    assert go2.calls == ["sit", "hello", "stretch"]


async def test_limits(make, go2):
    dispatcher = make(max_commands=2, max_total_s=0.15)
    assert dispatcher.submit(["sit", "hello", "stretch"], "tv1", "u1").reason == "too_many_commands"
    # dance1 (10 s) + dance2 (10 s) = 0,4 s na escala do teste.
    assert dispatcher.submit(["dance1", "dance2"], "tv1", "u2").reason == "too_long"
    assert dispatcher.submit(["nao_existe"], "tv1", "u3").reason == "unknown_command"
    assert dispatcher.submit([], "tv1", "u4").reason == "no_commands"
    await asyncio.sleep(0.05)
    assert go2.calls == [] and dispatcher.pending == 0


async def test_rejected_sentence_does_not_touch_the_queue(make, go2):
    """Frase recusada pelos limites não substitui a fila em andamento."""
    dispatcher = make(max_commands=2)
    dispatcher.submit(["sit", "hello"], "tv1", "u1")
    assert not dispatcher.submit(["sit", "hello", "stretch"], "tv1", "u2").accepted
    await idle(dispatcher)
    assert go2.calls == ["sit", "hello"]


async def test_cancel_local_command_clears_without_stopping(make, go2):
    dispatcher = make()
    dispatcher.submit(["sit", "hello", "stretch"], "tv1", "u1")
    await asyncio.sleep(SIT_S / 3)
    result = await dispatcher.cancel("local_command", "tv1")
    assert (result.cleared, result.stop_sent) == (3, None)
    await idle(dispatcher)
    assert go2.calls == ["sit"]


async def test_stop_clears_calls_the_api_and_is_idempotent(make, go2):
    dispatcher = make()
    dispatcher.submit(["sit", "hello"], "tv1", "u1")
    await asyncio.sleep(SIT_S / 3)
    first = await dispatcher.cancel("stop", "tv1")
    second = await dispatcher.cancel("stop", "tv1")
    assert (first.cleared, first.stop_sent) == (2, True)
    assert (second.cleared, second.stop_sent) == (0, True)
    await idle(dispatcher)
    assert go2.calls == ["sit", "stop", "stop"]


async def test_stop_interrupts_the_wait_of_the_running_command(make, go2):
    dispatcher = make(max_total_s=5)
    loop = asyncio.get_running_loop()
    dispatcher.submit(["dance1"], "tv1", "u1")  # 0,2 s de espera
    await asyncio.sleep(0.02)
    started = loop.time()
    await dispatcher.cancel("stop")
    await idle(dispatcher)
    assert loop.time() - started < 0.1


async def test_stop_with_empty_queue_still_reaches_the_api(make, go2):
    result = await make().cancel("stop")
    assert (result.cleared, result.stop_sent) == (0, True)
    assert go2.calls == ["stop"]


async def test_stop_not_delivered_is_reported(make, go2):
    go2.fail_on = {"stop"}
    dispatcher = make()
    dispatcher.submit(["sit", "hello"], "tv1", "u1")
    await asyncio.sleep(SIT_S / 3)
    result = await dispatcher.cancel("stop")
    assert (result.cleared, result.stop_sent) == (2, False)
    await idle(dispatcher)
    assert "hello" not in go2.calls  # a fila é limpa mesmo sem a API


# ─── Corrida entre cancelamento e consumidor ───────────────────────────────


async def test_race_command_already_on_the_wire_arrives_before_the_stop(make, go2):
    """O comando já estava a caminho da go2-api quando a parada chegou.

    A parada tem de ser entregue DEPOIS dele (senão o robô executaria o
    comando após parar), e o resto da fila não sai.
    """
    dispatcher = make()
    go2.hold = "sit"
    dispatcher.submit(["sit", "hello"], "tv1", "u1")
    await go2.holding.wait()
    stopping = asyncio.create_task(dispatcher.cancel("stop"))
    await asyncio.sleep(0.02)
    assert not stopping.done()  # espera o envio em curso terminar
    go2.release.set()
    assert (await stopping).cleared == 2
    await idle(dispatcher)
    assert go2.calls == ["sit", "stop"]


async def test_race_command_about_to_be_sent_is_dropped(make, go2):
    """O consumidor já tirou o comando da fila, mas ainda não enviou."""
    dispatcher = make()
    await dispatcher._api_lock.acquire()  # segura o consumidor antes do envio
    dispatcher.submit(["sit", "hello"], "tv1", "u1")
    await asyncio.sleep(0.01)
    assert dispatcher._running is not None and len(dispatcher._queue) == 1
    stopping = asyncio.create_task(dispatcher.cancel("stop"))
    await asyncio.sleep(0.01)
    dispatcher._api_lock.release()
    assert (await stopping).cleared == 2
    await idle(dispatcher)
    assert go2.calls == ["stop"]


async def test_race_nothing_is_sent_after_a_stop(make, go2):
    """Parada em instantes aleatórios: depois dela, nenhum comando sai."""
    rng = random.Random(7)
    dispatcher = make(max_total_s=5)
    for round_ in range(40):
        dispatcher.submit(["stand_up", "sit", "hello"], "tv1", f"u{round_}")
        await asyncio.sleep(rng.uniform(0, 0.15))
        await dispatcher.cancel("stop")
        sent = len(go2.calls)
        assert go2.calls[-1] == "stop"
        await asyncio.sleep(0.02)
        assert len(go2.calls) == sent, go2.calls[sent:]
        assert dispatcher.pending == 0


async def test_race_replacement_drops_the_command_about_to_be_sent(make, go2):
    dispatcher = make()
    await dispatcher._api_lock.acquire()
    dispatcher.submit(["sit"], "tv1", "u1")
    await asyncio.sleep(0.01)
    dispatcher.submit(["stand_down"], "tv2", "u2")
    dispatcher._api_lock.release()
    await idle(dispatcher)
    assert go2.calls == ["stand_down"]


# ─── Falha da go2-api ──────────────────────────────────────────────────────


async def test_api_failure_clears_the_queue(make, go2):
    go2.fail_on = {"hello"}
    dispatcher = make()
    dispatcher.submit(["sit", "hello", "stretch"], "tv1", "u1")
    await idle(dispatcher)
    assert go2.calls == ["sit", "hello"]  # `stretch` não saiu
    assert "hello" in dispatcher.last_error

    go2.fail_on = set()
    assert dispatcher.submit(["stand_up"], "tv1", "u2").accepted
    await idle(dispatcher)
    assert go2.calls[-1] == "stand_up"  # o consumidor segue vivo


# ─── Dois clientes ─────────────────────────────────────────────────────────


async def test_two_edges_share_one_queue(make, go2):
    """A fila é global: a frase de um edge substitui a de outro."""
    dispatcher = make()
    dispatcher.submit(["sit", "hello"], "tv1", "u1")
    await asyncio.sleep(SIT_S / 3)
    assert dispatcher.submit(["stand_down"], "tv2", "u2").replaced == 2
    await idle(dispatcher)
    assert go2.calls == ["sit", "stand_down"]


async def test_two_edges_at_the_same_time(make, go2):
    """Várias frases de dois edges no mesmo instante: vale só a última, inteira."""
    dispatcher = make()

    async def speak(edge: str, commands: list[str], number: int) -> None:
        await asyncio.sleep(0)
        assert dispatcher.submit(commands, edge, f"{edge}-{number}").accepted

    await asyncio.gather(
        *(speak("tv1", ["sit", "hello"], n) for n in range(5)),
        *(speak("tv2", ["stand_up", "stretch"], n) for n in range(5)),
    )
    await idle(dispatcher)
    # Nenhuma mistura: os comandos executados são de uma frase só.
    assert go2.calls in (["sit", "hello"], ["stand_up", "stretch"])


async def test_two_edges_with_reject_policy(make, go2):
    dispatcher = make(cross_edge_policy="reject")
    dispatcher.submit(["sit", "hello"], "tv1", "u1")
    refused = dispatcher.submit(["stand_down"], "tv2", "u2")
    assert (refused.accepted, refused.reason) == (False, "busy_other_edge")
    assert dispatcher.submit(["stand_up"], "tv1", "u3").accepted  # o dono pode
    # A parada vale de qualquer cliente.
    assert (await dispatcher.cancel("stop", "tv2")).stop_sent
    assert dispatcher.submit(["stand_down"], "tv2", "u4").accepted
    await idle(dispatcher)
    assert go2.calls[-1] == "stand_down"


# ─── Movimento ─────────────────────────────────────────────────────────────


async def test_move_is_prepared_once(make, go2):
    """`balance_stand` antes do primeiro movimento, não entre movimentos."""
    dispatcher = make()
    dispatcher.submit(["move_forward", "turn_left"], "tv1", "u1")
    await idle(dispatcher)
    assert go2.calls == ["balance_stand", "move:0.3,0,0", "move:0,0,0.5"]


async def test_move_after_a_posture_is_prepared_again(make, go2):
    dispatcher = make()
    dispatcher.submit(["move_forward", "sit", "move_forward"], "tv1", "u1")
    await idle(dispatcher)
    assert go2.calls == [
        "balance_stand",
        "move:0.3,0,0",
        "sit",
        "balance_stand",
        "move:0.3,0,0",
    ]


async def test_local_command_invalidates_the_prepared_state(make, go2):
    dispatcher = make()
    dispatcher.submit(["move_forward"], "tv1", "u1")
    await idle(dispatcher)
    await dispatcher.cancel("local_command")
    dispatcher.submit(["move_forward"], "tv1", "u2")
    await idle(dispatcher)
    assert go2.calls.count("balance_stand") == 2


async def test_obstacle_avoidance_is_switched_on_before_moving(make, go2):
    go2.obstacle = False
    dispatcher = make()
    dispatcher.submit(["move_forward"], "tv1", "u1")
    await asyncio.wait_for(dispatcher.wait_idle(), timeout=5)
    assert go2.calls == ["obstacle_on", "balance_stand", "move:0.3,0,0"]


async def test_no_move_without_obstacle_avoidance_confirmed(make, go2):
    go2.obstacle = None
    go2.obstacle_after_put = None
    dispatcher = make()
    dispatcher.submit(["move_forward", "sit"], "tv1", "u1")
    await asyncio.wait_for(dispatcher.wait_idle(), timeout=5)
    assert go2.calls == ["obstacle_on"]  # nem o movimento nem o que vinha depois
    assert "desvio" in dispatcher.last_error


async def test_obstacle_check_can_be_disabled(make, go2):
    go2.obstacle = None
    dispatcher = make(ensure_obstacle_avoidance=False)
    dispatcher.submit(["move_forward"], "tv1", "u1")
    await idle(dispatcher)
    assert go2.calls == ["balance_stand", "move:0.3,0,0"]


async def test_stop_is_never_queued(make, go2):
    assert make().submit(["stop"], "tv1", "u1").reason == "unknown_command"
