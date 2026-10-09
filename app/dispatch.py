"""A fila de execução: o único lugar do sistema que tem fila.

Um consumidor só. Para cada comando, chama a go2-api e espera a duração do
catálogo (a API responde 202 na hora e não avisa o fim).

A fila é GLOBAL: o robô é um só, então frases de edge_ids diferentes disputam
a mesma fila (`cross_edge_policy`).

Corrida entre cancelamento e consumidor, em duas partes:

- Contador de geração. Todo cancelamento, substituição ou falha incrementa
  `_gen`. Cada comando carrega a geração em que entrou, e o consumidor confere
  antes de cada envio e depois de cada espera: comando de geração velha não sai.
- `_api_lock`. "Conferir a geração e enviar" é uma seção crítica, e a parada
  só é enviada depois de pegar o mesmo lock. Assim um comando que já estava a
  caminho chega à go2-api ANTES do `stop`, nunca depois.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from app.catalog import Call, Catalog, Step
from app.go2_client import OBSTACLE_AVOIDANCE, Go2ApiError

log = logging.getLogger(__name__)

# Depois de mandar ligar o desvio de obstáculo, a confirmação é perguntada
# algumas vezes (mesmos números do edge).
CONFIRM_TRIES = 3
CONFIRM_WAIT_S = 0.3


class Go2(Protocol):
    async def call(self, call: Call) -> None: ...
    async def obstacle_avoidance(self) -> bool | None: ...


@dataclass(frozen=True)
class Limits:
    max_commands: int = 5
    max_total_s: float = 15.0
    replace: bool = True
    cross_edge_policy: str = "replace"
    ensure_obstacle_avoidance: bool = True
    obstacle_avoidance_ttl_s: float = 30.0


@dataclass(frozen=True)
class Submission:
    accepted: bool
    reason: str | None = None
    replaced: int = 0  # comandos descartados pela substituição


@dataclass(frozen=True)
class Cancellation:
    cleared: int
    stop_sent: bool | None = None  # None = não era uma parada
    error: str | None = None


@dataclass
class _Entry:
    gen: int
    step: Step
    utterance_id: str
    edge_id: str


class Dispatcher:
    def __init__(self, go2: Go2, catalog: Catalog, limits: Limits | None = None) -> None:
        self._go2 = go2
        self._catalog = catalog
        self._limits = limits or Limits()
        self._gen = 0
        self._queue: deque[_Entry] = deque()
        self._running: _Entry | None = None
        self._changed = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._api_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        # O robô está em `balance_stand`, até onde o servidor sabe.
        self._ready_to_move = False
        self._obstacle_ok_until = 0.0
        self.last_error: str | None = None

    # ─── Ciclo de vida ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._task = asyncio.create_task(self._consume(), name="dispatcher")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def wait_idle(self) -> None:
        """Volta quando a fila esvaziou e nada está em execução."""
        await self._idle.wait()

    @property
    def pending(self) -> int:
        return len(self._queue) + (1 if self._is_running() else 0)

    def _is_running(self) -> bool:
        return self._running is not None and self._running.gen == self._gen

    # ─── Entrada ───────────────────────────────────────────────────────────

    def submit(self, command_ids: list[str], edge_id: str, utterance_id: str) -> Submission:
        """Enfileira os comandos de uma frase, todos ou nenhum.

        Síncrono de propósito: sem `await` no meio, duas frases simultâneas
        não se intercalam, e a resposta já sabe se a frase entrou.
        """
        limits = self._limits
        try:
            steps = [self._catalog.step(command_id) for command_id in command_ids]
        except KeyError:
            return Submission(False, "unknown_command")
        if not steps:
            return Submission(False, "no_commands")
        if len(steps) > limits.max_commands:
            return Submission(False, "too_many_commands")
        total = self._total_s(steps)
        if total > limits.max_total_s:
            return Submission(False, "too_long")

        owner = self._owner()
        if limits.cross_edge_policy == "reject" and owner not in (None, edge_id):
            return Submission(False, "busy_other_edge")

        replaced = 0
        if limits.replace:
            replaced = self._clear()
        else:
            waiting = [entry.step for entry in self._queue]
            if len(waiting) + len(steps) > limits.max_commands:
                return Submission(False, "queue_full")
            if self._total_s(waiting) + total > limits.max_total_s:
                return Submission(False, "queue_full")
        for step in steps:
            self._queue.append(_Entry(self._gen, step, utterance_id, edge_id))
        self._idle.clear()
        self._changed.set()
        return Submission(True, replaced=replaced)

    async def cancel(self, reason: str, edge_id: str | None = None) -> Cancellation:
        """Limpa a fila e interrompe o comando em andamento.

        Com `reason="stop"`, também manda a parada à go2-api. É idempotente:
        repetir só reenvia o `stop`, que a API aceita sempre.
        """
        cleared = self._clear()
        log.info("cancel reason=%s edge_id=%s cleared=%d", reason, edge_id, cleared)
        if reason != "stop":
            # Um comando que o servidor não viu mexeu no robô: não dá mais
            # para supor que ele está pronto para andar.
            self._ready_to_move = False
            return Cancellation(cleared)
        async with self._api_lock:
            try:
                await self._go2.call(self._catalog.stop_call)
            except Go2ApiError as error:
                self.last_error = str(error)
                log.error("parada não entregue à go2-api: %s", error)
                return Cancellation(cleared, stop_sent=False, error=str(error))
        return Cancellation(cleared, stop_sent=True)

    def _owner(self) -> str | None:
        if self._is_running():
            return self._running.edge_id
        return self._queue[0].edge_id if self._queue else None

    def _total_s(self, steps: list[Step]) -> float:
        """Duração no pior caso, contando a postura de preparo dos movimentos."""
        prepare = self._catalog.prepare
        extra = prepare.wait_s if prepare and any(step.is_move for step in steps) else 0.0
        return sum(step.wait_s for step in steps) + extra

    def _clear(self) -> int:
        """Invalida tudo o que está na fila e em execução. Devolve quantos eram."""
        cleared = self.pending
        self._gen += 1
        self._queue.clear()
        self._changed.set()
        return cleared

    # ─── Consumidor ────────────────────────────────────────────────────────

    async def _consume(self) -> None:
        while True:
            while not self._queue:
                self._running = None
                self._idle.set()
                self._changed.clear()
                await self._changed.wait()
            entry = self._queue.popleft()
            self._running = entry
            try:
                await self._execute(entry)
            except Go2ApiError as error:
                # Falha: nada do que vinha depois é executado.
                self._ready_to_move = False
                self.last_error = str(error)
                dropped = self._clear() if entry.gen == self._gen else 0
                log.error(
                    "falha na go2-api em %s (utterance_id=%s): %s; %d comando(s) descartado(s)",
                    entry.step.id,
                    entry.utterance_id,
                    error,
                    dropped,
                )
            except Exception:
                log.exception("erro inesperado no consumidor; fila limpa")
                self._clear()

    async def _execute(self, entry: _Entry) -> None:
        step, gen = entry.step, entry.gen
        prepare = self._catalog.prepare
        if step.is_move:
            if not await self._obstacle_avoidance_on(gen):
                return
            if prepare is not None and not self._ready_to_move:
                if not await self._send(prepare, entry):
                    return
                if not await self._sleep(prepare.wait_s, gen):
                    return
        if not await self._send(step, entry):
            return
        self._ready_to_move = step.is_move or (prepare is not None and step.id == prepare.id)
        await self._sleep(step.wait_s, gen)

    async def _send(self, step: Step, entry: _Entry) -> bool:
        """Envia um comando se a geração ainda vale. False = foi cancelado."""
        async with self._api_lock:
            if entry.gen != self._gen:
                return False
            started = time.monotonic()
            await self._go2.call(step.call)
        log.info(
            "despachado %s",
            step.id,
            extra={
                "event": "dispatch",
                "utterance_id": entry.utterance_id,
                "edge_id": entry.edge_id,
                "command": step.id,
                "dispatch_ms": round((time.monotonic() - started) * 1000, 1),
            },
        )
        return True

    async def _sleep(self, seconds: float, gen: int) -> bool:
        """Espera `seconds`, ou menos se a geração mudar. False = foi cancelado."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        while self._gen == gen:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return True
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except TimeoutError:
                return True
        return False

    async def _obstacle_avoidance_on(self, gen: int) -> bool:
        """Garante o desvio de obstáculo ligado. False = cancelado no meio.

        Levanta `Go2ApiError` sem a confirmação: o movimento não sai. As
        leituras ficam fora do `_api_lock`, porque esperam o robô por até
        alguns segundos e a parada não pode ficar atrás delas.
        """
        limits = self._limits
        if not limits.ensure_obstacle_avoidance or time.monotonic() < self._obstacle_ok_until:
            return True
        enabled = await self._go2.obstacle_avoidance()
        if enabled is not True:
            async with self._api_lock:
                if gen != self._gen:
                    return False
                await self._go2.call(Call("PUT", OBSTACLE_AVOIDANCE, {"enabled": True}))
            for _ in range(CONFIRM_TRIES):
                if not await self._sleep(CONFIRM_WAIT_S, gen):
                    return False
                enabled = await self._go2.obstacle_avoidance()
                if enabled is True:
                    break
        if gen != self._gen:
            return False
        if enabled is not True:
            raise Go2ApiError("desvio de obstáculo não confirmado como ligado")
        self._obstacle_ok_until = time.monotonic() + limits.obstacle_avoidance_ttl_s
        return True
