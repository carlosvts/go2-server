import asyncio
import io
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.audio import SAMPLE_RATE
from app.catalog import Call, Catalog, load_catalog
from app.go2_client import OBSTACLE_AVOIDANCE, Go2ApiError
from app.transcribe.base import Transcription

ROOT = Path(__file__).resolve().parent.parent
# As durações do catálogo são de segundos; nos testes viram centésimos.
TIME_SCALE = 0.02


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return load_catalog(ROOT / "catalog.yaml")


@pytest.fixture(scope="session")
def fast_catalog(catalog: Catalog) -> Catalog:
    """O catálogo real, com as esperas encolhidas."""
    return replace(
        catalog,
        commands={
            name: replace(command, max_duration_s=command.max_duration_s * TIME_SCALE)
            for name, command in catalog.commands.items()
        },
        rules={
            key: replace(rule, max_duration_s=rule.max_duration_s * TIME_SCALE)
            for key, rule in catalog.rules.items()
        },
        prepare=replace(catalog.prepare, wait_s=catalog.prepare.wait_s * TIME_SCALE),
    )


def label(call: Call) -> str:
    """Nome curto de uma chamada à go2-api, para comparar nos testes."""
    if call.endpoint == OBSTACLE_AVOIDANCE:
        return "obstacle_on"
    if call.endpoint.endswith("/stop"):
        return "stop"
    if call.endpoint.endswith("/move"):
        return "move:{vx:g},{vy:g},{vyaw:g}".format(**call.body)
    return call.body["cmd"]


class FakeGo2:
    """go2-api falsa: registra as chamadas, na ordem em que chegaram."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_on: set[str] = set()
        self.obstacle: bool | None = True
        self.obstacle_after_put: bool | None = True
        # Quando definido, a chamada com este nome fica presa até `release`.
        self.hold: str | None = None
        self.holding = asyncio.Event()
        self.release = asyncio.Event()

    async def call(self, call: Call) -> None:
        name = label(call)
        if name == self.hold:
            self.holding.set()
            await self.release.wait()
        self.calls.append(name)
        if name in self.fail_on:
            raise Go2ApiError(f"falha simulada em {name}")
        if name == "obstacle_on":
            self.obstacle = self.obstacle_after_put

    async def obstacle_avoidance(self) -> bool | None:
        return self.obstacle


@pytest.fixture
def go2() -> FakeGo2:
    return FakeGo2()


class StubEmbedder:
    """Embeddings sob controle do teste.

    Cada texto novo ganha um eixo só dele (cosseno 0 com todos os outros).
    `mix` define um texto como combinação de outros, e com isso o cosseno
    exato dele com cada exemplo do catálogo.
    """

    DIM = 1024

    def __init__(self) -> None:
        self._axis: dict[str, int] = {}
        self._mixes: dict[str, dict[str, float]] = {}

    def mix(self, text: str, weights: dict[str, float]) -> None:
        self._mixes[text] = weights

    def _basis(self, text: str) -> np.ndarray:
        vector = np.zeros(self.DIM, dtype=np.float32)
        vector[self._axis.setdefault(text, len(self._axis))] = 1.0
        return vector

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            if text in self._mixes:
                vector = sum(w * self._basis(other) for other, w in self._mixes[text].items())
            else:
                vector = self._basis(text)
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)


def wav_bytes(seconds: float, rate: int = SAMPLE_RATE, channels: int = 1, width: int = 2) -> bytes:
    """Um tom, só para ter áudio válido: a transcrição dos testes é falsa."""
    samples = int(seconds * rate)
    tone = (np.sin(np.arange(samples) * 2 * np.pi * 220 / rate) * 8000).astype(np.int16)
    data = np.repeat(tone, channels).tobytes() if width == 2 else bytes(samples * channels)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(data)
    return buffer.getvalue()


class FakeTranscriber:
    """Transcrição falsa: o texto é escolhido pela duração do áudio."""

    def __init__(self) -> None:
        self._by_samples: dict[int, tuple[str, str | None, float]] = {}
        self.calls = 0

    def say(
        self, seconds: float, text: str, reject: str | None = None, delay: float = 0.0
    ) -> bytes:
        """Registra o que "foi dito" num áudio desta duração e devolve o WAV."""
        self._by_samples[int(seconds * SAMPLE_RATE)] = (text, reject, delay)
        return wav_bytes(seconds)

    def transcribe(self, audio: np.ndarray) -> Transcription:
        import time

        self.calls += 1
        text, reject, delay = self._by_samples[len(audio)]
        time.sleep(delay)
        return Transcription(text, reject, {"fake": True})
