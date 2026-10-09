"""Configuração, lida de variáveis de ambiente / `.env` (prefixo `GO2S_`).

Os padrões são conservadores de propósito: ver `docs/riscos.md`.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="GO2S_", extra="ignore", frozen=True
    )

    # ─── go2-api ───────────────────────────────────────────────────────────
    go2_api_url: str = "http://127.0.0.1:8000"
    go2_api_timeout_s: float = Field(default=2.0, gt=0)
    # Antes de um movimento, confere (e liga) o desvio de obstáculo do robô,
    # como o edge faz. Sem a confirmação, o movimento não sai.
    ensure_obstacle_avoidance: bool = True
    obstacle_avoidance_timeout_s: float = Field(default=4.0, gt=0)
    # Por quanto tempo uma confirmação do desvio de obstáculo vale.
    obstacle_avoidance_ttl_s: float = Field(default=30.0, ge=0)

    catalog_path: Path = ROOT / "catalog.yaml"

    # ─── Transcrição ───────────────────────────────────────────────────────
    stt_engine: Literal["whisper", "vosk"] = "whisper"
    whisper_model: str = "medium"
    # "auto" usa a GPU (CUDA) se houver e funcionar, senão a CPU. O `medium` é
    # pensado para a GPU: na CPU ele pode passar do tempo que a TV Box espera.
    whisper_device: Literal["auto", "cuda", "cpu"] = "auto"
    # "auto" = float16 na GPU e int8 na CPU.
    whisper_compute_type: str = "auto"
    # Threads de CPU por transcrição (só valem na CPU). Baixo de propósito: o servidor divide o
    # PC com a go2-api, que reenvia o `Move` a 40 Hz.
    whisper_cpu_threads: int = Field(default=4, ge=1)
    whisper_beam_size: int = Field(default=5, ge=1)
    vosk_model_path: Path = ROOT / "models" / "vosk-model-pt-fb-v0.1.1-20220516_2113"
    # Transcrições simultâneas, fora a faixa reservada para áudio curto.
    stt_workers: int = Field(default=1, ge=1)
    # Áudio até esta duração usa a faixa reservada: é onde chega a parada
    # falada do modo thin, que não pode esperar atrás de uma frase longa.
    short_audio_s: float = Field(default=2.5, gt=0)
    # Rejeição de alucinação do Whisper (silêncio, ruído).
    no_speech_threshold: float = Field(default=0.6, ge=0, le=1)
    min_avg_logprob: float = -1.0
    max_compression_ratio: float = 2.4
    hallucinations: list[str] = [
        "legendas pela comunidade amara.org",
        "legendas pela comunidade amara org",
        "obrigado por assistir",
        "obrigado",
        "obrigada",
        "tchau",
        "tchau tchau",
        "e aí",
        "inscreva-se no canal",
        "até a próxima",
        "www.opensubtitles.org",
    ]

    # ─── Wake word ─────────────────────────────────────────────────────────
    # Grafias que o motor de transcrição costuma produzir para a wake word do
    # edge ("hey jarvis"). Comparadas com tolerância a erro pequeno.
    wake_words: list[str] = [
        "hey jarvis",
        "ei jarvis",
        "hei jarvis",
        "rei jarvis",
        "e jarvis",
        "ok jarvis",
        "jarvis",
    ]

    # ─── Interpretação ─────────────────────────────────────────────────────
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_threads: int = Field(default=2, ge=1)
    # Limiar alto de propósito: é o ponto sem falso positivo em eval/phrases.yaml
    # (scripts/calibrate.py). Abaixo de 0.95 o MiniLM já confunde frases curtas.
    tau_accept: float = Field(default=0.95, ge=0, le=1)
    tau_margin: float = Field(default=0.10, ge=0, le=1)
    tau_slot_accept: float = Field(default=0.95, ge=0, le=1)
    tau_slot_margin: float = Field(default=0.10, ge=0, le=1)
    # Limiar do que se parece com parada ("fica parado"). Mais frouxo de
    # propósito: parar por engano é seguro.
    tau_stop_accept: float = Field(default=0.80, ge=0, le=1)
    max_clauses: int = Field(default=3, ge=1)

    # ─── Fila ──────────────────────────────────────────────────────────────
    queue_max_commands: int = Field(default=5, ge=1)
    queue_max_total_s: float = Field(default=15.0, gt=0)
    # Uma frase nova aceita substitui a fila (e interrompe o comando em
    # andamento). False = a frase nova entra no fim, se couber.
    replace_queue: bool = True
    # A fila é uma só para todos os edge_ids, porque o robô é um só. "replace":
    # a frase de um cliente substitui a de outro. "reject": enquanto houver
    # comandos de um cliente, a frase de outro é recusada (parada e cancel
    # valem sempre, de qualquer cliente).
    cross_edge_policy: Literal["replace", "reject"] = "replace"
    # Frase mais velha que isto ao terminar de ser interpretada não é
    # executada. Idade medida pelo relógio do servidor, do recebimento.
    max_utterance_age_s: float = Field(default=4.0, gt=0)
    idempotency_ttl_s: float = Field(default=600.0, gt=0)

    # ─── Calibração e observabilidade ──────────────────────────────────────
    save_utterances: bool = False
    save_dir: Path = ROOT / "captures"
    stats_window: int = Field(default=1000, ge=10)
