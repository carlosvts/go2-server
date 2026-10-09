"""Compara motores de transcrição NESTA máquina: latência, memória e erro.

    uv run python scripts/bench_stt.py                          # todos, um por processo
    uv run python scripts/bench_stt.py --engines whisper:base,whisper:small
    uv run python scripts/bench_stt.py --audio-dir gravacoes/   # fala real

Rode no PC que vai hospedar o servidor: os números só valem para o hardware
onde foram medidos.

Sem `--audio-dir`, o áudio é fala SINTÉTICA (Piper ou espeak-ng, ver tts.py) das frases de
`eval/phrases.yaml`, mais três trechos sem fala (silêncio e ruído). Serve para
comparar os motores entre si e medir latência; a taxa de erro em fala real só
sai com gravações. Com `--audio-dir`, cada `nome.wav` (mono 16 kHz PCM16) tem
ao lado um `nome.txt` com a frase dita e, opcionalmente, um `nome.expect` com o
desfecho esperado em YAML (`[sit]`, `stop`, `reject` ou `ignored`).

Por motor:

    carga_s        tempo para carregar o modelo
    ram_mb         pico de memória do processo (modelo + execução)
    lat_ms         latência por frase (média, p95)
    parada_ms      latência média das frases que são só parada
    wer            erro de palavras contra a frase dita (sem acento e pontuação)
    acerto         frases em que o desfecho final (comandos, parada, rejeição) é o certo
    fp             frases em que o servidor executaria um comando errado ou indevido
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tts import engine_name, synthesize  # noqa: E402

from app.audio import SAMPLE_RATE, decode_wav  # noqa: E402
from app.catalog import load_catalog  # noqa: E402
from app.config import Settings  # noqa: E402
from app.interpret.evaluate import REJECT, STOP, outcome  # noqa: E402
from app.interpret.interpreter import Interpreter, Thresholds  # noqa: E402
from app.interpret.text import edit_distance, fold, tokenize, words  # noqa: E402

BENCH = ROOT / "bench"
NON_SPEECH = {
    "silencio": ["synth", "2.0", "sine", "0", "vol", "0"],
    "ruido_branco": ["synth", "2.0", "whitenoise", "vol", "0.05"],
    "ruido_marrom": ["synth", "3.0", "brownnoise", "vol", "0.3"],
}


def synthetic_cases(data: Path) -> list[dict]:
    """Gera (uma vez) o áudio de cada frase rotulada e dos trechos sem fala."""
    folder = BENCH / f"audio-{engine_name()}"
    folder.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, case in enumerate(yaml.safe_load(data.read_text(encoding="utf-8"))):
        path = folder / f"{index:03d}.wav"
        if not path.exists():
            path.write_bytes(synthesize(case["text"], index))
        cases.append({"path": str(path), "text": case["text"], "expect": case["expect"]})
    for name, effect in NON_SPEECH.items():
        path = folder / f"{name}.wav"
        if not path.exists():
            subprocess.run(
                ["sox", "-n", "-r", "16000", "-c", "1", "-b", "16", str(path), *effect], check=True
            )
        cases.append({"path": str(path), "text": "", "expect": REJECT})
    return cases


def recorded_cases(folder: Path) -> list[dict]:
    cases = []
    for path in sorted(folder.glob("*.wav")):
        text = path.with_suffix(".txt")
        expect = path.with_suffix(".expect")
        cases.append(
            {
                "path": str(path),
                "text": text.read_text(encoding="utf-8").strip() if text.exists() else None,
                "expect": yaml.safe_load(expect.read_text(encoding="utf-8"))
                if expect.exists()
                else None,
            }
        )
    return cases


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Erros de palavra e tamanho da referência, sem acento nem pontuação."""
    ref = [fold(word) for word in words(tokenize(reference))]
    hyp = [fold(word) for word in words(tokenize(hypothesis))]
    return edit_distance(ref, hyp, limit=len(ref) + len(hyp)), len(ref)


def load_engine(spec: str, settings: Settings, threads: int):
    kind, _, name = spec.partition(":")
    if kind == "whisper":
        from app.transcribe.whisper import WhisperTranscriber

        return WhisperTranscriber(
            model=name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=threads,
            num_workers=1,
            beam_size=settings.whisper_beam_size,
            no_speech_threshold=settings.no_speech_threshold,
            min_avg_logprob=settings.min_avg_logprob,
            max_compression_ratio=settings.max_compression_ratio,
            hallucinations=settings.hallucinations,
        )
    if kind == "vosk":
        from app.transcribe.vosk import VoskTranscriber

        return VoskTranscriber(Path(name) if name else settings.vosk_model_path)
    raise SystemExit(f"motor desconhecido: {spec}")


def run_engine(spec: str, cases: list[dict], threads: int) -> dict:
    settings = Settings()
    started = time.perf_counter()
    engine = load_engine(spec, settings, threads)
    load_s = time.perf_counter() - started
    engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # aquecimento

    # O desfecho é medido só com o caminho lexical: assim a comparação é dos
    # motores de transcrição, sem o modelo de embeddings no meio.
    interpreter = Interpreter(load_catalog(settings.catalog_path), None, Thresholds())
    latencies, stop_latencies, audio_s = [], [], 0.0
    errors = reference_words = hits = judged = false_positives = 0
    rows = []
    for case in cases:
        audio = decode_wav(Path(case["path"]).read_bytes())
        started = time.perf_counter()
        transcription = engine.transcribe(audio)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        audio_s += len(audio) / SAMPLE_RATE

        # Mesma regra do serviço: transcrição descartada pelo motor não é
        # interpretada, a menos que seja só parada (parar é sempre seguro).
        got, _ = outcome(interpreter, transcription.text, settings.wake_words)
        if got != STOP and (transcription.reject_reason or not transcription.text.strip()):
            got = REJECT
        if case["text"]:
            wrong, total = word_errors(case["text"], transcription.text)
            errors += wrong
            reference_words += total
        if case["expect"] is not None:
            judged += 1
            hits += got == case["expect"]
            false_positives += isinstance(got, list) and got != case["expect"]
            if case["expect"] == STOP:
                stop_latencies.append(elapsed_ms)
        rows.append(
            {
                "file": Path(case["path"]).name,
                "said": case["text"],
                "heard": transcription.text,
                "stt_reject": transcription.reject_reason,
                "expect": case["expect"],
                "got": got,
                "ms": round(elapsed_ms),
            }
        )
    return {
        "engine": spec,
        "threads": threads,
        "load_s": round(load_s, 1),
        # ru_maxrss vem em KB no Linux.
        "ram_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "lat_mean_ms": round(float(np.mean(latencies))),
        "lat_p95_ms": round(float(np.percentile(latencies, 95))),
        "stop_ms": round(float(np.mean(stop_latencies))) if stop_latencies else None,
        "rtf": round(sum(latencies) / 1000 / audio_s, 3),
        "wer": round(errors / reference_words, 3) if reference_words else None,
        "accuracy": round(hits / judged, 3) if judged else None,
        "false_positives": false_positives,
        "cases": len(cases),
        "rows": rows,
    }


def main() -> int:
    settings = Settings()
    default_engines = ["whisper:base", "whisper:small", "whisper:medium"]
    if settings.vosk_model_path.is_dir():
        default_engines.append(f"vosk:{settings.vosk_model_path}")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--engines", default=",".join(default_engines))
    parser.add_argument("--engine", help="roda só este motor e imprime JSON (uso interno)")
    parser.add_argument("--audio-dir", type=Path, help="pasta com nome.wav + nome.txt")
    parser.add_argument("--data", type=Path, default=ROOT / "eval" / "phrases.yaml")
    parser.add_argument("--threads", type=int, default=settings.whisper_cpu_threads)
    parser.add_argument("--errors", action="store_true", help="lista as frases com desfecho errado")
    args = parser.parse_args()

    cases = recorded_cases(args.audio_dir) if args.audio_dir else synthetic_cases(args.data)
    if args.engine:
        print(json.dumps(run_engine(args.engine, cases, args.threads), ensure_ascii=False))
        return 0

    kind = "fala real" if args.audio_dir else f"fala SINTÉTICA, {engine_name()}"
    print(f"{len(cases)} áudios ({kind}), {args.threads} threads por transcrição\n")
    header = (
        f"{'motor':<28} {'carga_s':>7} {'ram_mb':>7} {'lat_ms':>7} {'p95_ms':>7} "
        f"{'parada_ms':>9} {'wer':>6} {'acerto':>7} {'fp':>3}"
    )
    print(header)
    results = []
    for spec in args.engines.split(","):
        # Um processo por motor: o pico de memória de um não contamina o outro.
        command = [sys.executable, __file__, "--engine", spec, "--threads", str(args.threads)]
        command += ["--data", str(args.data)]
        if args.audio_dir:
            command += ["--audio-dir", str(args.audio_dir)]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            print(f"{spec:<28} FALHOU: {done.stderr.strip().splitlines()[-1][:120]}")
            continue
        result = json.loads(done.stdout.strip().splitlines()[-1])
        results.append(result)
        name = spec if len(spec) <= 28 else spec[:12] + "…" + spec[-15:]
        print(
            f"{name:<28} {result['load_s']:>7} {result['ram_mb']:>7} {result['lat_mean_ms']:>7} "
            f"{result['lat_p95_ms']:>7} {result['stop_ms'] or '-':>9} {result['wer'] or '-':>6} "
            f"{result['accuracy'] or '-':>7} {result['false_positives']:>3}"
        )
        if args.errors:
            for row in result["rows"]:
                if row["expect"] is not None and row["got"] != row["expect"]:
                    print(f"    {row['said']!r} → ouviu {row['heard']!r}: veio {row['got']}")
    BENCH.mkdir(exist_ok=True)
    out = BENCH / "stt_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndetalhe por frase em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
