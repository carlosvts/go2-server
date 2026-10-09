"""Precisão e revocação do interpretador por limiar, num conjunto rotulado.

    uv run python scripts/calibrate.py
    uv run python scripts/calibrate.py --models \
        sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,intfloat/multilingual-e5-small
    uv run python scripts/calibrate.py --show 0.85,0.10     # os erros num limiar

A prioridade é PRECISÃO: um falso positivo move um robô físico.

    acerto          a saída é exatamente os comandos esperados
    falso positivo  a saída tem comandos e não era para ter, ou são os errados
    precisão        acertos / frases em que o servidor executaria algo
    revocação       acertos / frases que eram comandos

Parada indevida (o servidor para quando não devia) é contada à parte: é um
erro, mas não move o robô. Os limiares só afetam o caminho de embeddings; a
linha "só lexical" mostra o que já passa sem modelo nenhum.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.catalog import load_catalog  # noqa: E402
from app.config import Settings  # noqa: E402
from app.interpret.evaluate import STOP, outcome  # noqa: E402
from app.interpret.interpreter import Interpreter, Thresholds  # noqa: E402

ACCEPTS = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
MARGINS = (0.00, 0.05, 0.10, 0.15, 0.20)


@dataclass
class Score:
    hits: int = 0
    false_positives: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    wrong_stops: list[str] = field(default_factory=list)
    stop_hits: int = 0
    stops: int = 0
    positives: int = 0

    @property
    def precision(self) -> float:
        acted = self.hits + len(self.false_positives)
        return self.hits / acted if acted else 1.0

    @property
    def recall(self) -> float:
        return self.hits / self.positives if self.positives else 1.0


def evaluate(interpreter: Interpreter, cases: list[dict], wake_words: list[str]) -> Score:
    score = Score()
    for case in cases:
        expected, text = case["expect"], case["text"]
        got, result = outcome(interpreter, text, wake_words)
        detail = f"{text!r}: esperado {expected}, veio {got}"
        if result.reject_reason:
            detail += f" ({result.reject_reason})"
        if isinstance(expected, list):
            score.positives += 1
        if expected == STOP:
            score.stops += 1
            score.stop_hits += got == STOP
        if isinstance(got, list):
            if got == expected:
                score.hits += 1
            else:
                score.false_positives.append(detail)
        elif isinstance(expected, list):
            score.misses.append(detail)
        if got == STOP and expected != STOP:
            score.wrong_stops.append(detail)
    return score


def main() -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ROOT / "eval" / "phrases.yaml")
    parser.add_argument("--models", default=settings.embed_model, help="separados por vírgula")
    parser.add_argument("--show", help="accept,margin: lista os erros neste limiar")
    args = parser.parse_args()

    cases = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    catalog = load_catalog(settings.catalog_path)
    positives = sum(isinstance(case["expect"], list) for case in cases)
    print(f"{len(cases)} frases ({positives} comandos, {len(cases) - positives} negativas/parada)")

    baseline = evaluate(Interpreter(catalog, None), cases, settings.wake_words)
    print(
        f"\nsó lexical (sem embeddings): precisão {baseline.precision:.3f}  "
        f"revocação {baseline.recall:.3f}  falsos positivos {len(baseline.false_positives)}"
    )
    for line in baseline.false_positives:
        print(f"  FALSO POSITIVO {line}")

    from app.interpret.embed import SentenceTransformerEmbedder

    for model in args.models.split(","):
        started = time.perf_counter()
        embedder = SentenceTransformerEmbedder(model, settings.embed_threads)
        load_s = time.perf_counter() - started
        started = time.perf_counter()
        embedder.embed([case["text"] for case in cases])
        embed_ms = (time.perf_counter() - started) * 1000 / len(cases)
        print(f"\n=== {model}  (carga {load_s:.1f}s, {embed_ms:.1f} ms por frase)")
        print(
            "accept margin | precisão revocação | acertos  FP  perdidas  paradas  parada indevida"
        )
        best: tuple[float, float, float, float] | None = None
        for accept in ACCEPTS:
            for margin in MARGINS:
                thresholds = Thresholds(accept, margin, accept, margin, settings.max_clauses)
                score = evaluate(
                    Interpreter(catalog, embedder, thresholds), cases, settings.wake_words
                )
                fps, lost = len(score.false_positives), len(score.misses)
                print(
                    f"  {accept:.2f}   {margin:.2f} |   {score.precision:.3f}     "
                    f"{score.recall:.3f} |   {score.hits:3d}   {fps:2d}     {lost:3d}"
                    f"    {score.stop_hits}/{score.stops}      {len(score.wrong_stops)}"
                )
                # No empate, fica o limiar mais exigente.
                rank = (score.precision, score.recall, margin, accept)
                if best is None or rank >= best:
                    best = rank
        print(
            f"melhor (precisão primeiro, depois revocação): accept={best[3]:.2f} "
            f"margin={best[2]:.2f}  precisão {best[0]:.3f}  revocação {best[1]:.3f}"
        )
        if args.show:
            accept, margin = (float(value) for value in args.show.split(","))
            thresholds = Thresholds(accept, margin, accept, margin, settings.max_clauses)
            score = evaluate(Interpreter(catalog, embedder, thresholds), cases, settings.wake_words)
            print(f"\nerros com accept={accept} margin={margin}:")
            for name, lines in (
                ("FALSO POSITIVO", score.false_positives),
                ("PARADA INDEVIDA", score.wrong_stops),
                ("PERDIDA", score.misses),
            ):
                for line in lines:
                    print(f"  {name} {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
