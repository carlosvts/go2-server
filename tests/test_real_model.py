"""O conjunto rotulado, com o modelo de embeddings real e os limiares padrão.

Pulado se o modelo não estiver no cache local (não baixa nada).
"""

from pathlib import Path

import pytest
import yaml

from app.config import Settings
from app.interpret.evaluate import outcome
from app.interpret.interpreter import Interpreter, Thresholds

pytestmark = pytest.mark.model
CASES = yaml.safe_load(
    (Path(__file__).resolve().parent.parent / "eval" / "phrases.yaml").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def results(catalog, monkeypatch_module):
    settings = Settings(_env_file=None)
    monkeypatch_module.setenv("HF_HUB_OFFLINE", "1")
    try:
        from app.interpret.embed import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(settings.embed_model, settings.embed_threads)
    except Exception as error:  # modelo fora do cache
        pytest.skip(f"modelo de embeddings indisponível: {error}")
    thresholds = Thresholds(
        settings.tau_accept,
        settings.tau_margin,
        settings.tau_slot_accept,
        settings.tau_slot_margin,
        settings.max_clauses,
        settings.tau_stop_accept,
    )
    interpreter = Interpreter(catalog, embedder, thresholds)
    return [(case, outcome(interpreter, case["text"], settings.wake_words)[0]) for case in CASES]


@pytest.fixture(scope="module")
def monkeypatch_module():
    patcher = pytest.MonkeyPatch()
    yield patcher
    patcher.undo()


def test_no_false_positive_in_the_labeled_set(results):
    """Nenhuma frase pode virar um comando que não era o esperado."""
    wrong = [(c["text"], got) for c, got in results if isinstance(got, list) and got != c["expect"]]
    assert wrong == []


def test_every_stop_stops(results):
    missed = [c["text"] for c, got in results if c["expect"] == "stop" and got != "stop"]
    assert missed == []


def test_nothing_stops_by_mistake(results):
    wrong = [c["text"] for c, got in results if got == "stop" and c["expect"] != "stop"]
    assert wrong == []


def test_recall_does_not_regress(results):
    positives = [(c, got) for c, got in results if isinstance(c["expect"], list)]
    hits = sum(got == c["expect"] for c, got in positives)
    assert hits / len(positives) >= 0.75
