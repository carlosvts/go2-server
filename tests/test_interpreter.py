"""Interpretador: texto → comandos do catálogo, parada ou rejeição."""

import pytest

from app.interpret.evaluate import IGNORED, REJECT, STOP, outcome
from app.interpret.interpreter import Interpreter, Kind, Thresholds
from tests.conftest import StubEmbedder

WAKE_WORDS = ["hey jarvis", "ei jarvis", "rei jarvis", "e jarvis", "jarvis"]


@pytest.fixture
def stub() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def interpreter(catalog, stub) -> Interpreter:
    return Interpreter(catalog, stub, Thresholds(accept=0.8, margin=0.1))


def run(interpreter: Interpreter, text: str):
    return outcome(interpreter, text, WAKE_WORDS)


# ─── Positivos ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("senta", ["sit"]),
        ("Senta!", ["sit"]),
        ("levanta", ["stand_up"]),
        ("deita", ["stand_down"]),
        ("cumprimente", ["hello"]),
        ("alonga", ["stretch"]),
        ("coração", ["finger_heart"]),
        ("coracao", ["finger_heart"]),  # sem acento
        ("por favor senta", ["sit"]),
        ("robô, senta agora", ["sit"]),
        ("desligar motores", ["damp"]),
    ],
)
def test_single_action(interpreter, text, expected):
    commands, result = run(interpreter, text)
    assert commands == expected
    assert result.confidence == 1.0
    assert result.clauses[0].method == "lexical"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("andar para frente", "move_forward"),
        ("andar para trás", "move_backward"),
        ("andar para a direita", "strafe_right"),
        ("andar para esquerda", "strafe_left"),
        ("virar para a direita", "turn_right"),
        ("virar para esquerda", "turn_left"),
        ("anda pra frente", "move_forward"),  # "pra" e conjugação
        ("gira pra direita", "turn_right"),  # sinônimo do verbo
        ("vá para trás", "move_backward"),
        ("por favor anda um pouco para frente", "move_forward"),
        ("andar para a esqerda", "strafe_left"),  # erro de uma letra
    ],
)
def test_movement_by_slots(interpreter, text, expected):
    commands, result = run(interpreter, text)
    assert commands == [expected]
    assert result.clauses[0].method == "slots"


def test_movement_verbs_with_the_same_direction_do_not_mix(interpreter, catalog):
    """O verbo decide o comando; a direção igual não pode confundir os dois."""
    for direction in ("direita", "esquerda"):
        walk, _ = run(interpreter, f"andar para a {direction}")
        turn, _ = run(interpreter, f"virar para a {direction}")
        assert walk != turn
        walk_body = catalog.step(walk[0]).call.body
        turn_body = catalog.step(turn[0]).call.body
        assert walk_body["vy"] != 0 and walk_body["vyaw"] == 0
        assert turn_body["vyaw"] != 0 and turn_body["vy"] == 0
    # No referencial do robô, esquerda é positivo.
    assert catalog.step("turn_left").call.body["vyaw"] > 0
    assert catalog.step("strafe_right").call.body["vy"] < 0


def test_mishearing_of_turn_is_not_a_walk(interpreter):
    """O Whisper ouviu "vira pra esquerda" como "ir para a esquerda"."""
    assert run(interpreter, "ir para a esquerda")[0] == REJECT


def test_compound_sentence(interpreter):
    assert run(interpreter, "senta e depois levanta")[0] == ["sit", "stand_up"]
    assert run(interpreter, "Levanta, anda para frente e vira para a esquerda.")[0] == [
        "stand_up",
        "move_forward",
        "turn_left",
    ]
    assert run(interpreter, "senta, em seguida deita")[0] == ["sit", "stand_down"]


# ─── Negativos ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text", ["não senta", "nunca levanta", "senta e não deita", "anda para frente sem parar"]
)
def test_negation_rejects_the_whole_sentence(interpreter, text):
    commands, result = run(interpreter, text)
    assert commands == REJECT
    assert result.reject_reason == "negation"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("anda", "movement_missing_direction"),
        ("para frente", "movement_missing_verb"),
        ("frente", "movement_missing_verb"),
        ("virar para frente", "movement_invalid_combination"),
        ("anda vira direita", "movement_ambiguous"),
        ("pula para frente", "movement_unknown_word"),
        ("qual é a previsão do tempo", "low_similarity"),
        ("toca uma música", "low_similarity"),
    ],
)
def test_rejections(interpreter, text, reason):
    commands, result = run(interpreter, text)
    assert commands == REJECT
    assert result.reject_reason == reason


def test_one_bad_clause_rejects_everything(interpreter):
    """Nada de execução parcial."""
    commands, result = run(interpreter, "senta e toca uma música")
    assert commands == REJECT
    assert result.commands == []
    assert result.clauses[0].command_id == "sit"


def test_too_many_clauses(interpreter):
    commands, result = run(interpreter, "senta, levanta, deita e alonga")
    assert commands == REJECT
    assert result.reject_reason == "too_many_clauses"


def test_disabled_command_is_rejected(interpreter):
    commands, result = run(interpreter, "dança")
    assert commands == REJECT
    assert result.reject_reason == "command_disabled"


def test_accented_verb_is_not_the_connective(interpreter):
    """ "é" não é o conectivo "e": a frase não pode ser cortada ali."""
    _, result = run(interpreter, "qual é a previsão do tempo")
    assert len(result.clauses) == 1


# ─── Parada ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["para", "Pare!", "parar", "stop", "para para", "pare agora"])
def test_stop_only(interpreter, text):
    assert interpreter.is_stop_only(text)
    assert run(interpreter, text)[0] == STOP


@pytest.mark.parametrize("text", ["senta e para", "anda para", "pode parar"])
def test_stop_inside_a_sentence_still_stops(interpreter, text):
    """Parar é sempre seguro: cláusula de parada ou frase terminada em parada."""
    assert run(interpreter, text)[0] == STOP


def test_preposition_is_not_a_stop(interpreter):
    assert not interpreter.is_stop_only("andar para frente")
    assert run(interpreter, "andar para frente")[0] == ["move_forward"]
    # Sem o verbo sobra "para frente": não é parada nem movimento.
    assert run(interpreter, "para frente")[0] == REJECT


# ─── Wake word ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "hey jarvis senta",
        "Hey, Jarvis. Senta!",
        "Ei Jarvis, senta",
        "rei jarves senta",
        "Jarvis senta",
    ],
)
def test_wake_word_at_the_start_is_removed(interpreter, text):
    assert run(interpreter, text)[0] == ["sit"]


@pytest.mark.parametrize("text", ["hey jarvis", "Hey, Jarvis.", "ei jarvis", "", "..."])
def test_only_the_wake_word_is_ignored(interpreter, text):
    assert run(interpreter, text)[0] == IGNORED


def test_wake_word_then_stop(interpreter):
    assert run(interpreter, "hey jarvis para")[0] == STOP


def test_wake_word_only_at_the_start(interpreter):
    assert run(interpreter, "senta jarvis")[0] == REJECT


# ─── Embeddings ────────────────────────────────────────────────────────────


def test_embedding_accepts_a_clear_neighbour(interpreter, stub):
    stub.mix("fique sentado", {"senta": 0.95, "deita": 0.2})
    commands, result = run(interpreter, "fique sentado")
    assert commands == ["sit"]
    clause = result.clauses[0]
    assert clause.method == "embedding"
    assert clause.score >= 0.8 and clause.margin >= 0.1
    assert result.confidence == pytest.approx(clause.score)


def test_embedding_needs_the_similarity(interpreter, stub):
    stub.mix("sentar no chão", {"senta": 0.6, "fora": 0.8})
    _, result = run(interpreter, "sentar no chão")
    assert result.reject_reason == "low_similarity"


def test_embedding_needs_the_margin(catalog, stub):
    """Cosseno alto não basta: o segundo comando tem de ficar longe."""
    interpreter = Interpreter(catalog, stub, Thresholds(accept=0.6, margin=0.1))
    stub.mix("abaixa", {"senta": 0.72, "deita": 0.69})
    _, result = run(interpreter, "abaixa")
    assert result.kind is Kind.REJECTED
    assert result.reject_reason == "ambiguous"
    assert result.clauses[0].score >= 0.6


def test_margin_is_between_commands_not_examples(catalog, stub):
    """Dois exemplos do mesmo comando não são ambiguidade."""
    interpreter = Interpreter(catalog, stub, Thresholds(accept=0.6, margin=0.1))
    stub.mix("sentadinho", {"senta": 0.7, "sentar": 0.7})
    assert run(interpreter, "sentadinho")[0] == ["sit"]


def test_lexical_only_command_is_never_reached_by_embedding(interpreter, stub):
    """`damp` derruba o robô: só por frase exata."""
    stub.mix("corta a energia", {"desligar motores": 1.0})
    commands, result = run(interpreter, "corta a energia")
    assert commands == REJECT
    assert result.reject_reason == "lexical_only_command"


def test_disabled_command_works_as_a_distractor(interpreter, stub):
    stub.mix("dançar muito", {"dançar": 1.0})
    _, result = run(interpreter, "dançar muito")
    assert result.reject_reason == "command_disabled"


def test_embedding_near_a_stop_word_stops(interpreter, stub):
    stub.mix("fica parado", {"parar": 0.95, "senta": 0.1})
    assert run(interpreter, "fica parado")[0] == STOP


def test_slot_by_embedding(interpreter, stub):
    """Verbo fora do léxico: o slot fecha por similaridade com o léxico do verbo."""
    stub.mix("marchar", {"andar": 0.95, "virar": 0.2})
    commands, result = run(interpreter, "marchar para frente")
    assert commands == ["move_forward"]
    assert result.clauses[0].method == "slots"
    assert result.clauses[0].score < 1.0


def test_slot_by_embedding_needs_the_margin(interpreter, stub):
    stub.mix("mexer", {"andar": 0.7, "virar": 0.68})
    _, result = run(interpreter, "mexer para a direita")
    assert result.reject_reason == "movement_unknown_word"


def test_movement_like_sentence_still_needs_both_slots(interpreter, stub):
    """Parecer com uma frase de movimento não basta: verbo e direção são obrigatórios."""
    stub.mix("avante marche", {"andar para frente": 1.0})
    _, result = run(interpreter, "avante marche")
    assert result.kind is Kind.REJECTED
    assert result.clauses[0].method == "slots"


def test_without_embedder_only_lexical(catalog):
    interpreter = Interpreter(catalog, None)
    assert run(interpreter, "senta")[0] == ["sit"]
    assert run(interpreter, "fique sentado")[1].reject_reason == "no_match"


def test_output_only_contains_catalog_ids(interpreter, catalog):
    for text in ("senta", "andar para frente", "levanta e vira para a direita", "para"):
        _, result = run(interpreter, text)
        assert result.commands and set(result.commands) <= catalog.ids


def test_stop_by_embedding_has_its_own_looser_threshold(catalog, stub):
    interpreter = Interpreter(catalog, stub, Thresholds(accept=0.95, margin=0.1, stop_accept=0.8))
    stub.mix("fica quieto", {"parar": 0.85, "fora": 0.5})
    stub.mix("fique sentadinho", {"senta": 0.85, "fora": 0.5})
    assert run(interpreter, "fica quieto")[0] == STOP
    assert run(interpreter, "fique sentadinho")[0] == REJECT  # comando segue no limiar alto


def test_stop_like_below_every_threshold_is_rejected(catalog, stub):
    interpreter = Interpreter(catalog, stub, Thresholds(accept=0.6, margin=0.1, stop_accept=0.8))
    stub.mix("quase nada", {"parar": 0.5, "fora": 0.9})
    assert run(interpreter, "quase nada")[1].reject_reason == "low_similarity"
