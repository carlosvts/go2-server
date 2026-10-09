import pytest

from app.wake_word import strip_wake_word

VARIANTS = ["hey jarvis", "ei jarvis", "rei jarvis", "e jarvis", "jarvis"]


@pytest.mark.parametrize(
    ("text", "rest"),
    [
        ("hey jarvis senta", "senta"),
        ("Hey, Jarvis. Senta!", "Senta!"),
        ("Ei Jarvis, anda para frente", "anda para frente"),
        ("Rei Jarvis levanta", "levanta"),
        ("hey jarves senta", "senta"),  # erro pequeno
        ("Jarvis, para", "para"),
        ("E Jarvis senta", "senta"),
        ("hey jarvis", ""),
        ("Hey Jarvis!", ""),
    ],
)
def test_removed_from_the_start(text, rest):
    assert strip_wake_word(text, VARIANTS) == (rest, True)


@pytest.mark.parametrize(
    "text", ["senta", "anda para frente", "senta jarvis", "e senta", "para", "", "hey senta"]
)
def test_harmless_without_the_wake_word(text):
    assert strip_wake_word(text, VARIANTS) == (text, False)


def test_removed_only_once():
    assert strip_wake_word("jarvis jarvis senta", VARIANTS) == ("jarvis senta", True)
