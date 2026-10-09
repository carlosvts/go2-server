"""Remoção da wake word do início da transcrição.

No modo thin a TV Box manda o áudio com a wake word, e o motor de transcrição
a escreve de vários jeitos ("Hey Jarvis", "Ei, Jarvis", "Rei Jarvis"). A
comparação é sem acento e sem pontuação, com tolerância a erro pequeno.
"""

import re

from app.interpret.text import edit_distance, fold

_WORD = re.compile(r"\w+")


def _tolerance(variant: str) -> int:
    if len(variant) <= 3:
        return 0
    return 1 if len(variant) <= 7 else 2


def strip_wake_word(text: str, variants: list[str]) -> tuple[str, bool]:
    """Devolve o texto sem a wake word inicial e se ela estava lá.

    Inofensivo quando a wake word não está presente: o texto volta igual.
    """
    spans = list(_WORD.finditer(text))
    heard = [fold(match.group().lower()) for match in spans]
    candidates = sorted(
        (fold(variant.lower()).split() for variant in variants), key=len, reverse=True
    )
    for variant in candidates:
        size = len(variant)
        if not variant or len(heard) < size:
            continue
        target = " ".join(variant)
        limit = _tolerance(target)
        if edit_distance(" ".join(heard[:size]), target, limit) <= limit:
            rest = text[spans[size - 1].end() :]
            return rest.lstrip(" \t\n,.;:!?-"), True
    return text, False
