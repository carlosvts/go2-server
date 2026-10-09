"""Normalização de texto: a base de toda comparação lexical."""

import unicodedata
from collections.abc import Mapping

COMMA = ","
# Pontuação que separa cláusulas. O Whisper escreve "Senta. Depois levanta."
_BREAKS = ",;.!?:"


def fold(text: str) -> str:
    """Sem acentos. Usado só para comparar; os embeddings veem o texto acentuado."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def tokenize(text: str, replacements: Mapping[str, str] | None = None) -> list[str]:
    """Minúsculas, sem pontuação, com `COMMA` no lugar de cada pausa escrita.

    `replacements` troca palavras inteiras ("pra" → "para"); o valor pode ter
    mais de uma palavra ("pro" → "para o").
    """
    chars = []
    for char in text.lower():
        if char in _BREAKS:
            chars.append(f" {COMMA} ")
        elif char.isalnum():
            chars.append(char)
        else:
            chars.append(" ")
    tokens: list[str] = []
    for token in "".join(chars).split():
        replaced = (replacements or {}).get(fold(token))
        tokens.extend(replaced.split() if replaced is not None else [token])
    return tokens


def words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token != COMMA]


def remove_sequences(tokens: list[str], sequences: list[tuple[str, ...]]) -> list[str]:
    """Tira de `tokens` cada ocorrência das sequências (já sem acento), a mais longa primeiro."""
    folded = [fold(token) for token in tokens]
    keep = [True] * len(tokens)
    for sequence in sorted(sequences, key=len, reverse=True):
        size = len(sequence)
        for start in range(len(tokens) - size + 1):
            window = range(start, start + size)
            if all(keep[i] for i in window) and tuple(folded[start : start + size]) == sequence:
                for i in window:
                    keep[i] = False
    return [token for token, kept in zip(tokens, keep, strict=True) if kept]


def edit_distance(a: str, b: str, limit: int = 3) -> int:
    """Distância de Levenshtein, com saída antecipada acima de `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]
