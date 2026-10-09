"""O desfecho de uma frase em texto, como o serviço o decide depois de transcrever.

Usado pela calibração e pelos testes, para medir exatamente o caminho de
produção: remoção da wake word e interpretação.
"""

from app.interpret.interpreter import Interpretation, Interpreter, Kind
from app.wake_word import strip_wake_word

STOP = "stop"
REJECT = "reject"
IGNORED = "ignored"


def outcome(
    interpreter: Interpreter, text: str, wake_words: list[str]
) -> tuple[str | list[str], Interpretation]:
    """`STOP`, `REJECT`, `IGNORED` ou a lista de IDs de comando."""
    stripped, _ = strip_wake_word(text, wake_words)
    result = interpreter.interpret(stripped)
    if result.kind is Kind.EMPTY:
        return IGNORED, result
    if result.kind is Kind.STOP:
        return STOP, result
    if result.kind is Kind.REJECTED:
        return REJECT, result
    return result.commands, result
