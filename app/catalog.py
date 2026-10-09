"""Catálogo de comandos: o único vocabulário que o servidor pode executar.

Lido de `catalog.yaml`, que é gerado por `scripts/build_catalog.py`. Um catálogo
inconsistente derruba o serviço na partida, com todos os problemas listados.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.interpret.text import fold, tokenize


class CatalogError(Exception):
    """`catalog.yaml` inválido. A mensagem lista todos os problemas."""


@dataclass(frozen=True)
class Call:
    method: str
    endpoint: str
    body: dict[str, Any] | None = None


@dataclass(frozen=True)
class Command:
    """Ação de verbo único (postura ou gesto)."""

    id: str
    enabled: bool
    match: str  # "embedding" ou "lexical_only"
    call: Call
    max_duration_s: float
    examples: tuple[str, ...]


@dataclass(frozen=True)
class MoveRule:
    """Uma combinação verbo × direção e o `move` que ela vira."""

    id: str
    verb: str
    direction: str
    call: Call
    max_duration_s: float  # duração do move + folga


@dataclass(frozen=True)
class Step:
    """O que a fila precisa saber de um comando do catálogo."""

    id: str
    call: Call
    wait_s: float
    is_move: bool


@dataclass(frozen=True)
class Catalog:
    stop_id: str
    stop_call: Call
    stop_words: frozenset[str]  # sem acento
    commands: dict[str, Command]
    verbs: dict[str, tuple[str, ...]]  # verbo canônico → palavras aceitas
    directions: dict[str, tuple[str, ...]]
    rules: dict[tuple[str, str], MoveRule]
    move_ignore: frozenset[str]  # sem acento
    prepare: Step | None  # postura enviada antes de um movimento
    negation: frozenset[str]  # sem acento
    separators: frozenset[str]  # de uma palavra, como escritos e sem acento
    separator_phrases: tuple[tuple[str, ...], ...]  # de várias palavras
    replacements: dict[str, str]
    fillers: tuple[tuple[str, ...], ...]

    @property
    def ids(self) -> frozenset[str]:
        """Todo ID que pode aparecer numa resposta ou na fila."""
        moves = {rule.id for rule in self.rules.values()}
        return frozenset({self.stop_id, *self.commands, *moves})

    def step(self, command_id: str) -> Step:
        """O passo de fila de um comando. `KeyError` para ID fora do catálogo."""
        if command_id in self.commands:
            command = self.commands[command_id]
            return Step(command.id, command.call, command.max_duration_s, is_move=False)
        for rule in self.rules.values():
            if rule.id == command_id:
                return Step(rule.id, rule.call, rule.max_duration_s, is_move=True)
        raise KeyError(command_id)

    def tokenize(self, text: str) -> list[str]:
        return tokenize(text, self.replacements)


def _call(raw: dict[str, Any]) -> Call:
    return Call(raw["method"], raw["endpoint"], raw.get("body"))


def load_catalog(path: Path) -> Catalog:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    language = data["language"]
    replacements = {fold(key): value for key, value in language["replacements"].items()}

    def phrase(text: str) -> tuple[str, ...]:
        return tuple(fold(token) for token in tokenize(text, replacements))

    commands: dict[str, Command] = {}
    for name, raw in data["commands"].items():
        if raw["match"] not in ("embedding", "lexical_only"):
            problems.append(f"{name}: match {raw['match']!r} desconhecido")
        if not raw["examples"]:
            problems.append(f"{name}: sem exemplos")
        if raw["max_duration_s"] <= 0:
            problems.append(f"{name}: max_duration_s tem de ser positivo")
        commands[name] = Command(
            id=name,
            enabled=bool(raw["enabled"]),
            match=raw["match"],
            call=_call(raw["call"]),
            max_duration_s=float(raw["max_duration_s"]),
            examples=tuple(raw["examples"]),
        )

    movement = data["movement"]
    verbs = {verb: tuple(words) for verb, words in movement["verbs"].items()}
    directions = {name: tuple(words) for name, words in movement["directions"].items()}
    rules: dict[tuple[str, str], MoveRule] = {}
    seen = {data["stop"]["id"], *commands}
    for raw in movement["rules"]:
        key = (raw["verb"], raw["direction"])
        if raw["verb"] not in verbs or raw["direction"] not in directions:
            problems.append(f"regra {raw['id']}: verbo ou direção fora dos léxicos")
        if raw["id"] in seen:
            problems.append(f"id repetido: {raw['id']}")
        seen.add(raw["id"])
        if sorted(raw["args"]) != sorted(movement["params"]):
            problems.append(f"regra {raw['id']}: args diferentes de {movement['params']}")
            continue
        rules[key] = MoveRule(
            id=raw["id"],
            verb=raw["verb"],
            direction=raw["direction"],
            call=Call(movement["call"]["method"], movement["call"]["endpoint"], dict(raw["args"])),
            max_duration_s=float(raw["args"]["duration_s"]) + float(movement["settle_s"]),
        )

    # Uma palavra não pode servir a dois slots: a resolução ficaria ambígua.
    owners: dict[str, str] = {}
    for kind, lexicon in (("verbo", verbs), ("direção", directions)):
        for name, lexicon_words in lexicon.items():
            for word in lexicon_words:
                previous = owners.setdefault(fold(word), f"{kind} {name}")
                if previous != f"{kind} {name}":
                    problems.append(f"palavra {word!r} em {previous} e em {kind} {name}")

    prepare = None
    if movement.get("prepare"):
        name = movement["prepare"]["command"]
        if name not in commands:
            problems.append(f"prepare: comando {name!r} fora do catálogo")
        else:
            wait_s = float(movement["prepare"]["wait_s"])
            prepare = Step(name, commands[name].call, wait_s, is_move=False)

    separators = [phrase(text) for text in language["separators"]]
    # Conectivo de uma palavra vale como escrito ou sem acento ("então",
    # "entao"), mas não com acento a mais: "é" não é o conectivo "e".
    single_separators: set[str] = set()
    for text in language["separators"]:
        raw = tokenize(text, replacements)
        if len(raw) == 1:
            single_separators |= {raw[0], fold(raw[0])}
    if problems:
        raise CatalogError(f"{path} inválido:\n" + "\n".join(f"  - {p}" for p in problems))
    return Catalog(
        stop_id=data["stop"]["id"],
        stop_call=_call(data["stop"]["call"]),
        stop_words=frozenset(fold(word) for word in data["stop"]["words"]),
        commands=commands,
        verbs=verbs,
        directions=directions,
        rules=rules,
        move_ignore=frozenset(fold(word) for word in movement["ignore"]),
        prepare=prepare,
        negation=frozenset(fold(word) for word in language["negation"]),
        separators=frozenset(single_separators),
        separator_phrases=tuple(s for s in separators if len(s) > 1),
        replacements=replacements,
        fillers=tuple(phrase(text) for text in language["fillers"]),
    )
