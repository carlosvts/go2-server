"""Texto transcrito → comandos do catálogo, ou parada, ou rejeição.

Discriminativo: a saída é sempre uma seleção do catálogo. Por frase:

1. normaliza;
2. frase que é só parada ⇒ PARADA (antes de tudo, sem embedding);
3. negação em qualquer parte ⇒ rejeita a frase inteira;
4. segmenta em cláusulas; uma cláusula que é só parada, ou uma frase que
   termina em palavra de parada ⇒ PARADA (parar é sempre seguro);
5. resolve cada cláusula: frase exata → slots de movimento → vizinho mais
   próximo por embedding;
6. uma cláusula rejeitada rejeita a frase inteira.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.catalog import Catalog
from app.interpret.embed import Embedder
from app.interpret.text import COMMA, edit_distance, fold, remove_sequences, words

# Classes que existem só no índice de embeddings, para dar destino certo ao que
# se parece com parada ou com movimento em vez de cair na ação mais próxima.
STOP_CLASS = "__stop__"
MOVE_CLASS = "__movement__"
# Palavra de slot com erro de uma letra só vale a partir deste tamanho.
FUZZY_MIN_LENGTH = 5


class Kind(Enum):
    COMMANDS = "commands"
    STOP = "stop"
    REJECTED = "rejected"
    EMPTY = "empty"


@dataclass(frozen=True)
class Thresholds:
    accept: float = 0.95  # τ_accept: cosseno mínimo do vizinho mais próximo
    margin: float = 0.10  # τ_margin: folga mínima sobre o segundo comando
    slot_accept: float = 0.95  # idem, para uma palavra de verbo ou direção
    slot_margin: float = 0.10
    max_clauses: int = 3
    stop_accept: float = 0.80  # cosseno mínimo para o que se parece com parada


@dataclass
class Clause:
    text: str
    command_id: str | None = None
    method: str = ""  # "lexical", "slots" ou "embedding"
    score: float | None = None  # cosseno (1.0 quando lexical)
    margin: float | None = None  # top1 − top2
    reject_reason: str | None = None

    def log(self) -> dict:
        return {key: value for key, value in vars(self).items() if value is not None}


@dataclass
class Interpretation:
    kind: Kind
    normalized: str
    commands: list[str] = field(default_factory=list)
    confidence: float | None = None  # o menor score entre as cláusulas
    reject_reason: str | None = None
    clauses: list[Clause] = field(default_factory=list)


@dataclass(frozen=True)
class _Slot:
    kind: str  # "verb" ou "direction"
    name: str  # o canônico
    score: float
    margin: float | None = None


class Interpreter:
    def __init__(
        self, catalog: Catalog, embedder: Embedder | None, thresholds: Thresholds | None = None
    ) -> None:
        self._catalog = catalog
        self._embedder = embedder
        self._t = thresholds or Thresholds()

        # Frase exata (sem acento) → comando.
        self._exact: dict[str, str] = {}
        for command in catalog.commands.values():
            for example in command.examples:
                self._exact[self._key(catalog.tokenize(example))] = command.id

        # Palavra (sem acento) → slot.
        self._slot_words: dict[str, _Slot] = {}
        for kind, lexicon in (("verb", catalog.verbs), ("direction", catalog.directions)):
            for name, lexicon_words in lexicon.items():
                for word in lexicon_words:
                    self._slot_words[fold(word)] = _Slot(kind, name, 1.0)

        # Índices de embeddings, calculados uma vez só.
        self._intent_labels: list[str] = []
        self._intent_matrix = np.zeros((0, 1), dtype=np.float32)
        self._slot_labels: list[_Slot] = []
        self._slot_matrix = np.zeros((0, 1), dtype=np.float32)
        if embedder is not None:
            self._build_indexes(embedder)

    def _build_indexes(self, embedder: Embedder) -> None:
        catalog = self._catalog
        texts: list[str] = []
        for command in catalog.commands.values():
            for example in command.examples:
                texts.append(" ".join(words(catalog.tokenize(example))))
                self._intent_labels.append(command.id)
        for word in sorted(catalog.stop_words):
            texts.append(word)
            self._intent_labels.append(STOP_CLASS)
        for rule in catalog.rules.values():
            for verb in catalog.verbs[rule.verb]:
                texts.append(f"{verb} para {rule.direction}")
                self._intent_labels.append(MOVE_CLASS)
        self._intent_matrix = embedder.embed(texts)

        slot_texts: list[str] = []
        for kind, lexicon in (("verb", catalog.verbs), ("direction", catalog.directions)):
            for name, lexicon_words in lexicon.items():
                for word in lexicon_words:
                    slot_texts.append(word)
                    self._slot_labels.append(_Slot(kind, name, 1.0))
        self._slot_matrix = embedder.embed(slot_texts)

    # ─── Parada ────────────────────────────────────────────────────────────

    def is_stop_only(self, text: str) -> bool:
        """A frase é só parada. Barato: não usa embedding nem espera nada."""
        return self._all_stop(self._clean(self._catalog.tokenize(text)))

    def _all_stop(self, clause: list[str]) -> bool:
        return bool(clause) and all(fold(word) in self._catalog.stop_words for word in clause)

    # ─── Frase ─────────────────────────────────────────────────────────────

    def interpret(self, text: str) -> Interpretation:
        catalog = self._catalog
        tokens = catalog.tokenize(text)
        normalized = " ".join(tokens)
        if not words(tokens):
            return Interpretation(Kind.EMPTY, normalized, reject_reason="empty")

        spoken = self._clean(tokens)
        if self._all_stop(spoken):
            return Interpretation(Kind.STOP, normalized, [catalog.stop_id], confidence=1.0)
        if any(fold(word) in catalog.negation for word in spoken):
            return Interpretation(Kind.REJECTED, normalized, reject_reason="negation")

        clauses = self._segment(tokens)
        # "Anda... para!" chega como "anda para": preposição não termina frase.
        ends_in_stop = bool(spoken) and fold(spoken[-1]) in catalog.stop_words
        if ends_in_stop or any(self._all_stop(clause) for clause in clauses):
            return Interpretation(Kind.STOP, normalized, [catalog.stop_id], confidence=1.0)
        if not clauses:
            return Interpretation(Kind.REJECTED, normalized, reject_reason="no_match")
        if len(clauses) > self._t.max_clauses:
            return Interpretation(Kind.REJECTED, normalized, reject_reason="too_many_clauses")

        results = [self._resolve(clause) for clause in clauses]
        if any(result.command_id == catalog.stop_id for result in results):
            return Interpretation(
                Kind.STOP,
                normalized,
                [catalog.stop_id],
                clauses=results,
                confidence=min(r.score for r in results if r.command_id == catalog.stop_id),
            )
        rejected = next((result for result in results if result.reject_reason), None)
        if rejected is not None:
            return Interpretation(
                Kind.REJECTED, normalized, reject_reason=rejected.reject_reason, clauses=results
            )
        return Interpretation(
            Kind.COMMANDS,
            normalized,
            [result.command_id for result in results],
            confidence=min(result.score for result in results),
            clauses=results,
        )

    def _clean(self, tokens: list[str]) -> list[str]:
        return remove_sequences(words(tokens), list(self._catalog.fillers))

    def _segment(self, tokens: list[str]) -> list[list[str]]:
        """Corta nos conectivos e nas pausas; devolve as cláusulas sem enchimento."""
        catalog = self._catalog
        folded = [fold(token) for token in tokens]
        marked = list(tokens)
        for phrase in catalog.separator_phrases:
            for start in range(len(tokens) - len(phrase) + 1):
                if tuple(folded[start : start + len(phrase)]) == phrase:
                    marked[start : start + len(phrase)] = [COMMA] * len(phrase)
        clauses: list[list[str]] = [[]]
        for token in marked:
            if token == COMMA or token in catalog.separators:
                clauses.append([])
            else:
                clauses[-1].append(token)
        cleaned = (remove_sequences(clause, list(catalog.fillers)) for clause in clauses)
        return [clause for clause in cleaned if clause]

    # ─── Cláusula ──────────────────────────────────────────────────────────

    @staticmethod
    def _key(tokens: list[str]) -> str:
        return " ".join(fold(word) for word in words(tokens))

    def _resolve(self, clause: list[str]) -> Clause:
        result = Clause(text=" ".join(clause))

        exact = self._exact.get(self._key(clause))
        if exact is not None:
            return self._accept_command(result, exact, "lexical", 1.0, None)

        core = [word for word in clause if fold(word) not in self._catalog.move_ignore]
        if any(self._slot_lexical(word) for word in core):
            return self._resolve_movement(result, core)

        if self._embedder is None:
            result.reject_reason = "no_match"
            return result
        label, score, margin = self._nearest(
            self._embedder.embed([result.text])[0], self._intent_matrix, self._intent_labels
        )
        result.method, result.score, result.margin = "embedding", score, margin
        # O que se parece com parada tem limiar próprio, mais frouxo: parar
        # por engano é seguro.
        accept = self._t.accept
        if label == STOP_CLASS:
            accept = min(accept, self._t.stop_accept)
        if score < accept:
            result.reject_reason = "low_similarity"
        elif margin < self._t.margin:
            result.reject_reason = "ambiguous"
        elif label == STOP_CLASS:
            result.command_id = self._catalog.stop_id
        elif label == MOVE_CLASS:
            # Parece movimento e nenhuma palavra bateu no léxico: os dois
            # slots ainda têm de fechar, cada um por si.
            return self._resolve_movement(result, core)
        else:
            return self._accept_command(result, label, "embedding", score, margin)
        return result

    def _accept_command(
        self, result: Clause, command_id: str, method: str, score: float, margin: float | None
    ) -> Clause:
        command = self._catalog.commands[command_id]
        result.method, result.score, result.margin = method, score, margin
        if not command.enabled:
            result.reject_reason = "command_disabled"
        elif command.match == "lexical_only" and method != "lexical":
            result.reject_reason = "lexical_only_command"
        else:
            result.command_id = command_id
        return result

    def _resolve_movement(self, result: Clause, core: list[str]) -> Clause:
        """Verbo e direção, cada um resolvido sozinho. Os dois são obrigatórios."""
        result.method = "slots"
        slots: list[_Slot] = []
        for word in core:
            slot = self._slot_lexical(word) or self._slot_embedding(word)
            if slot is None:
                result.reject_reason = "movement_unknown_word"
                return result
            slots.append(slot)
        verbs = [slot for slot in slots if slot.kind == "verb"]
        directions = [slot for slot in slots if slot.kind == "direction"]
        if not verbs:
            result.reject_reason = "movement_missing_verb"
        elif not directions:
            result.reject_reason = "movement_missing_direction"
        elif len(verbs) > 1 or len(directions) > 1:
            result.reject_reason = "movement_ambiguous"
        else:
            rule = self._catalog.rules.get((verbs[0].name, directions[0].name))
            if rule is None:
                result.reject_reason = "movement_invalid_combination"
            else:
                result.command_id = rule.id
        if slots:
            result.score = min(slot.score for slot in slots)
        margins = [slot.margin for slot in slots if slot.margin is not None]
        result.margin = min(margins) if margins else None
        return result

    def _slot_lexical(self, word: str) -> _Slot | None:
        key = fold(word)
        if key in self._slot_words:
            return self._slot_words[key]
        if len(key) < FUZZY_MIN_LENGTH:
            return None
        near = {
            (slot.kind, slot.name)
            for known, slot in self._slot_words.items()
            if len(known) >= FUZZY_MIN_LENGTH and edit_distance(key, known, limit=1) <= 1
        }
        if len(near) != 1:
            return None
        kind, name = near.pop()
        return _Slot(kind, name, 1.0)

    def _slot_embedding(self, word: str) -> _Slot | None:
        if self._embedder is None:
            return None
        labels = [(slot.kind, slot.name) for slot in self._slot_labels]
        (kind, name), score, margin = self._nearest(
            self._embedder.embed([word])[0], self._slot_matrix, labels
        )
        if score < self._t.slot_accept or margin < self._t.slot_margin:
            return None
        return _Slot(kind, name, score, margin)

    @staticmethod
    def _nearest(vector: np.ndarray, matrix: np.ndarray, labels: list) -> tuple:
        """Rótulo mais próximo, seu cosseno e a folga sobre o melhor rótulo diferente."""
        similarities = matrix @ vector
        best: dict = {}
        for label, similarity in zip(labels, similarities, strict=True):
            best[label] = max(best.get(label, -1.0), float(similarity))
        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        return ranked[0][0], ranked[0][1], ranked[0][1] - runner_up
