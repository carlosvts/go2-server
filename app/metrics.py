"""Resumo por `reason`, para comparar os modos edge e thin lado a lado."""

from collections import Counter, deque

KNOWN_REASONS = ("unk", "low_conf", "too_long", "wake_word")
# `reason` vem do cliente: valores desconhecidos são aceitos, mas não podem
# criar chaves sem limite aqui.
MAX_OTHER_REASONS = 20


def _summary(values: deque[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "count": len(ordered),
        "mean_ms": round(sum(ordered) / len(ordered), 1),
        "p95_ms": round(p95, 1),
    }


class Stats:
    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._statuses: dict[str, Counter[str]] = {}
        self._latency: dict[str, deque[float]] = {}
        # A parada falada tem latência própria: é o caminho que mais importa.
        self._stop_latency: dict[str, deque[float]] = {}

    def _key(self, reason: str) -> str:
        if reason in KNOWN_REASONS or reason in self._statuses:
            return reason
        return reason[:32] if len(self._statuses) < MAX_OTHER_REASONS else "other"

    def record(self, reason: str, status: str, total_ms: float) -> None:
        key = self._key(reason)
        self._statuses.setdefault(key, Counter())[status] += 1
        self._latency.setdefault(key, deque(maxlen=self._window)).append(total_ms)
        if status == "stopped":
            self._stop_latency.setdefault(key, deque(maxlen=self._window)).append(total_ms)

    def summary(self) -> dict:
        return {
            reason: {
                "total": sum(statuses.values()),
                "by_status": dict(statuses),
                "latency": _summary(self._latency[reason]),
                "stop_latency": _summary(self._stop_latency.get(reason, deque())),
            }
            for reason, statuses in sorted(self._statuses.items())
        }
