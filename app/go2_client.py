"""Cliente HTTP da go2-api. Uma tentativa por chamada, sem retry."""

import httpx

from app.catalog import Call

OBSTACLE_AVOIDANCE = "/safety/obstacle-avoidance"


class Go2ApiError(Exception):
    """A go2-api recusou o comando ou não respondeu."""


class Go2Client:
    def __init__(
        self,
        base_url: str,
        timeout_s: float,
        read_timeout_s: float = 4.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport
        )
        # Os GET que esperam o robô demoram até GO2_REQUEST_TIMEOUT_S da API.
        self._read_timeout_s = read_timeout_s

    async def call(self, call: Call) -> None:
        """Envia um comando. Qualquer coisa diferente de 2xx é falha.

        A go2-api responde 202 (aceito e despachado, não "executado"), 422
        (corpo inválido ou acima dos limites dela) ou 503 (sem o robô).
        """
        try:
            response = await self._http.request(call.method, call.endpoint, json=call.body)
        except httpx.HTTPError as error:
            raise Go2ApiError(f"{call.method} {call.endpoint}: sem resposta ({error!r})") from error
        if not response.is_success:
            raise Go2ApiError(
                f"{call.method} {call.endpoint}: HTTP {response.status_code} {response.text[:200]}"
            )

    async def obstacle_avoidance(self) -> bool | None:
        """`enabled` informado pela API; None se ela não soube ou não respondeu."""
        try:
            response = await self._http.get(OBSTACLE_AVOIDANCE, timeout=self._read_timeout_s)
            enabled = response.json().get("enabled") if response.is_success else None
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        return enabled if isinstance(enabled, bool) else None

    async def status(self) -> dict | None:
        """`GET /status` da go2-api, para o /health. None se ela não respondeu."""
        try:
            response = await self._http.get("/status", timeout=1.0)
            return response.json() if response.is_success else None
        except (httpx.HTTPError, ValueError):
            return None

    async def aclose(self) -> None:
        await self._http.aclose()
