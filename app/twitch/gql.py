"""Client for Twitch's private GraphQL API (gql.twitch.tv).

This is the API the Twitch web client itself uses. It is undocumented and
unsupported: field names can change without notice, and automating it is at odds
with Twitch's Terms of Service. It is used here because the public Helix API has
no endpoint for reading another channel's predictions, reading your own
channel-point balance, or placing a prediction as a viewer.

Because the schema is undocumented, every operation is sent as part of a batch
so one broken query cannot take the others down with it, and `diagnose()` runs
each operation individually and reports the raw GraphQL error so a schema drift
is visible in the UI instead of silently disabling the app.
"""

from __future__ import annotations

import secrets
from typing import Any

import httpx

from ..config import TwitchSettings
from .models import ChannelInfo, PredictionEvent

GQL_URL = "https://gql.twitch.tv/gql"

Q_CHANNEL = """
query AutobetChannel($login: String!) {
  user(login: $login) {
    id
    login
    displayName
    stream { id }
  }
}
"""

Q_POINTS = """
query AutobetPoints($login: String!) {
  user(login: $login) {
    id
    channel {
      id
      self { communityPoints { balance } }
    }
  }
}
"""

Q_PREDICTIONS = """
query AutobetPredictions($login: String!, $count: Int!) {
  user(login: $login) {
    id
    channel {
      id
      predictions(first: $count) {
        edges {
          node {
            id
            title
            status
            createdAt
            endedAt
            lockedAt
            predictionWindowSeconds
            winningOutcome { id }
            outcomes {
              id
              title
              color
              totalPoints
              totalUsers
            }
          }
        }
      }
    }
  }
}
"""

M_MAKE_PREDICTION = """
mutation AutobetMakePrediction($input: MakePredictionInput!) {
  makePrediction(input: $input) {
    error { code }
  }
}
"""


class TwitchError(RuntimeError):
    """A GraphQL-level or transport-level failure talking to Twitch."""

    def __init__(self, message: str, *, code: str | None = None,
                 detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


class TwitchGQLClient:
    def __init__(self, settings: TwitchSettings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.settings.oauth_token:
            raise TwitchError("Twitch OAuth トークンが未設定です", code="NO_TOKEN")
        token = self.settings.oauth_token.strip()
        for prefix in ("OAuth ", "oauth:", "Bearer "):
            if token.lower().startswith(prefix.lower()):
                token = token[len(prefix):].strip()
        headers = {
            "Client-Id": self.settings.client_id,
            "Authorization": f"OAuth {token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Origin": "https://www.twitch.tv",
            "Referer": "https://www.twitch.tv/",
        }
        if self.settings.device_id:
            headers["X-Device-Id"] = self.settings.device_id
        return headers

    async def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.settings.request_timeout_sec,
                headers={"User-Agent": "Mozilla/5.0 twitch_autobet"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # -- transport ---------------------------------------------------------

    async def _post(self, ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """POST a batch of operations. Returns one result envelope per op."""
        client = await self.client()
        try:
            resp = await client.post(GQL_URL, json=ops, headers=self._headers())
        except httpx.HTTPError as exc:
            raise TwitchError(f"Twitch への接続に失敗しました: {exc}",
                              code="TRANSPORT") from exc

        if resp.status_code == 401:
            raise TwitchError(
                "Twitch の認証に失敗しました (401)。OAuth トークンを再設定してください",
                code="UNAUTHORIZED",
            )
        if resp.status_code == 429:
            raise TwitchError("Twitch にレート制限されました (429)", code="RATE_LIMIT")
        if resp.status_code >= 400:
            raise TwitchError(
                f"Twitch が HTTP {resp.status_code} を返しました",
                code="HTTP_ERROR",
                detail=resp.text[:1000],
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise TwitchError("Twitch の応答が JSON ではありません", code="BAD_JSON",
                              detail=resp.text[:1000]) from exc

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise TwitchError("Twitch の応答形式が想定外です", code="BAD_SHAPE",
                              detail=str(payload)[:1000])
        return payload

    @staticmethod
    def _op(query: str, variables: dict[str, Any], name: str) -> dict[str, Any]:
        return {"operationName": name, "query": query, "variables": variables}

    @staticmethod
    def _unwrap(envelope: dict[str, Any], what: str) -> dict[str, Any]:
        errors = envelope.get("errors")
        if errors:
            message = "; ".join(
                str(e.get("message", e)) for e in errors if isinstance(e, dict)
            ) or str(errors)
            raise TwitchError(f"{what} の取得に失敗しました: {message}",
                              code="GRAPHQL", detail=errors)
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise TwitchError(f"{what} の応答が空でした", code="EMPTY")
        return data

    # -- operations --------------------------------------------------------

    async def resolve_channel(self, login: str) -> ChannelInfo:
        login = login.strip().lower()
        results = await self._post(
            [self._op(Q_CHANNEL, {"login": login}, "AutobetChannel")]
        )
        data = self._unwrap(results[0], "チャンネル情報")
        user = data.get("user")
        if not user:
            raise TwitchError(f"チャンネル '{login}' が見つかりません", code="NOT_FOUND")
        return ChannelInfo(
            twitch_id=str(user["id"]),
            login=str(user.get("login") or login),
            display_name=str(user.get("displayName") or login),
            is_live=bool(user.get("stream")),
        )

    async def fetch_state(
        self, login: str, *, count: int = 1
    ) -> tuple[int | None, list[PredictionEvent], list[TwitchError]]:
        """One round trip for balance + recent predictions.

        Returns `(balance, events, soft_errors)`. Soft errors are per-operation
        GraphQL failures: the caller keeps whichever half succeeded.
        """
        login = login.strip().lower()
        results = await self._post(
            [
                self._op(Q_POINTS, {"login": login}, "AutobetPoints"),
                self._op(
                    Q_PREDICTIONS, {"login": login, "count": count}, "AutobetPredictions"
                ),
            ]
        )
        # A batch of N should come back as N envelopes, but be defensive.
        while len(results) < 2:
            results.append({"errors": [{"message": "応答が欠落しています"}]})

        soft: list[TwitchError] = []

        balance: int | None = None
        try:
            data = self._unwrap(results[0], "チャンネルポイント残高")
            node = ((data.get("user") or {}).get("channel") or {}).get("self") or {}
            points = node.get("communityPoints") or {}
            if points.get("balance") is not None:
                balance = int(points["balance"])
        except TwitchError as exc:
            soft.append(exc)

        events: list[PredictionEvent] = []
        try:
            data = self._unwrap(results[1], "予想イベント")
            channel = (data.get("user") or {}).get("channel") or {}
            channel_id = str(channel.get("id") or "")
            edges = ((channel.get("predictions") or {}).get("edges")) or []
            for edge in edges:
                node = (edge or {}).get("node")
                if not node:
                    continue
                node = dict(node)
                node.setdefault("channel_id", channel_id)
                events.append(PredictionEvent.parse(node))
        except TwitchError as exc:
            soft.append(exc)

        return balance, events, soft

    async def make_prediction(
        self, event_id: str, outcome_id: str, points: int
    ) -> None:
        """Place a bet. Raises TwitchError if Twitch rejects it."""
        variables = {
            "input": {
                "eventID": event_id,
                "outcomeID": outcome_id,
                "points": int(points),
                "transactionID": secrets.token_hex(16),
            }
        }
        results = await self._post(
            [self._op(M_MAKE_PREDICTION, variables, "AutobetMakePrediction")]
        )
        data = self._unwrap(results[0], "投票")
        payload = data.get("makePrediction") or {}
        err = payload.get("error")
        if err:
            code = str((err or {}).get("code") or "UNKNOWN")
            raise TwitchError(
                f"Twitch が投票を拒否しました ({code})", code=code, detail=err
            )

    async def diagnose(self, login: str) -> list[dict[str, Any]]:
        """Run each operation on its own and report pass/fail with raw errors.

        Exists because the private schema is unversioned: when Twitch renames a
        field this is what tells you which query broke and why.
        """
        checks: list[dict[str, Any]] = []

        async def run(name: str, coro) -> None:
            try:
                value = await coro
                checks.append({"name": name, "ok": True, "detail": value})
            except TwitchError as exc:
                checks.append(
                    {
                        "name": name,
                        "ok": False,
                        "error": str(exc),
                        "code": exc.code,
                        "detail": exc.detail,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics must not raise
                checks.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

        async def channel_check() -> Any:
            info = await self.resolve_channel(login)
            return {"twitch_id": info.twitch_id, "display_name": info.display_name,
                    "is_live": info.is_live}

        async def points_check() -> Any:
            results = await self._post(
                [self._op(Q_POINTS, {"login": login.lower()}, "AutobetPoints")]
            )
            data = self._unwrap(results[0], "チャンネルポイント残高")
            node = ((data.get("user") or {}).get("channel") or {}).get("self") or {}
            balance = (node.get("communityPoints") or {}).get("balance")
            if balance is None:
                raise TwitchError(
                    "残高フィールドが空でした。そのチャンネルでポイントを獲得していないか、"
                    "スキーマが変更された可能性があります",
                    code="EMPTY_BALANCE",
                )
            return {"balance": int(balance)}

        async def predictions_check() -> Any:
            results = await self._post(
                [self._op(Q_PREDICTIONS, {"login": login.lower(), "count": 5},
                          "AutobetPredictions")]
            )
            data = self._unwrap(results[0], "予想イベント")
            channel = (data.get("user") or {}).get("channel") or {}
            edges = ((channel.get("predictions") or {}).get("edges")) or []
            return {"recent_predictions": len(edges),
                    "titles": [((e or {}).get("node") or {}).get("title") for e in edges[:5]]}

        await run("チャンネル解決", channel_check())
        await run("チャンネルポイント残高", points_check())
        await run("予想イベント取得", predictions_check())
        return checks
