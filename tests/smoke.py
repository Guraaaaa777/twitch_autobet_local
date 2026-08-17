"""End-to-end smoke test with a stubbed Twitch API and a stubbed LLM.

Runs the real tracker code paths -- event ingestion, inference, Kelly sizing,
bet placement, settlement, retention -- against fakes, so the logic can be
checked without Twitch credentials or a GGUF model.

    python -m tests.smoke
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="autobet-smoke-"))
os.environ["TWITCH_AUTOBET_DATA"] = str(_TMP)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, log, store  # noqa: E402
from app.betting import OutcomeInput, decide  # noqa: E402
from app.llm.client import _normalise, _parse  # noqa: E402
from app.tracker import ChannelRuntime, Tracker  # noqa: E402
from app.twitch.gql import TwitchError  # noqa: E402
from app.twitch.models import PredictionEvent  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, extra: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {extra}")
        raise AssertionError(f"{name} {extra}")


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_event(status: str = "ACTIVE", *, o1: int = 6000, o2: int = 4000,
               winner: str | None = None, lock_in: float = 30.0) -> dict:
    now = datetime.now(UTC)
    return {
        "id": "event-1",
        "channel_id": "999",
        "title": "次のラウンドは勝てる?",
        "status": status,
        "created_at": iso(now - timedelta(seconds=90)),
        "locked_at": iso(now + timedelta(seconds=lock_in)),
        "prediction_window_seconds": 120,
        "winning_outcome_id": winner,
        "outcomes": [
            {"id": "o1", "title": "勝つ", "color": "BLUE",
             "total_points": o1, "total_users": 30},
            {"id": "o2", "title": "負ける", "color": "PINK",
             "total_points": o2, "total_users": 20},
        ],
    }


class StubTwitch:
    """Stands in for TwitchGQLClient."""

    def __init__(self) -> None:
        self.balance = 50_000
        self.event = make_event()
        self.placed: list[tuple[str, str, int]] = []
        self.reject: str | None = None

    async def fetch_state(self, login: str, count: int = 1):
        return self.balance, [PredictionEvent.parse(self.event)], []

    async def make_prediction(self, event_id: str, outcome_id: str, points: int) -> None:
        if self.reject:
            raise TwitchError(f"Twitch が投票を拒否しました ({self.reject})", code=self.reject)
        self.placed.append((event_id, outcome_id, points))
        self.balance -= points


# -- unit-level checks ------------------------------------------------------


def test_kelly() -> None:
    print("\n[ケリー基準]")
    cfg = config.BettingSettings()
    outs = [OutcomeInput("a", "Yes", 5000), OutcomeInput("b", "No", 5000)]

    d = decide(outs, {"a": 0.5, "b": 0.5}, 10_000, cfg)
    check("エッジなしなら投票しない", not d.should_bet)

    d = decide(outs, {"a": 0.7, "b": 0.3}, 10_000, cfg)
    check("優位側に賭ける", d.should_bet and d.outcome_id == "a", str(d.reason))
    check("最大賭け率 5% を超えない", d.amount <= 500, str(d.amount))
    check("フラクショナルケリーが効いている",
          d.amount < (d.kelly_stake or 0), f"{d.amount} vs {d.kelly_stake}")

    # Self-impact: a lopsided pool must not be treated as fixed odds.
    thin = [OutcomeInput("a", "Yes", 10_000), OutcomeInput("b", "No", 100)]
    d = decide(thin, {"a": 0.5, "b": 0.5}, 10_000, cfg)
    naive = (10_100 - 100) / 100
    check("自分の賭け金でオッズが下がることを織り込む",
          d.should_bet and (d.edge or 0) < 0.5 * (1 + naive) - 1,
          f"edge={d.edge}")

    d = decide(outs, {"a": 0.7, "b": 0.3}, 0, cfg)
    check("残高 0 なら投票しない", not d.should_bet)

    strict = config.BettingSettings(min_edge=0.9)
    d = decide(outs, {"a": 0.6, "b": 0.4}, 10_000, strict)
    check("最低エッジ未満なら見送る", not d.should_bet, str(d.reason))


def test_llm_parsing() -> None:
    print("\n[LLM 応答の解釈]")
    raw = '前置き {"probabilities": [{"choice": 1, "probability": 0.7}, ' \
          '{"choice": 2, "probability": 0.3}], "rationale": "根拠"} 後置き'
    parsed, warnings = _parse(raw, 2)
    check("前後に文字があっても JSON を取り出せる", parsed == {1: 0.7, 2: 0.3}, str(parsed))

    probs, warn2 = _normalise({"o1": 3.0, "o2": 1.0}, ["o1", "o2"])
    check("合計が 1 でなければ正規化する", abs(probs["o1"] - 0.75) < 1e-9, str(probs))
    check("正規化を警告として残す", any("正規化" in w for w in warn2))

    parsed, warnings = _parse('{"probabilities": [{"choice": 1, "probability": 0.9}]}', 2)
    probs, _ = _normalise({"o1": 0.9}, ["o1", "o2"])
    check("欠けた選択肢を警告する", any("2" in w for w in warnings), str(warnings))
    check("欠けた選択肢は 0 として正規化", abs(probs["o1"] - 1.0) < 1e-9, str(probs))

    try:
        _parse("これは JSON ではありません", 2)
        check("非 JSON はエラーになる", False)
    except Exception:
        check("非 JSON はエラーになる", True)


# -- integration ------------------------------------------------------------


async def test_flow() -> None:
    print("\n[追跡〜投票〜決着フロー]")
    log.bind_loop(asyncio.get_running_loop())
    config.update({"betting": {"dry_run": True, "kelly_fraction": 0.25,
                               "max_bet_ratio": 0.05, "min_edge": 0.05}})

    channel_id = store.create_channel("teststreamer", "TestStreamer", "999")
    store.update_channel(channel_id, fixed_probs={"enabled": True, "probs": [0.8, 0.2]})

    tracker = Tracker()
    stub = StubTwitch()
    tracker.client = lambda: stub  # type: ignore[method-assign]
    rt = ChannelRuntime(channel_id=channel_id, login="teststreamer",
                        display_name="TestStreamer", twitch_id="999", balance=50_000)
    tracker._channels = {channel_id: rt}
    tracker._running = True

    # 1. Event appears.
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    prediction = store.prediction_by_event("event-1")
    check("予想が保存される", prediction is not None)
    check("追跡中の予想として保持される", rt.active_event_id == "event-1")
    check("投票データが蓄積される",
          len(db.query("SELECT 1 FROM pool_snapshots")) == 1)
    check("投票情報がログに出る",
          any(r["category"] == "prediction" for r in log.recent(50)))
    check("締め切り前に投票タスクが予約される", "event-1" in tracker._bet_tasks)
    tracker._bet_tasks.pop("event-1").cancel()

    # 2. Bet fires (dry run).
    await tracker._execute_bet(rt, "event-1")
    bets = db.rows_to_dicts(db.query("SELECT * FROM bets"))
    check("投票が 1 件記録される", len(bets) == 1, str(bets))
    check("ドライランでは実際に投票しない", stub.placed == [], str(stub.placed))
    check("ドライランとして記録される", bets[0]["status"] == "dry_run", bets[0]["status"])
    check("優位な選択肢を選ぶ", bets[0]["outcome_id"] == "o1", str(bets[0]))
    check("上限 5% を超えない", bets[0]["amount"] <= 2500, str(bets[0]["amount"]))
    check("固定確率では LLM を呼ばない",
          db.query_one("SELECT source FROM inferences")["source"] == "fixed")
    check("推論結果がログに出る",
          any(r["category"] == "inference" for r in log.recent(50)))
    check("投票結果がログに出る",
          any(r["category"] == "bet" for r in log.recent(50)))
    amount = bets[0]["amount"]

    # 3. Re-running must not double-bet.
    await tracker._execute_bet(rt, "event-1")
    check("同じ予想に二重投票しない",
          len(db.query("SELECT 1 FROM bets")) == 1)

    # 4. Resolution.
    stub.event = make_event("RESOLVED", winner="o1")
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    bet = db.row_to_dict(db.query_one("SELECT * FROM bets"))
    check("的中として決着する", bet["result"] == "win", str(bet))
    expected = int(round(10_000 * amount / 6000))
    check("パリミュチュアル配当で精算する", bet["payout"] == expected,
          f"{bet['payout']} != {expected}")
    check("決着後にポイントを取得する",
          len(db.query("SELECT 1 FROM point_history")) >= 1)

    # 5. Live betting path.
    print("\n[実投票モード]")
    config.update({"betting": {"dry_run": False}})
    db.execute("DELETE FROM bets")
    db.execute("DELETE FROM predictions")
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-2"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-2", asyncio.Future()).cancel()
    await tracker._execute_bet(rt, "event-2")
    check("実投票モードでは Twitch を呼ぶ", len(stub.placed) == 1, str(stub.placed))
    check("送信内容が記録と一致する",
          stub.placed[0][0] == "event-2" and stub.placed[0][1] == "o1",
          str(stub.placed))
    check("placed として記録される",
          db.query_one("SELECT status FROM bets")["status"] == "placed")

    # 6. Twitch rejects the bet.
    db.execute("DELETE FROM bets")
    db.execute("DELETE FROM predictions")
    stub.reject = "NOT_ENOUGH_POINTS"
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-3"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-3", asyncio.Future()).cancel()
    await tracker._execute_bet(rt, "event-3")
    row = db.row_to_dict(db.query_one("SELECT * FROM bets"))
    check("拒否は failed として記録される", row["status"] == "failed", str(row))
    check("拒否理由が残る", "NOT_ENOUGH_POINTS" in (row["error"] or ""), str(row["error"]))
    check("拒否がエラーログに出る",
          any(r["level"] == "ERROR" for r in log.recent(20)))
    stub.reject = None

    # 7. Skip when the edge is too thin.
    print("\n[見送り判定]")
    db.execute("DELETE FROM bets")
    db.execute("DELETE FROM predictions")
    store.update_channel(channel_id, fixed_probs={"enabled": True, "probs": [0.6, 0.4]})
    stub.event = make_event("ACTIVE", o1=6000, o2=4000)
    stub.event["id"] = "event-4"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-4", asyncio.Future()).cancel()
    await tracker._execute_bet(rt, "event-4")
    row = db.row_to_dict(db.query_one("SELECT * FROM bets"))
    check("市場と一致する見立てなら見送る", row["status"] == "skipped", str(row))
    check("見送り理由が残る", bool(row["error"]), str(row))


async def test_retention() -> None:
    print("\n[保持時間]")
    channel = store.get_channel_by_login("teststreamer")
    assert channel
    cid = channel["id"]
    store.add_transcript(cid, "新しい発話")
    db.execute(
        "INSERT INTO transcripts (channel_id, ts, text) VALUES (?, datetime('now','-45 minutes'), ?)",
        (cid, "古い発話"),
    )
    text = store.recent_transcript_text(cid, 30, 10_000)
    check("保持時間内の文字起こしだけ渡す", "新しい発話" in text and "古い発話" not in text, text)

    removed = store.prune_transcripts(30)
    check("保持時間を過ぎた行を削除する", removed == 1, str(removed))
    check("新しい行は残る", len(db.query("SELECT 1 FROM transcripts")) == 1)


def test_api() -> None:
    print("\n[HTTP API]")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        res = client.get("/api/settings")
        check("設定を取得できる", res.status_code == 200 and "llama" in res.json())

        res = client.put("/api/settings", json={"poll_rate_sec": 9.5})
        check("設定を更新できる", res.status_code == 200
              and res.json()["poll_rate_sec"] == 9.5, res.text[:200])

        res = client.put("/api/settings", json={"poll_rate_sec": -1})
        check("不正な設定を拒否する", res.status_code == 400, res.text[:200])

        res = client.get("/api/channels")
        check("チャンネル一覧を取得できる", res.status_code == 200
              and any(c["login"] == "teststreamer" for c in res.json()))

        cid = next(c["id"] for c in res.json() if c["login"] == "teststreamer")
        res = client.patch(f"/api/channels/{cid}", json={"enabled": False})
        check("追跡除外に切り替えられる", res.status_code == 200
              and res.json()["enabled"] == 0, res.text[:200])

        res = client.patch(f"/api/channels/{cid}",
                           json={"fixed_probs": {"enabled": True, "probs": [1.0]}})
        check("固定確率の項目数不足を拒否する", res.status_code == 400, res.text[:200])

        res = client.patch(f"/api/channels/{cid}", json={"manual_info": "テスト情報"})
        check("手動情報を保存できる",
              res.status_code == 200 and res.json()["manual_info"] == "テスト情報")

        res = client.get(f"/api/channels/{cid}/points")
        check("ポイント推移を取得できる", res.status_code == 200 and "points" in res.json())

        res = client.get(f"/api/channels/{cid}/predictions")
        check("予想履歴を取得できる", res.status_code == 200)

        res = client.get("/api/logs?limit=10")
        check("ログを取得できる", res.status_code == 200 and res.json()["logs"])

        res = client.get("/api/status")
        check("状態を取得できる", res.status_code == 200 and res.json()["running"] is False)

        res = client.post("/api/tracking/start")
        check("認証未設定では追跡を開始できない", res.status_code == 400, res.text[:200])

        for path in ("/", "/settings", "/static/app.js", "/static/style.css",
                     "/static/common.js", "/static/settings.js"):
            check(f"{path} を配信できる", client.get(path).status_code == 200)

        res = client.delete(f"/api/channels/{cid}")
        check("登録解除できる", res.status_code == 200)
        check("蓄積データも削除される",
              db.query_one("SELECT 1 FROM predictions WHERE channel_id = ?", (cid,)) is None
              and db.query_one("SELECT 1 FROM transcripts WHERE channel_id = ?", (cid,)) is None)


def main() -> int:
    print(f"data dir: {_TMP}")
    db.connect()
    try:
        test_kelly()
        test_llm_parsing()
        asyncio.run(test_flow())
        asyncio.run(test_retention())
        test_api()
    except AssertionError as exc:
        print(f"\n失敗: {exc}")
        return 1
    finally:
        db.connect().close()
        shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n{len(PASSED)} 件すべて成功しました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
