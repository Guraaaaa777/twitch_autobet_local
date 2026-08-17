"""Stake sizing for Twitch channel-point predictions.

Twitch predictions are pari-mutuel, not fixed-odds: the losing pool is split
across the winning side in proportion to each better's stake. So our own stake
dilutes the payout we are betting for, and the usual `f* = (pb - q) / b` Kelly
formula -- which assumes `b` is a constant -- overstakes.

With `T_i` points already on outcome `i`, a total pool of `T`, and a stake `b`:

    payout(b) = (T + b) * b / (T_i + b)
    profit(b) = payout(b) - b = b * (T - T_i) / (T_i + b)

so the net odds `B(b) = (T - T_i) / (T_i + b)` shrink as `b` grows. We therefore
maximise expected log wealth numerically rather than in closed form:

    g(b) = p * ln(W + profit(b)) + (1 - p) * ln(W - b)

`g` is concave on `[0, W)` (a concave increasing `b/(T_i+b)` inside a log, plus a
concave `ln(W - b)`), so a ternary search finds the optimum reliably.

Caveat worth knowing: the pool keeps moving after we bet. `T` and `T_i` are the
values at the moment we submit, and late money will shift the true payout. This
is unknowable at decision time, which is one more reason to bet a fraction of
Kelly rather than the full amount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import BettingSettings


@dataclass(frozen=True)
class OutcomeInput:
    outcome_id: str
    title: str
    total_points: int


@dataclass
class Candidate:
    outcome_id: str
    title: str
    probability: float
    market_probability: float
    total_points: int
    full_kelly_stake: float
    stake: int
    edge: float
    net_odds: float
    log_growth: float
    expected_profit: float


@dataclass
class Decision:
    should_bet: bool
    reason: str
    outcome_id: str | None = None
    outcome_title: str | None = None
    amount: int = 0
    probability: float | None = None
    edge: float | None = None
    kelly_stake: float | None = None
    bankroll: int = 0
    candidates: list[Candidate] = field(default_factory=list)


def profit(stake: float, total_pool: float, outcome_points: float) -> float:
    """Net profit if `stake` on an outcome holding `outcome_points` wins."""
    denom = outcome_points + stake
    if denom <= 0:
        return 0.0
    return stake * (total_pool - outcome_points) / denom


def net_odds(stake: float, total_pool: float, outcome_points: float) -> float:
    """Profit per point staked."""
    denom = outcome_points + stake
    if denom <= 0:
        return 0.0
    return (total_pool - outcome_points) / denom


def _log_growth(stake: float, p: float, bankroll: float, total_pool: float,
                outcome_points: float) -> float:
    if stake <= 0:
        return math.log(bankroll)
    if stake >= bankroll:
        return -math.inf
    win_wealth = bankroll + profit(stake, total_pool, outcome_points)
    lose_wealth = bankroll - stake
    if win_wealth <= 0 or lose_wealth <= 0:
        return -math.inf
    return p * math.log(win_wealth) + (1.0 - p) * math.log(lose_wealth)


def optimal_stake(p: float, bankroll: float, total_pool: float,
                  outcome_points: float, *, iterations: int = 120) -> float:
    """Full-Kelly stake, found by ternary search on the concave log-growth."""
    lo, hi = 0.0, max(0.0, bankroll * (1.0 - 1e-9))
    if hi <= 0 or total_pool <= outcome_points:
        return 0.0
    for _ in range(iterations):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _log_growth(m1, p, bankroll, total_pool, outcome_points) < _log_growth(
            m2, p, bankroll, total_pool, outcome_points
        ):
            lo = m1
        else:
            hi = m2
    best = (lo + hi) / 2.0
    # A stake is only worth taking if it beats standing pat.
    if _log_growth(best, p, bankroll, total_pool, outcome_points) <= math.log(bankroll):
        return 0.0
    return best


def outcome_metrics(outcomes: list[OutcomeInput]) -> tuple[int, dict[str, float]]:
    """Total pool and the market-implied probability of each outcome."""
    total = sum(max(0, o.total_points) for o in outcomes)
    if total <= 0:
        implied = {o.outcome_id: 1.0 / len(outcomes) for o in outcomes} if outcomes else {}
    else:
        implied = {o.outcome_id: max(0, o.total_points) / total for o in outcomes}
    return total, implied


def decide(
    outcomes: list[OutcomeInput],
    probabilities: dict[str, float],
    bankroll: int,
    cfg: BettingSettings,
) -> Decision:
    """Pick an outcome and a stake, or explain why we are standing down."""
    if not outcomes:
        return Decision(False, "選択肢がありません", bankroll=bankroll)
    if bankroll <= 0:
        return Decision(False, "チャンネルポイント残高が 0 です", bankroll=bankroll)

    total_pool, implied = outcome_metrics(outcomes)
    if total_pool <= 0:
        return Decision(
            False,
            "プールが空のためオッズを計算できません",
            bankroll=bankroll,
        )

    hard_cap = min(
        float(bankroll),
        float(cfg.max_bet_points),
        float(bankroll) * cfg.max_bet_ratio,
    )

    candidates: list[Candidate] = []
    for o in outcomes:
        p = float(probabilities.get(o.outcome_id, 0.0))
        p = min(max(p, 0.0), 1.0)
        pts = max(0, o.total_points)

        full = optimal_stake(p, float(bankroll), float(total_pool), float(pts))
        stake = int(min(full * cfg.kelly_fraction, hard_cap))
        stake = max(0, min(stake, bankroll))

        b = net_odds(float(stake), float(total_pool), float(pts))
        candidates.append(
            Candidate(
                outcome_id=o.outcome_id,
                title=o.title,
                probability=p,
                market_probability=implied.get(o.outcome_id, 0.0),
                total_points=pts,
                full_kelly_stake=full,
                stake=stake,
                edge=p * (1.0 + b) - 1.0,
                net_odds=b,
                log_growth=_log_growth(
                    float(stake), p, float(bankroll), float(total_pool), float(pts)
                ),
                expected_profit=p * profit(float(stake), float(total_pool), float(pts))
                - (1.0 - p) * stake,
            )
        )

    candidates.sort(key=lambda c: c.log_growth, reverse=True)
    best = candidates[0]

    if best.stake <= 0 or best.full_kelly_stake <= 0:
        return Decision(
            False,
            "ケリー基準の最適賭け金が 0 でした(優位性なし)",
            bankroll=bankroll,
            candidates=candidates,
        )
    if best.edge < cfg.min_edge:
        return Decision(
            False,
            f"エッジ {best.edge:.1%} が閾値 {cfg.min_edge:.1%} 未満です",
            bankroll=bankroll,
            candidates=candidates,
        )
    if best.probability < cfg.min_confidence:
        return Decision(
            False,
            f"推定確率 {best.probability:.1%} が閾値 {cfg.min_confidence:.1%} 未満です",
            bankroll=bankroll,
            candidates=candidates,
        )
    if best.stake < cfg.min_bet_points:
        return Decision(
            False,
            f"算出賭け金 {best.stake} が最低額 {cfg.min_bet_points} 未満です",
            bankroll=bankroll,
            candidates=candidates,
        )

    return Decision(
        should_bet=True,
        reason=(
            f"エッジ {best.edge:.1%} / 推定 {best.probability:.1%} "
            f"vs 市場 {best.market_probability:.1%}"
        ),
        outcome_id=best.outcome_id,
        outcome_title=best.title,
        amount=best.stake,
        probability=best.probability,
        edge=best.edge,
        kelly_stake=best.full_kelly_stake,
        bankroll=bankroll,
        candidates=candidates,
    )
