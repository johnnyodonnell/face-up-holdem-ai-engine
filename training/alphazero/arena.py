"""Arena: head-to-head promotion gate for new self-play challengers.

A "match" is `ARENA_HANDS_PER_MATCH` hands where exactly one seat plays the
challenger network and the other seven play the current champion. The
challenger seat is rotated across all `NUM_SEATS` positions (one per arena
match) so the gate is insensitive to positional advantage — relative
dealer-button distance averages out across the 8 matches.

Action selection in arena uses **argmax over the policy head**, with
illegal actions masked. This matches v1's production engine pattern (no
search at play time) and is the most relevant proxy for the deployed bot's
strength.

Promotion: average challenger BB/100 across all arena hands. If it exceeds
`ARENA_BB100_THRESHOLD`, the challenger is promoted to best.pt.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import config
from alphazero.network import PolicyValueNet, infer
from games.holdem import (
    ACTION_SIZE,
    BIG_BLIND,
    NUM_SEATS,
    STARTING_STACK,
    _action_for_idx,
    apply_action_idx,
    create_game,
    current_seat,
    encode,
    is_terminal,
    legal_action_mask,
    start_new_hand,
)


def _argmax_legal(policy_logits: list[float], legal_mask: list[bool]) -> int:
    """Pick the legal action with the largest logit. Ties broken by lowest idx."""
    best_idx = -1
    best_logit = -math.inf
    for i in range(ACTION_SIZE):
        if not legal_mask[i]:
            continue
        if policy_logits[i] > best_logit:
            best_logit = policy_logits[i]
            best_idx = i
    return best_idx if best_idx >= 0 else 0


def _net_argmax_action(net: PolicyValueNet, state: dict) -> int:
    mover = current_seat(state)
    encoded = encode(state, mover)
    logits, _ = infer(net, encoded)
    mask = legal_action_mask(state)
    return _argmax_legal(logits, mask)


def _play_one_hand(
    challenger: PolicyValueNet,
    champion: PolicyValueNet,
    challenger_seat: int,
    rng: random.Random,
) -> int:
    """Play one hand. Return the challenger's chip delta (signed, in chips)."""
    state = create_game(rng)
    start_chips = state["players"][challenger_seat]["chips"]
    state = start_new_hand(state, rng=rng)
    safety = 0
    while not is_terminal(state):
        actor = current_seat(state)
        if actor < 0:
            break
        net = challenger if actor == challenger_seat else champion
        action_idx = _net_argmax_action(net, state)
        # Action might be illegal-after-masking-falls-through; fall back to
        # check/call then fold.
        action = _action_for_idx(state, action_idx)
        if action is None:
            for fallback in (1, 0):  # check/call, then fold
                action = _action_for_idx(state, fallback)
                if action is not None:
                    break
        state = apply_action_idx(state, action_idx)
        safety += 1
        if safety > 500:
            raise RuntimeError("arena hand did not terminate")
    return state["players"][challenger_seat]["chips"] - start_chips


@dataclass
class ArenaResult:
    matches: int
    hands: int
    bb100: float
    bb100_ci95: float
    challenger_promoted: bool


def play_series(
    challenger: PolicyValueNet,
    champion: PolicyValueNet,
    num_matches: int = config.ARENA_MATCHES,
    hands_per_match: int = config.ARENA_HANDS_PER_MATCH,
    promotion_threshold: float = config.ARENA_BB100_THRESHOLD,
    rng: Optional[random.Random] = None,
) -> ArenaResult:
    """Run a rotated-seat arena and return the result.

    The challenger seat advances by 1 each match: match 0 challenger=seat 0,
    match 1 challenger=seat 1, ..., match 7 challenger=seat 7. If
    `num_matches > NUM_SEATS`, the rotation wraps around.
    """
    rng = rng or random.Random()
    challenger.eval()
    champion.eval()

    deltas = []
    total_hands = 0
    for m in range(num_matches):
        challenger_seat = m % NUM_SEATS
        for _ in range(hands_per_match):
            delta = _play_one_hand(challenger, champion, challenger_seat, rng)
            deltas.append(delta)
            total_hands += 1

    n = len(deltas)
    if n == 0:
        return ArenaResult(matches=num_matches, hands=0, bb100=0.0,
                           bb100_ci95=0.0, challenger_promoted=False)

    mean_delta = sum(deltas) / n
    var = sum((d - mean_delta) ** 2 for d in deltas) / max(n - 1, 1)
    sem = math.sqrt(var / n)
    bb100 = mean_delta / BIG_BLIND * 100
    bb100_ci95 = sem / BIG_BLIND * 100 * 1.96

    promoted = bb100 - bb100_ci95 > promotion_threshold
    return ArenaResult(
        matches=num_matches,
        hands=n,
        bb100=bb100,
        bb100_ci95=bb100_ci95,
        challenger_promoted=promoted,
    )
