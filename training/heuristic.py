"""Equity-based heuristic baseline for face-up Hold'em.

Two roles in the project:
  1. Absolute-strength yardstick. AlphaZero's arena tells us "newer net beats
     older net" but cannot say "the net is actually any good". This heuristic
     gives an external reference point: trained-net BB/100 vs this heuristic.
  2. Rules-port cross-check. The same algorithm exists in JS
     (`src/engine/heuristic.js`). When seeded identically the two sides MUST
     pick the same action — `scripts/parity_heuristic.py` enforces this.

The algorithm:
  * Monte Carlo equity vs the live opponents' visible hole cards. Face-up
    means we know everyone's hand; the only unknowns are future board cards.
  * Sample the remaining board cards `K` times, evaluate each 7-card hand,
    accumulate fractional pot share (chops divide).
  * Threshold table on equity + pot odds picks an action over the 7-action
    discretization defined in `games/holdem.py`.

The heuristic is NOT shipped to the browser as the production engine. It
lives only as a measuring stick.
"""

from __future__ import annotations

from typing import Optional

from games.hand_evaluator import evaluate_hand, compare_hand_results
from games.holdem import (
    ACTION_FOLD,
    ACTION_CHECK_CALL,
    ACTION_RAISE_HALF_POT,
    ACTION_RAISE_POT,
    ACTION_RAISE_TWO_POT,
    ACTION_ALL_IN,
    SUITS,
    _action_for_idx,
)


# ── Deterministic PRNG (matches JS) ───────────────────────────────


class LCG:
    """Linear congruential generator. Same constants as parity_corpus.mjs
    so JS and Python produce identical sequences from the same seed."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next_float(self) -> float:
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state / 0x100000000


# ── Card-set helpers ──────────────────────────────────────────────


def _card_key(card):
    """Canonical sort key: rank then suit, matching games/holdem.card_index."""
    return (card["rank"] - 2) * 4 + SUITS.index(card["suit"])


def _all_cards():
    cards = []
    for suit in SUITS:
        for rank in range(2, 15):
            cards.append({"rank": rank, "suit": suit, "id": _id_for(rank, suit)})
    return cards


def _id_for(rank: int, suit: str) -> str:
    labels = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
              10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
    return f"{labels[rank]}{suit}"


# ── Monte Carlo equity ────────────────────────────────────────────


def equity(state, hero_seat: int, num_samples: int, rng: LCG) -> float:
    """Hero's expected fractional pot share at showdown given everyone's
    visible hole cards and the current community cards.

    Live opponents = seats other than hero whose status is 'active' or
    'all_in' (i.e., still contesting the pot). Folded and eliminated seats
    are excluded.

    If there are no live opponents, the hero is uncontested → equity = 1.0.
    """
    hero = state["players"][hero_seat]
    if not hero["holeCards"]:
        return 0.0

    live_opps = [
        p for p in state["players"]
        if p["seat"] != hero_seat
        and p["status"] in ("active", "all_in")
        and len(p["holeCards"]) == 2
    ]
    if not live_opps:
        return 1.0

    community = list(state["communityCards"])
    known_ids = {c["id"] for c in hero["holeCards"]}
    for p in live_opps:
        for c in p["holeCards"]:
            known_ids.add(c["id"])
    for c in community:
        known_ids.add(c["id"])

    remaining = sorted(
        (c for c in _all_cards() if c["id"] not in known_ids),
        key=_card_key,
    )
    needed = 5 - len(community)

    if needed == 0:
        # River already out — single evaluation.
        return _equity_one_board(hero, live_opps, community)

    total = 0.0
    for _ in range(num_samples):
        sampled = _sample_without_replacement(remaining, needed, rng)
        board = community + sampled
        total += _equity_one_board(hero, live_opps, board)
    return total / num_samples


def _equity_one_board(hero, live_opps, full_board):
    hero_hand = evaluate_hand(hero["holeCards"] + full_board)
    opp_hands = [evaluate_hand(o["holeCards"] + full_board) for o in live_opps]
    best = hero_hand
    for h in opp_hands:
        if compare_hand_results(h, best) > 0:
            best = h
    # Count how many players tie with the best.
    ties = 0
    if compare_hand_results(hero_hand, best) == 0:
        ties += 1
    for h in opp_hands:
        if compare_hand_results(h, best) == 0:
            ties += 1
    if compare_hand_results(hero_hand, best) == 0:
        return 1.0 / ties
    return 0.0


def _sample_without_replacement(deck, count, rng: LCG):
    """Partial Fisher-Yates: in-place swap the first `count` slots into
    sampled positions. Caller owns the returned list (a fresh copy)."""
    a = list(deck)
    for i in range(count):
        # pick j ∈ [i, len(a))
        r = rng.next_float()
        j = i + int(r * (len(a) - i))
        a[i], a[j] = a[j], a[i]
    return a[:count]


# ── Action selection ──────────────────────────────────────────────


# Thresholds tuned for "plays sensibly, not too tight". The exact values
# don't matter for the heuristic's role as baseline; they just need to
# match between JS and Python.
EQUITY_ALL_IN = 0.90
EQUITY_RAISE_TWO_POT = 0.80
EQUITY_RAISE_POT = 0.65
EQUITY_RAISE_HALF_POT = 0.50

# Default per-decision MC budget. 200 samples → standard error ~3.5%, plenty
# for threshold-based decisions.
DEFAULT_NUM_SAMPLES = 200


def choose_action_idx(state, equity_value: float) -> int:
    """Map equity → action index. Falls back through the legal mask: if the
    chosen action is illegal (e.g., we want to raise but can't), the next
    legal option is taken in priority order raise→call→fold."""
    seat = state["actionOnSeat"]
    can_raise = (
        state["maxRaiseAmount"] > 0
        and state["maxRaiseAmount"] > state["currentStreetBet"]
    )

    actor = state["players"][seat]
    to_call = state["currentStreetBet"] - actor["currentBet"]
    can_check = to_call == 0

    # Raise tier — only consult if we can actually raise.
    if can_raise:
        if equity_value >= EQUITY_ALL_IN:
            return ACTION_ALL_IN
        if equity_value >= EQUITY_RAISE_TWO_POT:
            return ACTION_RAISE_TWO_POT
        if equity_value >= EQUITY_RAISE_POT:
            return ACTION_RAISE_POT
        if equity_value >= EQUITY_RAISE_HALF_POT:
            return ACTION_RAISE_HALF_POT

    # Call/fold tier.
    pot_before_call = state["totalPot"] + sum(p["currentBet"] for p in state["players"])
    pot_after_call = pot_before_call + to_call
    if to_call == 0:
        required_equity = 0.0
    else:
        required_equity = to_call / pot_after_call

    if equity_value >= required_equity:
        return ACTION_CHECK_CALL  # which becomes check or call per state
    if can_check:
        return ACTION_CHECK_CALL
    return ACTION_FOLD


# ── Public entry points ───────────────────────────────────────────


def best_action_idx(state, seed: int = 1, num_samples: int = DEFAULT_NUM_SAMPLES) -> int:
    """Pick an action index for the actor on `state`. Deterministic given
    `seed`. Used by the AlphaZero pipeline as the baseline opponent."""
    seat = state["actionOnSeat"]
    if seat < 0:
        return ACTION_CHECK_CALL  # caller error; nothing legal here
    rng = LCG(seed)
    eq = equity(state, seat, num_samples, rng)
    return choose_action_idx(state, eq)


def best_move(state, seed: int = 1, num_samples: int = DEFAULT_NUM_SAMPLES) -> Optional[dict]:
    """Return the concrete {type, amount?} action — matches the JS
    `bestMove(state)` contract. Used by parity tests and by Python-side
    play simulations against `random.js`-equivalent opponents."""
    idx = best_action_idx(state, seed, num_samples)
    action = _action_for_idx(state, idx)
    if action is not None:
        return action
    # Fallback chain: requested action was illegal (rare; can happen if
    # equity exactly equals required and the mask rejects). Try check/call,
    # then fold.
    fallback = _action_for_idx(state, ACTION_CHECK_CALL)
    if fallback is not None:
        return fallback
    return _action_for_idx(state, ACTION_FOLD)
