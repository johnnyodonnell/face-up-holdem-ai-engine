"""Sanity check: confirm the heuristic beats random play by a clear margin.

Each "match" is one fresh game where 1 seat plays the heuristic and the
other 7 play random-legal actions. The heuristic seat is rotated across
all 8 positions to neutralize positional advantage. Reports heuristic
performance as BB/100 with a 95% CI.

If the heuristic is doing anything useful, we should see ≥ +10 BB/100 here
with a tight CI. If it's near zero or negative, something is wrong.

Run from training/:
    python scripts/heuristic_vs_random.py
    python scripts/heuristic_vs_random.py --hands 5000 --samples 80
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING_DIR = HERE.parent
sys.path.insert(0, str(TRAINING_DIR))

from games.holdem import (  # noqa: E402
    BIG_BLIND,
    NUM_SEATS,
    apply_action,
    create_game,
    current_seat,
    legal_actions,
    start_new_hand,
)
from heuristic import best_move  # noqa: E402


def random_action(state, rng: random.Random):
    legal = legal_actions(state)
    choice = dict(rng.choice(legal))
    if choice["type"] == "raise":
        lo = state["minRaiseAmount"]
        hi = state["maxRaiseAmount"]
        choice["amount"] = rng.randint(lo, hi)
    return choice


def play_one_hand(hero_seat: int, rng: random.Random, hand_seed: int, num_samples: int):
    state = create_game(rng)
    start_chips = state["players"][hero_seat]["chips"]
    state = start_new_hand(state, rng=rng)
    safety = 0
    while state["phase"] not in ("hand_complete", "game_over"):
        actor = current_seat(state)
        if actor < 0:
            break
        if actor == hero_seat:
            action = best_move(state, seed=hand_seed * 100 + actor, num_samples=num_samples)
        else:
            action = random_action(state, rng)
        state = apply_action(state, action)
        safety += 1
        if safety > 500:
            raise RuntimeError(f"hand {hand_seed} did not terminate")
    end_chips = state["players"][hero_seat]["chips"]
    return end_chips - start_chips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=80,
                        help="MC samples per heuristic decision")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    deltas = []
    t0 = time.time()

    for hand_idx in range(args.hands):
        hero_seat = hand_idx % NUM_SEATS  # rotate positions
        delta = play_one_hand(hero_seat, rng, hand_idx + 1, args.samples)
        deltas.append(delta)
        if (hand_idx + 1) % 200 == 0:
            n = len(deltas)
            mean = sum(deltas) / n
            bb100 = mean / BIG_BLIND * 100
            elapsed = time.time() - t0
            print(
                f"  {hand_idx + 1:5d}/{args.hands}: "
                f"mean_delta={mean:+.3f} chips ({bb100:+.2f} BB/100), "
                f"{elapsed:.1f}s elapsed"
            )

    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / max(n - 1, 1)
    sem = math.sqrt(var / n)
    bb100 = mean / BIG_BLIND * 100
    bb100_ci95 = sem / BIG_BLIND * 100 * 1.96

    print()
    print(f"== heuristic vs random ({n} hands, {args.samples} MC samples/decision) ==")
    print(f"   mean delta:  {mean:+.4f} chips")
    print(f"   stddev:      {math.sqrt(var):.2f}")
    print(f"   BB/100:      {bb100:+.2f} ± {bb100_ci95:.2f} (95% CI)")
    print(f"   wall time:   {time.time() - t0:.1f}s")

    if bb100 - bb100_ci95 > 0:
        print("   verdict: heuristic IS beating random with non-overlapping CI")
    else:
        print("   verdict: heuristic edge NOT statistically distinguishable from zero")


if __name__ == "__main__":
    main()
