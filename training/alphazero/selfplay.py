"""Generate self-play training records by playing complete hands with MCTS.

One self-play "match" is one hand of Hold'em where the same network plays
every seat. At each decision, MCTS expands the tree under `num_simulations`
PUCT simulations (with Dirichlet noise on the root prior) and the chosen
action is sampled from the visit-count distribution with temperature 1 for
the first `TEMPERATURE_DECISIONS` plies and argmax thereafter.

After the hand ends, each decision's value target is back-filled from the
final per-seat chip deltas in the recording actor's mover frame.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

import config
from alphazero.mcts import (
    network_evaluator,
    root_visit_distribution,
    run_mcts,
)
from games.holdem import (
    NUM_SEATS,
    STARTING_STACK,
    apply_action_idx,
    create_game,
    current_seat,
    encode,
    is_terminal,
    start_new_hand,
    terminal_payoff,
)


def _sample_from_distribution(distribution: list[float], rng: random.Random) -> int:
    """Sample an action index from a probability distribution that sums to 1."""
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(distribution):
        acc += p
        if r <= acc:
            return i
    # Floating-point slack: pick the last non-zero index
    for i in range(len(distribution) - 1, -1, -1):
        if distribution[i] > 0:
            return i
    return 0


def _argmax(distribution: list[float]) -> int:
    best_i = 0
    best_v = distribution[0]
    for i in range(1, len(distribution)):
        if distribution[i] > best_v:
            best_v = distribution[i]
            best_i = i
    return best_i


def _value_target_for_decision(
    final_state: dict,
    start_chips: list[int],
    actor_seat: int,
) -> np.ndarray:
    """Per-seat payoff vector in `actor_seat`'s mover frame.

    Slot 0 = payoff for the actor; slot s = payoff for absolute seat
    (actor + s) mod NUM_SEATS.
    """
    vec = np.zeros(NUM_SEATS, dtype=np.float32)
    for slot in range(NUM_SEATS):
        seat = (actor_seat + slot) % NUM_SEATS
        vec[slot] = terminal_payoff(final_state, seat, start_chips[seat])
    return vec


def play_one_hand(
    net,
    rng: Optional[random.Random] = None,
    num_simulations: int = config.NUM_SIMULATIONS,
    starting_state: Optional[dict] = None,
) -> list[tuple[list[float], np.ndarray, np.ndarray]]:
    """Play one hand of self-play and return its training records.

    Each record is `(encoded_state, policy_target, value_target)` where:
      * encoded_state is the encoder output in the acting seat's frame,
      * policy_target is the MCTS root-visit distribution over ACTION_SIZE,
      * value_target is the per-seat payoff vector in the acting seat's frame.

    `starting_state` lets the caller chain hands across a multi-hand session;
    if None, we create a fresh game.
    """
    rng = rng or random.Random()

    state = starting_state if starting_state is not None else create_game(rng)
    # Snapshot chips BEFORE blinds are posted — that's the natural reference
    # for "chip delta this hand".
    start_chips = [p["chips"] for p in state["players"]]
    state = start_new_hand(state, rng=rng)

    evaluator = network_evaluator(net)

    records: list[tuple[list[float], np.ndarray, np.ndarray, int]] = []
    decision_count = 0

    while not is_terminal(state):
        actor = current_seat(state)
        if actor < 0:
            break

        root = run_mcts(
            state,
            evaluator,
            start_chips=start_chips,
            num_simulations=num_simulations,
            add_dirichlet_noise=True,
            rng=rng,
        )
        visit_dist = root_visit_distribution(root)

        encoded = encode(state, actor)
        records.append(
            (encoded, np.asarray(visit_dist, dtype=np.float32), None, actor)  # value filled later
        )

        if decision_count < config.TEMPERATURE_DECISIONS:
            action_idx = _sample_from_distribution(visit_dist, rng)
        else:
            action_idx = _argmax(visit_dist)
        decision_count += 1

        state = apply_action_idx(state, action_idx)

    # Back-fill value targets from the terminal state.
    final_records = []
    for encoded, policy, _, actor in records:
        value = _value_target_for_decision(state, start_chips, actor)
        final_records.append((encoded, policy, value))

    return final_records


def play_many_hands(
    net,
    num_hands: int,
    rng: Optional[random.Random] = None,
    num_simulations: int = config.NUM_SIMULATIONS,
) -> list[tuple[list[float], np.ndarray, np.ndarray]]:
    """Play `num_hands` independent hands and return the concatenated records.

    Each hand starts from a fresh game (everyone at STARTING_STACK). This is
    deliberate: keeps the self-play data distribution centered on the
    canonical opening stack rather than drifting into late-game short-stack
    regimes that won't be encountered uniformly.
    """
    rng = rng or random.Random()
    all_records = []
    for _ in range(num_hands):
        all_records.extend(play_one_hand(net, rng=rng, num_simulations=num_simulations))
    return all_records
