"""PUCT MCTS for face-up Texas Hold'em.

Two node kinds:

  * **decision**  — some seat has to act. Children are indexed by action_idx
    (the 7-action discretization defined in games/holdem.py). The child for
    action_idx `a` may itself be a decision node (regular transition), a
    chance node (the action ends the betting round on a street with cards
    still to be dealt), or a terminal node (the action ended the hand).

  * **chance**    — cards are about to be dealt. Children are indexed by a
    sample index in [0, CHANCE_SAMPLES) and populated lazily: on each visit
    we pick a random sample index, and if its child doesn't exist yet we
    sample a fresh deck of unseen cards and apply the triggering action to
    materialize the child. Subsequent visits to the same sample index reuse
    the materialized child.

Values are stored as **per-seat vectors** (one scalar per seat, in absolute
seat indexing). This is the N-player generalization of fox-lite's scalar
"root-mover-frame Q with sign-flip" pattern: in non-2-player non-zero-sum
games the sign-flip approach is wrong, and the only correct way to back up
a leaf evaluation through multiple opponents' decision nodes is to carry
the full per-seat utility vector.

Leaf evaluation:
  * Encode the leaf state in the leaf's-mover frame.
  * Network outputs (policy[ACTION_SIZE], value[NUM_SEATS]) where the value
    vector is also in mover-frame slot order.
  * Translate the value vector to absolute-seat indexing so it can back up
    through ancestors that may have any seat as actor.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

import config
from games.holdem import (
    ACTION_SIZE,
    NUM_SEATS,
    SUITS,
    STARTING_STACK,
    _action_for_idx,
    apply_action,
    apply_action_idx,
    current_seat,
    encode,
    is_terminal,
    legal_action_mask,
    terminal_payoff,
)


# Network evaluator: (state, mover_seat) -> (policy_logits, value_in_mover_frame).
# The value vector has NUM_SEATS entries; entry i is the value for the seat at
# mover-frame slot i (= absolute seat (mover + i) mod NUM_SEATS).
Evaluator = Callable[[dict, int], tuple[list[float], list[float]]]


# ── Node ──────────────────────────────────────────────────────────


# Kinds: keep as constants for cheap comparison.
DECISION = "decision"
CHANCE = "chance"
TERMINAL = "terminal"


class Node:
    __slots__ = (
        "kind",
        "state",
        "mover",          # decision: actor; chance: parent decision's actor; terminal: -1
        "action_idx",     # chance only: the action triggering the deal
        "prior",
        "visit_count",
        "value_sum",      # list[float] of length NUM_SEATS, absolute frame
        "children",       # dict[int -> Node]
        "expanded",       # decision only
        "legal_mask",     # decision only (cached)
    )

    def __init__(self, kind: str, state: dict, mover: int = -1,
                 action_idx: int = -1, prior: float = 0.0):
        self.kind = kind
        self.state = state
        self.mover = mover
        self.action_idx = action_idx
        self.prior = prior
        self.visit_count = 0
        self.value_sum = [0.0] * NUM_SEATS
        self.children: dict[int, "Node"] = {}
        self.expanded = False
        self.legal_mask: Optional[list[bool]] = None

    def mean_value_for(self, seat: int) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum[seat] / self.visit_count


# ── Helpers ───────────────────────────────────────────────────────


def _mover_frame_to_absolute(value_in_mover_frame: list[float], mover: int) -> list[float]:
    """Translate a mover-frame value vector to absolute-seat indexing.

    `value_in_mover_frame[i]` is the value for the seat at slot i in the
    mover's frame = absolute seat `(mover + i) mod NUM_SEATS`. Returns a
    list where index s holds the value for absolute seat s.
    """
    abs_vec = [0.0] * NUM_SEATS
    for slot in range(NUM_SEATS):
        seat = (mover + slot) % NUM_SEATS
        abs_vec[seat] = value_in_mover_frame[slot]
    return abs_vec


def _classify_post_action(state_before: dict, state_after: dict) -> str:
    """Classify the child state produced by applying an action.

    Returns one of: 'decision', 'chance', 'terminal'. A child is a chance
    transition iff the action caused community cards to be dealt and the
    hand isn't yet locked into a single deterministic outcome — but in
    Hold'em even the "deal remaining and go to showdown" path (all-in
    scenarios) is chance-determined, so we ALSO classify a deal-into-
    terminal as chance. The chance node's K samples then materialize K
    different terminal outcomes.
    """
    cards_dealt = len(state_after["communityCards"]) > len(state_before["communityCards"])
    if cards_dealt:
        return CHANCE
    if state_after["phase"] in ("hand_complete", "game_over"):
        return TERMINAL
    return DECISION


def _all_cards():
    """Generate the full 52-card deck in canonical order (rank-major then suit).

    Matches games.holdem._create_deck but as a local helper to avoid
    relying on a non-exported function.
    """
    labels = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
              10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
    cards = []
    for suit in SUITS:
        for rank in range(2, 15):
            cards.append({"rank": rank, "suit": suit, "id": f"{labels[rank]}{suit}"})
    return cards


def _sample_unseen_deck(state: dict, rng: random.Random) -> list[dict]:
    """A 52-card list where the leading unseen cards are randomly shuffled.

    Known cards (any seat's hole cards + already-revealed community cards)
    are placed AFTER the unseen cards. The rules core only ever consumes
    cards from the front of `state["deck"]` for community deals, so what
    matters is the order of the unseen cards at the top.
    """
    known_ids = set()
    for p in state["players"]:
        for c in p["holeCards"]:
            known_ids.add(c["id"])
    for c in state["communityCards"]:
        known_ids.add(c["id"])

    all_cards = _all_cards()
    unseen = [c for c in all_cards if c["id"] not in known_ids]
    known = [c for c in all_cards if c["id"] in known_ids]
    rng.shuffle(unseen)
    return unseen + known


def _terminal_value_vec(state: dict, start_chips: list[int]) -> list[float]:
    """Per-seat payoff vector at a terminal state.

    Each seat's payoff is its chip delta this hand divided by STARTING_STACK,
    clipped to [-1, 1] to match the network's tanh-bounded value head.
    """
    return [terminal_payoff(state, seat, start_chips[seat]) for seat in range(NUM_SEATS)]


def _terminal_value_vec_from_state_diff(
    state_terminal: dict, start_chips: list[int]
) -> list[float]:
    """Alias used in the search to keep call sites readable."""
    return _terminal_value_vec(state_terminal, start_chips)


# ── Expansion ─────────────────────────────────────────────────────


def _masked_softmax_priors(logits: list[float], legal_idxs: list[int]) -> dict[int, float]:
    """Softmax over only the legal action logits."""
    legal_logits = [logits[i] for i in legal_idxs]
    m = max(legal_logits)
    exps = [math.exp(lg - m) for lg in legal_logits]
    z = sum(exps)
    return {action_idx: exps[k] / z for k, action_idx in enumerate(legal_idxs)}


def _expand_decision_node(
    node: Node,
    evaluator: Evaluator,
    start_chips: list[int],
) -> list[float]:
    """Expand the decision node, build its children, return abs-frame value vec."""
    assert node.kind == DECISION
    assert not node.expanded

    mask = legal_action_mask(node.state)
    legal_idxs = [i for i, m in enumerate(mask) if m]
    node.legal_mask = mask

    if not legal_idxs:
        # Degenerate: no legal actions but state isn't terminal. Shouldn't
        # happen — fall back to a single-child no-op of sorts. Treat as
        # terminal for safety.
        node.kind = TERMINAL
        return _terminal_value_vec(node.state, start_chips)

    policy_logits, value_in_mover_frame = evaluator(node.state, node.mover)
    priors = _masked_softmax_priors(policy_logits, legal_idxs)

    for action_idx in legal_idxs:
        candidate_state = apply_action_idx(node.state, action_idx)
        classification = _classify_post_action(node.state, candidate_state)

        if classification == DECISION:
            child = Node(
                kind=DECISION,
                state=candidate_state,
                mover=candidate_state["actionOnSeat"],
                prior=priors[action_idx],
            )
        elif classification == TERMINAL:
            child = Node(
                kind=TERMINAL,
                state=candidate_state,
                prior=priors[action_idx],
            )
        else:  # CHANCE
            # We discard candidate_state because it used `node.state["deck"]`
            # as a single concrete sample. The chance node lazily samples its
            # own children when visited.
            child = Node(
                kind=CHANCE,
                state=node.state,        # the pre-action state — what we'll re-apply with sampled decks
                mover=node.mover,        # the actor about to act; carried for diagnostic clarity
                action_idx=action_idx,
                prior=priors[action_idx],
            )
        node.children[action_idx] = child

    node.expanded = True
    return _mover_frame_to_absolute(value_in_mover_frame, node.mover)


def _materialize_chance_child(
    node: Node,
    sample_idx: int,
    rng: random.Random,
) -> Node:
    """Create the `sample_idx`-th child of a chance node by sampling a fresh
    deck of unseen cards and applying the triggering action. Returns a
    decision or terminal node."""
    sampled_deck = _sample_unseen_deck(node.state, rng)
    state_with_deck = dict(node.state)
    state_with_deck["deck"] = sampled_deck
    result_state = apply_action_idx(state_with_deck, node.action_idx)

    if result_state["phase"] in ("hand_complete", "game_over"):
        return Node(kind=TERMINAL, state=result_state)
    return Node(
        kind=DECISION,
        state=result_state,
        mover=result_state["actionOnSeat"],
    )


# ── Selection ─────────────────────────────────────────────────────


def _puct_select(node: Node, c_puct: float) -> tuple[int, Node]:
    """PUCT child selection at a decision node. Ties broken by lowest action
    index. Q is in the actor's perspective: we pick the child whose mean
    value for seat=node.mover is highest, plus the exploration bonus."""
    sqrt_n = math.sqrt(node.visit_count) if node.visit_count > 0 else 0.0
    best_score = -math.inf
    best_action = next(iter(node.children))
    best_child = node.children[best_action]
    for action_idx in sorted(node.children.keys()):
        child = node.children[action_idx]
        q = child.mean_value_for(node.mover)
        # If sqrt_n is 0 (root never visited the children yet), the U term
        # would also be 0 and selection collapses to highest prior at zero
        # visits.
        u = c_puct * child.prior * sqrt_n / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action_idx
            best_child = child
    return best_action, best_child


# ── Main entry point ──────────────────────────────────────────────


def run_mcts(
    root_state: dict,
    evaluator: Evaluator,
    start_chips: list[int],
    num_simulations: int,
    c_puct: float = config.C_PUCT,
    chance_samples: int = config.CHANCE_SAMPLES,
    add_dirichlet_noise: bool = False,
    dirichlet_alpha: float = config.DIRICHLET_ALPHA,
    dirichlet_epsilon: float = config.DIRICHLET_EPSILON,
    rng: Optional[random.Random] = None,
) -> Node:
    """Run `num_simulations` PUCT simulations from `root_state` and return the
    populated root node.

    `start_chips` is each seat's chip count at the **start of the hand** —
    used so terminal values are correctly normalized to chip deltas. The
    selfplay caller snapshots this just before calling start_new_hand.

    Values throughout the tree are stored in absolute-seat indexing; the
    root's `mean_value_for(root_state['actionOnSeat'])` is the value
    estimate for the actor on root.
    """
    rng = rng or random.Random()

    if is_terminal(root_state):
        # Root is terminal — nothing to search. Return a node with the
        # terminal value already accumulated once so callers reading
        # `visit_count` don't see zero.
        root = Node(kind=TERMINAL, state=root_state)
        val_vec = _terminal_value_vec(root_state, start_chips)
        root.visit_count = 1
        root.value_sum = list(val_vec)
        return root

    root = Node(kind=DECISION, state=root_state, mover=current_seat(root_state))
    root_abs_val = _expand_decision_node(root, evaluator, start_chips)
    # Seed the root with the expansion-time evaluation.
    root.visit_count += 1
    for i in range(NUM_SEATS):
        root.value_sum[i] += root_abs_val[i]

    # Dirichlet noise on root priors (self-play only).
    if add_dirichlet_noise and root.children:
        actions = sorted(root.children.keys())
        # numpy-free Dirichlet: sample from gamma(alpha, 1) and normalize.
        gammas = [rng.gammavariate(dirichlet_alpha, 1.0) for _ in actions]
        total = sum(gammas) or 1.0
        noise = [g / total for g in gammas]
        for action_idx, n in zip(actions, noise):
            child = root.children[action_idx]
            child.prior = child.prior * (1 - dirichlet_epsilon) + n * dirichlet_epsilon

    for _ in range(num_simulations):
        path = [root]
        node = root
        # Descend until we hit a leaf (unexpanded decision, fresh chance child,
        # or terminal).
        leaf_value_vec: Optional[list[float]] = None
        while True:
            if node.kind == TERMINAL:
                leaf_value_vec = _terminal_value_vec(node.state, start_chips)
                break

            if node.kind == DECISION:
                if not node.expanded:
                    leaf_value_vec = _expand_decision_node(node, evaluator, start_chips)
                    break
                _, child = _puct_select(node, c_puct)
                node = child
                path.append(node)
                continue

            # CHANCE
            k = rng.randrange(chance_samples)
            if k not in node.children:
                child = _materialize_chance_child(node, k, rng)
                node.children[k] = child
                path.append(child)
                if child.kind == TERMINAL:
                    leaf_value_vec = _terminal_value_vec(child.state, start_chips)
                else:
                    leaf_value_vec = _expand_decision_node(child, evaluator, start_chips)
                break
            node = node.children[k]
            path.append(node)

        # Backup
        assert leaf_value_vec is not None
        for n in path:
            n.visit_count += 1
            for i in range(NUM_SEATS):
                n.value_sum[i] += leaf_value_vec[i]

    return root


# ── Visit-count distribution (policy target) ──────────────────────


def root_visit_distribution(root: Node) -> list[float]:
    """Length-ACTION_SIZE distribution over root's children visit counts.

    Non-legal / non-explored actions get 0. Sums to 1 over the legal,
    visited set; if the root saw zero descents into any child (impossible
    after a successful run_mcts call), returns a uniform distribution.
    """
    counts = [0.0] * ACTION_SIZE
    total = 0.0
    for action_idx, child in root.children.items():
        counts[action_idx] = float(child.visit_count)
        total += child.visit_count
    if total > 0:
        return [c / total for c in counts]
    # fallback (extremely unlikely after MCTS): uniform over legal actions
    if root.legal_mask is not None:
        legal = [i for i, m in enumerate(root.legal_mask) if m]
    else:
        legal = list(range(ACTION_SIZE))
    if not legal:
        return [0.0] * ACTION_SIZE
    p = 1.0 / len(legal)
    out = [0.0] * ACTION_SIZE
    for i in legal:
        out[i] = p
    return out


# ── Network-backed evaluator factory ──────────────────────────────


def network_evaluator(net) -> Evaluator:
    """Wrap a PolicyValueNet into the MCTS Evaluator protocol."""
    from alphazero.network import infer

    def evaluator(state: dict, mover: int) -> tuple[list[float], list[float]]:
        encoded = encode(state, mover)
        return infer(net, encoded)

    return evaluator
