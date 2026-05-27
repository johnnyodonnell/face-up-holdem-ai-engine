"""Pure 7-card Texas Hold'em hand evaluator.

Port of ../src/engine/handEvaluator.js. Returns dicts with `handRank` and
`kickers` — same shape as the JS side so parity checks compare byte-for-byte.
Cards are dicts: {'rank': int 2..14, 'suit': str 'h'|'d'|'c'|'s'}.
"""

from __future__ import annotations

from itertools import combinations


HAND_RANK = {
    "HIGH_CARD": 0,
    "PAIR": 1,
    "TWO_PAIR": 2,
    "THREE_OF_A_KIND": 3,
    "STRAIGHT": 4,
    "FLUSH": 5,
    "FULL_HOUSE": 6,
    "FOUR_OF_A_KIND": 7,
    "STRAIGHT_FLUSH": 8,
    "ROYAL_FLUSH": 9,
}


def evaluate_hand(seven_cards):
    best_rank = None
    best_kickers = None
    for combo in combinations(seven_cards, 5):
        rank, kickers = _evaluate_five(combo)
        if best_rank is None or _compare(rank, kickers, best_rank, best_kickers) > 0:
            best_rank = rank
            best_kickers = kickers
    return {"handRank": best_rank, "kickers": best_kickers}


def compare_hand_results(a, b):
    return _compare(a["handRank"], a["kickers"], b["handRank"], b["kickers"])


def _compare(rank_a, kickers_a, rank_b, kickers_b):
    if rank_a != rank_b:
        return rank_a - rank_b
    for i in range(min(len(kickers_a), len(kickers_b))):
        if kickers_a[i] != kickers_b[i]:
            return kickers_a[i] - kickers_b[i]
    return 0


def _evaluate_five(cards):
    ranks = sorted((c["rank"] for c in cards), reverse=True)
    suits = [c["suit"] for c in cards]
    is_flush = len(set(suits)) == 1
    is_straight, high_card = _check_straight(ranks)

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    groups = sorted(rank_counts.items(), key=lambda g: (-g[1], -g[0]))
    counts = [g[1] for g in groups]

    if is_straight and is_flush:
        if high_card == 14:
            return HAND_RANK["ROYAL_FLUSH"], [14]
        return HAND_RANK["STRAIGHT_FLUSH"], [high_card]

    if counts[0] == 4:
        return HAND_RANK["FOUR_OF_A_KIND"], [groups[0][0], groups[1][0]]

    if counts[0] == 3 and counts[1] == 2:
        return HAND_RANK["FULL_HOUSE"], [groups[0][0], groups[1][0]]

    if is_flush:
        return HAND_RANK["FLUSH"], list(ranks)

    if is_straight:
        return HAND_RANK["STRAIGHT"], [high_card]

    if counts[0] == 3:
        kickers = sorted((g[0] for g in groups[1:]), reverse=True)
        return HAND_RANK["THREE_OF_A_KIND"], [groups[0][0], *kickers]

    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([groups[0][0], groups[1][0]], reverse=True)
        return HAND_RANK["TWO_PAIR"], [*pairs, groups[2][0]]

    if counts[0] == 2:
        kickers = sorted((g[0] for g in groups[1:]), reverse=True)
        return HAND_RANK["PAIR"], [groups[0][0], *kickers]

    return HAND_RANK["HIGH_CARD"], list(ranks)


def _check_straight(sorted_ranks):
    unique = sorted(set(sorted_ranks), reverse=True)
    if len(unique) != 5:
        return False, 0
    if unique[0] - unique[4] == 4:
        return True, unique[0]
    # Wheel: A-2-3-4-5
    if unique == [14, 5, 4, 3, 2]:
        return True, 5
    return False, 0
