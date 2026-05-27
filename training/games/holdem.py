"""Pure Face-Up Texas Hold'em rules in Python.

Port of ../src/engine/game.js and ../src/engine/potCalculator.js. Same state
shape, same transitions, so the JS↔Python parity corpus compares byte-for-byte.

The shuffle is the only nondeterminism in the rules. Public entry points that
introduce randomness (`start_new_hand`) accept an optional pre-shuffled `deck`
parameter so parity tests can replay JS-side shuffles deterministically.
"""

from __future__ import annotations

import copy
import random as _random

from .hand_evaluator import evaluate_hand, compare_hand_results


SMALL_BLIND = 1
BIG_BLIND = 2
STARTING_STACK = 200
NUM_SEATS = 8
HUMAN_SEAT = 0

BOT_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace"]
SUITS = ["h", "d", "c", "s"]
RANK_LABELS = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}


# ── Deck ──────────────────────────────────────────────────────────


def card_label(card):
    return f"{RANK_LABELS[card['rank']]}{card['suit']}"


def _create_deck():
    cards = []
    for suit in SUITS:
        for rank in range(2, 15):
            cards.append({"rank": rank, "suit": suit, "id": f"{RANK_LABELS[rank]}{suit}"})
    return cards


def _shuffle(cards, rng):
    a = list(cards)
    for i in range(len(a) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def _deal(deck, count):
    return deck[:count], deck[count:]


# ── Game / player construction ────────────────────────────────────


def _create_player(seat, name, is_human):
    return {
        "seat": seat,
        "name": name,
        "isHuman": is_human,
        "chips": STARTING_STACK,
        "status": "active",
        "holeCards": [],
        "currentBet": 0,
        "totalBetThisHand": 0,
        "hasActedThisStreet": False,
        "isDealer": False,
        "isSB": False,
        "isBB": False,
        "handRank": None,
        "handKickers": None,
    }


def create_game(rng=None):
    rng = rng or _random.Random()
    players = [_create_player(0, "You", True)]
    for i, name in enumerate(BOT_NAMES):
        players.append(_create_player(i + 1, name, False))
    initial_dealer = int(rng.random() * NUM_SEATS)
    return {
        "handNumber": 0,
        "phase": "waiting",
        "deck": [],
        "communityCards": [],
        "players": players,
        "pots": [],
        "totalPot": 0,
        "dealerSeat": initial_dealer,
        "actionOnSeat": -1,
        "lastAggressorSeat": -1,
        "currentStreetBet": 0,
        "lastRaiseSize": BIG_BLIND,
        "callAmount": 0,
        "canCheck": False,
        "minRaiseAmount": 0,
        "maxRaiseAmount": 0,
        "winners": [],
    }


# ── Start a new hand ──────────────────────────────────────────────


def start_new_hand(prev_state, rng=None, deck=None):
    """Begin the next hand from `prev_state`.

    Pass `deck` (52-card list) to replay a JS-recorded shuffle for parity.
    Otherwise a fresh shuffle is drawn from `rng` (or the global Random).
    """
    state = copy.deepcopy(prev_state)
    rng = rng or _random.Random()

    for p in state["players"]:
        p["holeCards"] = []
        p["handRank"] = None
        p["handKickers"] = None
        p["isDealer"] = False
        p["isSB"] = False
        p["isBB"] = False
        if p["status"] != "eliminated":
            p["status"] = "active"
            p["currentBet"] = 0
            p["totalBetThisHand"] = 0
            p["hasActedThisStreet"] = False

    state["communityCards"] = []
    state["pots"] = []
    state["totalPot"] = 0
    state["winners"] = []
    state["handNumber"] += 1
    state["lastRaiseSize"] = BIG_BLIND

    state["dealerSeat"] = _next_non_eliminated_seat(state["players"], state["dealerSeat"])
    state["players"][state["dealerSeat"]]["isDealer"] = True

    active_count = sum(1 for p in state["players"] if p["status"] != "eliminated")

    if active_count == 2:
        # Heads-up: dealer posts SB
        sb_seat = state["dealerSeat"]
        bb_seat = _next_non_eliminated_seat(state["players"], state["dealerSeat"])
    else:
        sb_seat = _next_non_eliminated_seat(state["players"], state["dealerSeat"])
        bb_seat = _next_non_eliminated_seat(state["players"], sb_seat)

    _post_blind(state["players"][sb_seat], SMALL_BLIND)
    state["players"][sb_seat]["isSB"] = True
    _post_blind(state["players"][bb_seat], BIG_BLIND)
    state["players"][bb_seat]["isBB"] = True

    state["deck"] = list(deck) if deck is not None else _shuffle(_create_deck(), rng)
    for p in state["players"]:
        if p["status"] != "eliminated":
            dealt, remaining = _deal(state["deck"], 2)
            p["holeCards"] = dealt
            state["deck"] = remaining

    state["phase"] = "preflop"
    state["currentStreetBet"] = BIG_BLIND
    state["lastAggressorSeat"] = bb_seat

    first_to_act = _next_active_seat_for_betting(state["players"], bb_seat)
    if first_to_act == -1:
        return _deal_remaining_and_showdown(state)

    state["actionOnSeat"] = first_to_act
    _update_action_info(state)
    return state


# ── Apply an action (human or bot) ────────────────────────────────


def apply_action(prev_state, action):
    state = copy.deepcopy(prev_state)
    seat = state["actionOnSeat"]
    player = state["players"][seat]

    a_type = action["type"]
    if a_type == "fold":
        player["status"] = "folded"
    elif a_type == "check":
        pass
    elif a_type == "call":
        call_amount = min(state["currentStreetBet"] - player["currentBet"], player["chips"])
        player["chips"] -= call_amount
        player["currentBet"] += call_amount
        if player["chips"] == 0:
            player["status"] = "all_in"
    elif a_type == "raise":
        raise_total = min(action["amount"], player["chips"] + player["currentBet"])
        cost = raise_total - player["currentBet"]
        player["chips"] -= cost
        state["lastRaiseSize"] = raise_total - state["currentStreetBet"]
        state["currentStreetBet"] = raise_total
        player["currentBet"] = raise_total
        state["lastAggressorSeat"] = seat
        for p in state["players"]:
            if p["seat"] != seat and p["status"] == "active":
                p["hasActedThisStreet"] = False
        if player["chips"] == 0:
            player["status"] = "all_in"
    else:
        raise ValueError(f"unknown action type: {a_type}")

    player["hasActedThisStreet"] = True
    return _advance_action(state)


# ── Legal actions for the current actor ───────────────────────────


def legal_actions(state):
    """Returns the JS-shape list of {type, amount?} dicts for the actor."""
    if state["actionOnSeat"] < 0:
        return []
    player = state["players"][state["actionOnSeat"]]
    if not player or player["status"] != "active":
        return []

    actions = []
    to_call = state["currentStreetBet"] - player["currentBet"]

    if to_call == 0:
        actions.append({"type": "check"})
    else:
        actions.append({"type": "fold"})
        actions.append({"type": "call"})

    if state["maxRaiseAmount"] > 0 and state["maxRaiseAmount"] > state["currentStreetBet"]:
        actions.append({"type": "raise", "amount": state["minRaiseAmount"]})

    return actions


# ── Advance / end-of-street / showdown ────────────────────────────


def _advance_action(state):
    non_folded = [p for p in state["players"] if p["status"] not in ("folded", "eliminated")]
    if len(non_folded) == 1:
        return _award_pot_to_last_player(state)

    next_seat = _find_next_seat_to_act(state)
    if next_seat == -1:
        return _end_betting_round(state)

    state["actionOnSeat"] = next_seat
    _update_action_info(state)
    return state


def _find_next_seat_to_act(state):
    n = len(state["players"])
    seat = (state["actionOnSeat"] + 1) % n
    for _ in range(n):
        p = state["players"][seat]
        if p["status"] == "active" and not p["hasActedThisStreet"]:
            return seat
        seat = (seat + 1) % n
    return -1


def _end_betting_round(state):
    _collect_bets_into_pot(state)

    active_players = [p for p in state["players"] if p["status"] == "active"]
    non_folded = [p for p in state["players"] if p["status"] not in ("folded", "eliminated")]

    if len(active_players) <= 1 and len(non_folded) > 1:
        return _deal_remaining_and_showdown(state)

    next_p = _next_phase(state["phase"])
    state["phase"] = next_p

    if next_p == "showdown":
        return _run_showdown(state)

    _deal_street_cards(state)

    state["currentStreetBet"] = 0
    state["lastAggressorSeat"] = -1
    state["lastRaiseSize"] = BIG_BLIND
    for p in state["players"]:
        p["hasActedThisStreet"] = False

    first_to_act = _next_active_seat_for_betting(state["players"], state["dealerSeat"])
    if first_to_act == -1:
        return _deal_remaining_and_showdown(state)

    state["actionOnSeat"] = first_to_act
    _update_action_info(state)
    return state


def _collect_bets_into_pot(state):
    for p in state["players"]:
        state["totalPot"] += p["currentBet"]
        p["totalBetThisHand"] += p["currentBet"]
        p["currentBet"] = 0


def _next_phase(current):
    return {
        "preflop": "flop",
        "flop": "turn",
        "turn": "river",
        "river": "showdown",
    }.get(current, "showdown")


def _deal_street_cards(state):
    if state["phase"] == "flop":
        cards_to_deal = 3
    elif state["phase"] in ("turn", "river"):
        cards_to_deal = 1
    else:
        return
    # Burn one
    state["deck"] = state["deck"][1:]
    dealt, remaining = _deal(state["deck"], cards_to_deal)
    state["communityCards"] = state["communityCards"] + dealt
    state["deck"] = remaining


def _deal_remaining_and_showdown(state):
    if any(p["currentBet"] > 0 for p in state["players"]):
        _collect_bets_into_pot(state)
    needed = 5 - len(state["communityCards"])
    for _ in range(needed):
        state["deck"] = state["deck"][1:]  # burn
        dealt, remaining = _deal(state["deck"], 1)
        state["communityCards"] = state["communityCards"] + dealt
        state["deck"] = remaining
    state["phase"] = "showdown"
    return _run_showdown(state)


def _run_showdown(state):
    if any(p["currentBet"] > 0 for p in state["players"]):
        _collect_bets_into_pot(state)

    for p in state["players"]:
        if p["status"] in ("active", "all_in"):
            seven = p["holeCards"] + state["communityCards"]
            result = evaluate_hand(seven)
            p["handRank"] = result["handRank"]
            p["handKickers"] = result["kickers"]

    state["pots"] = calculate_showdown_pots(state["players"])
    state["totalPot"] = get_total_pot(state["pots"])
    state["winners"] = _award_pots(state)

    _finalize_hand(state)
    return state


def _award_pot_to_last_player(state):
    if any(p["currentBet"] > 0 for p in state["players"]):
        _collect_bets_into_pot(state)
    winner = next(p for p in state["players"] if p["status"] not in ("folded", "eliminated"))
    winner["chips"] += state["totalPot"]
    state["winners"] = [{"seat": winner["seat"], "amountWon": state["totalPot"]}]
    state["totalPot"] = 0
    state["pots"] = []
    _finalize_hand(state)
    return state


def _award_pots(state):
    winnings_map = {}

    for pot in state["pots"]:
        eligible = [
            state["players"][seat]
            for seat in pot["eligibleSeats"]
            if state["players"][seat]["status"] != "folded"
            and state["players"][seat]["handRank"] is not None
        ]
        if not eligible:
            continue

        best = [eligible[0]]
        for cand in eligible[1:]:
            cmp = compare_hand_results(
                {"handRank": cand["handRank"], "kickers": cand["handKickers"]},
                {"handRank": best[0]["handRank"], "kickers": best[0]["handKickers"]},
            )
            if cmp > 0:
                best = [cand]
            elif cmp == 0:
                best.append(cand)

        share = pot["amount"] // len(best)
        remainder = pot["amount"] - share * len(best)
        for i, p in enumerate(best):
            amount = share + (remainder if i == 0 else 0)
            p["chips"] += amount
            winnings_map[p["seat"]] = winnings_map.get(p["seat"], 0) + amount

    winners = [{"seat": s, "amountWon": a} for s, a in winnings_map.items()]
    winners.sort(key=lambda w: -w["amountWon"])
    return winners


def _finalize_hand(state):
    for p in state["players"]:
        if p["chips"] == 0 and p["status"] != "eliminated":
            p["status"] = "eliminated"
    state["totalPot"] = 0
    state["actionOnSeat"] = -1
    state["callAmount"] = 0
    state["canCheck"] = False
    state["minRaiseAmount"] = 0
    state["maxRaiseAmount"] = 0

    human = state["players"][HUMAN_SEAT]
    active_bots = [p for p in state["players"] if not p["isHuman"] and p["status"] != "eliminated"]
    if human["status"] == "eliminated" or len(active_bots) == 0:
        state["phase"] = "game_over"
    else:
        state["phase"] = "hand_complete"


# ── Helpers ───────────────────────────────────────────────────────


def _update_action_info(state):
    player = state["players"][state["actionOnSeat"]]
    to_call = state["currentStreetBet"] - player["currentBet"]

    state["canCheck"] = to_call == 0
    state["callAmount"] = min(to_call, player["chips"])

    min_raise_increment = max(state["lastRaiseSize"], BIG_BLIND)
    min_raise_to = state["currentStreetBet"] + min_raise_increment
    max_raise_to = player["chips"] + player["currentBet"]

    if max_raise_to > state["currentStreetBet"] and player["chips"] > to_call:
        state["minRaiseAmount"] = min(min_raise_to, max_raise_to)
        state["maxRaiseAmount"] = max_raise_to
    else:
        state["minRaiseAmount"] = 0
        state["maxRaiseAmount"] = 0


def _post_blind(player, amount):
    actual = min(amount, player["chips"])
    player["chips"] -= actual
    player["currentBet"] = actual
    if player["chips"] == 0:
        player["status"] = "all_in"


def _next_non_eliminated_seat(players, current_seat):
    seat = (current_seat + 1) % len(players)
    while players[seat]["status"] == "eliminated":
        seat = (seat + 1) % len(players)
    return seat


def _next_active_seat_for_betting(players, current_seat):
    n = len(players)
    seat = (current_seat + 1) % n
    for _ in range(n):
        if players[seat]["status"] == "active":
            return seat
        seat = (seat + 1) % n
    return -1


# ── Public read-only helpers ──────────────────────────────────────


def is_humans_turn(state):
    return state["actionOnSeat"] == HUMAN_SEAT


def is_bots_turn(state):
    return (
        state["actionOnSeat"] >= 0
        and state["actionOnSeat"] != HUMAN_SEAT
        and state["players"][state["actionOnSeat"]]["status"] == "active"
    )


# ── Pot calculator (ported from potCalculator.js) ─────────────────


def calculate_pots(players):
    return _build_pots(players, lambda p: p["currentBet"])


def calculate_showdown_pots(players):
    return _build_pots(players, lambda p: p["totalBetThisHand"])


def _build_pots(players, get_bet):
    active_bets = []
    folded_bets_total = 0

    for p in players:
        if p["status"] == "eliminated":
            continue
        bet = get_bet(p)
        if bet == 0:
            continue
        if p["status"] == "folded":
            folded_bets_total += bet
        else:
            active_bets.append({"seat": p["seat"], "bet": bet})

    if not active_bets:
        return []

    active_bets.sort(key=lambda b: b["bet"])

    unique_levels = sorted({b["bet"] for b in active_bets})

    pots = []
    previous_level = 0
    remaining = list(active_bets)

    for level in unique_levels:
        increment = level - previous_level
        if increment > 0:
            pot_amount = increment * len(remaining)
            eligible_seats = [b["seat"] for b in remaining]
            pots.append({"amount": pot_amount, "eligibleSeats": eligible_seats})
        remaining = [b for b in remaining if b["bet"] > level]
        previous_level = level

    if pots and folded_bets_total > 0:
        pots[0]["amount"] += folded_bets_total

    return pots


def get_total_pot(pots):
    return sum(pot["amount"] for pot in pots)


# ── AlphaZero `Game`-protocol adapter ─────────────────────────────
#
# The pipeline (`alphazero/`) consumes the rules core through this layer.
# Module-level functions satisfy the `Game` protocol from
# `game_interface.py`. The discretized action space and the pot-fraction
# raise sizing decisions live here, not in the rules.

ACTION_SIZE = 7

# Action-index legend (also documented in PLAN/README):
#   0 fold
#   1 check or call (whichever the state permits)
#   2 raise min-legal
#   3 raise to 0.5 × pot
#   4 raise to 1.0 × pot
#   5 raise to 2.0 × pot
#   6 all-in
ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_RAISE_MIN = 2
ACTION_RAISE_HALF_POT = 3
ACTION_RAISE_POT = 4
ACTION_RAISE_TWO_POT = 5
ACTION_ALL_IN = 6

_POT_FRACTIONS = {
    ACTION_RAISE_HALF_POT: 0.5,
    ACTION_RAISE_POT: 1.0,
    ACTION_RAISE_TWO_POT: 2.0,
}


def current_seat(state):
    return state["actionOnSeat"]


def is_terminal(state):
    return state["phase"] in ("hand_complete", "game_over")


def _action_for_idx(state, idx):
    """Return the concrete {type, amount?} action dict for `idx`, or None if
    illegal in `state`. Used both by the mask and by application.
    """
    seat = state["actionOnSeat"]
    if seat < 0:
        return None
    actor = state["players"][seat]
    if actor["status"] != "active":
        return None

    to_call = state["currentStreetBet"] - actor["currentBet"]
    can_check = to_call == 0

    if idx == ACTION_FOLD:
        # Folding when no bet is faced is wasteful and not produced by the JS
        # legalActions; we mirror that — fold is only legal when there's
        # something to call.
        return None if can_check else {"type": "fold"}

    if idx == ACTION_CHECK_CALL:
        return {"type": "check"} if can_check else {"type": "call"}

    # Raise actions (idx 2..6) require an actual raise to be legal at all.
    if state["maxRaiseAmount"] <= 0 or state["maxRaiseAmount"] <= state["currentStreetBet"]:
        return None

    lo = state["minRaiseAmount"]
    hi = state["maxRaiseAmount"]

    if idx == ACTION_RAISE_MIN:
        return {"type": "raise", "amount": lo}

    if idx == ACTION_ALL_IN:
        return {"type": "raise", "amount": hi}

    fraction = _POT_FRACTIONS.get(idx)
    if fraction is None:
        return None

    # Pot-sized raise convention: my raise increment over the call equals the
    # pot after I call. raise_to = currentStreetBet + fraction × pot_after_call.
    pot_at_action = state["totalPot"] + sum(p["currentBet"] for p in state["players"])
    pot_after_call = pot_at_action + to_call
    raise_to = state["currentStreetBet"] + int(round(fraction * pot_after_call))
    clamped = max(lo, min(hi, raise_to))
    return {"type": "raise", "amount": clamped}


def legal_action_mask(state):
    return [_action_for_idx(state, i) is not None for i in range(ACTION_SIZE)]


def apply_action_idx(state, idx):
    action = _action_for_idx(state, idx)
    if action is None:
        raise ValueError(f"illegal action idx {idx} in state phase={state['phase']}")
    return apply_action(state, action)


def terminal_payoff(state, seat, start_chips_for_seat):
    """Chip delta this hand for `seat`, normalized to ~[-1, 1] by STARTING_STACK."""
    delta = state["players"][seat]["chips"] - start_chips_for_seat
    v = delta / STARTING_STACK
    return max(-1.0, min(1.0, v))


# ── State encoder ─────────────────────────────────────────────────
#
# Fixed-size float vector, canonicalized to the mover's perspective so one
# shared network plays every seat. Seats are re-indexed so the mover is
# "slot 0", the player to the mover's left is "slot 1", and so on.
#
# Card index: card_idx = (rank - 2) * 4 + suit_index, suit_index ∈
# {h:0, d:1, c:2, s:3}. The browser-side encoder in src/engine/nnGame.js
# must use the same mapping byte-for-byte.

_SUIT_INDEX = {"h": 0, "d": 1, "c": 2, "s": 3}
_NUM_CARDS = 52
_NUM_STATUS = 4  # active, folded, all_in, eliminated
_STATUS_INDEX = {"active": 0, "folded": 1, "all_in": 2, "eliminated": 3}
_PHASE_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# Layout (each block contributes the listed number of dims):
#   my hole cards            2 × 52 = 104
#   opponent hole cards      7 × 2 × 52 = 728
#   community cards          5 × 52 = 260
#   per-seat stacks          8
#   per-seat current bets    8
#   per-seat total bets      8
#   per-seat status          8 × 4 = 32
#   phase                    4
#   position vs button       8
#   pot                      1
#   bet to call              1
#   min raise                1
#   max raise                1
INPUT_SIZE = 1164


def card_index(card):
    return (card["rank"] - 2) * 4 + _SUIT_INDEX[card["suit"]]


def encode(state, mover_seat):
    n = len(state["players"])
    out = [0.0] * INPUT_SIZE
    offset = 0

    # hole cards in slot order: mover's 2 cards, then each opponent's 2 cards
    # in seat-order starting from mover's left.
    for i in range(n):
        p = state["players"][(mover_seat + i) % n]
        if len(p["holeCards"]) >= 1:
            out[offset + card_index(p["holeCards"][0])] = 1.0
        offset += _NUM_CARDS
        if len(p["holeCards"]) >= 2:
            out[offset + card_index(p["holeCards"][1])] = 1.0
        offset += _NUM_CARDS

    # community cards in 5 fixed slots (flop1, flop2, flop3, turn, river)
    for i in range(5):
        if i < len(state["communityCards"]):
            out[offset + card_index(state["communityCards"][i])] = 1.0
        offset += _NUM_CARDS

    # per-seat stacks (slot order: mover first)
    for i in range(n):
        p = state["players"][(mover_seat + i) % n]
        out[offset + i] = p["chips"] / STARTING_STACK
    offset += n

    # per-seat current-street bets
    for i in range(n):
        p = state["players"][(mover_seat + i) % n]
        out[offset + i] = p["currentBet"] / STARTING_STACK
    offset += n

    # per-seat total bet this hand
    for i in range(n):
        p = state["players"][(mover_seat + i) % n]
        out[offset + i] = p["totalBetThisHand"] / STARTING_STACK
    offset += n

    # per-seat status one-hot
    for i in range(n):
        p = state["players"][(mover_seat + i) % n]
        out[offset + i * _NUM_STATUS + _STATUS_INDEX[p["status"]]] = 1.0
    offset += n * _NUM_STATUS

    # phase one-hot
    if state["phase"] in _PHASE_INDEX:
        out[offset + _PHASE_INDEX[state["phase"]]] = 1.0
    offset += len(_PHASE_INDEX)

    # position relative to button: distance from button to mover, in slot order
    distance = (mover_seat - state["dealerSeat"]) % n
    out[offset + distance] = 1.0
    offset += n

    # scalar features
    out[offset] = state["totalPot"] / STARTING_STACK
    offset += 1
    out[offset] = state["callAmount"] / STARTING_STACK
    offset += 1
    out[offset] = state["minRaiseAmount"] / STARTING_STACK
    offset += 1
    out[offset] = state["maxRaiseAmount"] / STARTING_STACK
    offset += 1

    assert offset == INPUT_SIZE, f"encoder layout mismatch: offset={offset}, INPUT_SIZE={INPUT_SIZE}"
    return out
