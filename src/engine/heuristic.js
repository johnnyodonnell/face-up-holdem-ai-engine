// Equity-based heuristic baseline for face-up Hold'em.
//
// Mirror of training/heuristic.py — same LCG, same algorithm, same thresholds.
// When invoked with the same `seed` on the same state, this returns the same
// action as the Python side. Enforced by scripts/parity_heuristic.{mjs,py}.
//
// Roles in the project:
//   1. Cross-check of the rules-port + heuristic algorithm.
//   2. A drop-in opponent for manual testing in the browser (swap the engine
//      import in App.jsx to './engine/heuristic.js').
//
// NOT the production engine. The trained network is the production engine.

import { evaluateHand, compareHandResults } from './handEvaluator.js'

const SUITS = ['h', 'd', 'c', 's']
const RANK_LABELS = {
  2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
  10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A',
}

// Action indices (must match the values defined in games/holdem.py).
export const ACTION_FOLD = 0
export const ACTION_CHECK_CALL = 1
export const ACTION_RAISE_MIN = 2
export const ACTION_RAISE_HALF_POT = 3
export const ACTION_RAISE_POT = 4
export const ACTION_RAISE_TWO_POT = 5
export const ACTION_ALL_IN = 6

// Thresholds — must match training/heuristic.py.
const EQUITY_ALL_IN = 0.90
const EQUITY_RAISE_TWO_POT = 0.80
const EQUITY_RAISE_POT = 0.65
const EQUITY_RAISE_HALF_POT = 0.50

export const DEFAULT_NUM_SAMPLES = 200

// ── Deterministic PRNG (matches Python LCG) ───────────────────────

export function makeLCG(seed) {
  let state = seed >>> 0
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return state / 0x100000000
  }
}

// ── Card-set helpers ──────────────────────────────────────────────

function cardKey(card) {
  return (card.rank - 2) * 4 + SUITS.indexOf(card.suit)
}

function allCards() {
  const cards = []
  for (const suit of SUITS) {
    for (let rank = 2; rank <= 14; rank++) {
      cards.push({ rank, suit, id: `${RANK_LABELS[rank]}${suit}` })
    }
  }
  return cards
}

// ── Monte Carlo equity ────────────────────────────────────────────

export function equity(state, heroSeat, numSamples, rngFn) {
  const hero = state.players[heroSeat]
  if (!hero.holeCards || hero.holeCards.length === 0) return 0.0

  const liveOpps = state.players.filter(
    (p) =>
      p.seat !== heroSeat &&
      (p.status === 'active' || p.status === 'all_in') &&
      p.holeCards.length === 2,
  )
  if (liveOpps.length === 0) return 1.0

  const community = state.communityCards.slice()
  const knownIds = new Set()
  for (const c of hero.holeCards) knownIds.add(c.id)
  for (const p of liveOpps) for (const c of p.holeCards) knownIds.add(c.id)
  for (const c of community) knownIds.add(c.id)

  const remaining = allCards()
    .filter((c) => !knownIds.has(c.id))
    .sort((a, b) => cardKey(a) - cardKey(b))

  const needed = 5 - community.length
  if (needed === 0) {
    return equityOneBoard(hero, liveOpps, community)
  }

  let total = 0.0
  for (let s = 0; s < numSamples; s++) {
    const sampled = sampleWithoutReplacement(remaining, needed, rngFn)
    const board = community.concat(sampled)
    total += equityOneBoard(hero, liveOpps, board)
  }
  return total / numSamples
}

function equityOneBoard(hero, liveOpps, fullBoard) {
  const heroHand = evaluateHand(hero.holeCards.concat(fullBoard))
  const oppHands = liveOpps.map((o) => evaluateHand(o.holeCards.concat(fullBoard)))
  let best = heroHand
  for (const h of oppHands) {
    if (compareHandResults(h, best) > 0) best = h
  }
  let ties = 0
  if (compareHandResults(heroHand, best) === 0) ties += 1
  for (const h of oppHands) {
    if (compareHandResults(h, best) === 0) ties += 1
  }
  if (compareHandResults(heroHand, best) === 0) return 1.0 / ties
  return 0.0
}

function sampleWithoutReplacement(deck, count, rngFn) {
  // Partial Fisher-Yates. Matches Python's _sample_without_replacement.
  const a = deck.slice()
  for (let i = 0; i < count; i++) {
    const r = rngFn()
    const j = i + Math.floor(r * (a.length - i))
    const tmp = a[i]
    a[i] = a[j]
    a[j] = tmp
  }
  return a.slice(0, count)
}

// ── Action selection ──────────────────────────────────────────────

export function chooseActionIdx(state, equityValue) {
  const seat = state.actionOnSeat
  const canRaise =
    state.maxRaiseAmount > 0 && state.maxRaiseAmount > state.currentStreetBet

  const actor = state.players[seat]
  const toCall = state.currentStreetBet - actor.currentBet
  const canCheck = toCall === 0

  if (canRaise) {
    if (equityValue >= EQUITY_ALL_IN) return ACTION_ALL_IN
    if (equityValue >= EQUITY_RAISE_TWO_POT) return ACTION_RAISE_TWO_POT
    if (equityValue >= EQUITY_RAISE_POT) return ACTION_RAISE_POT
    if (equityValue >= EQUITY_RAISE_HALF_POT) return ACTION_RAISE_HALF_POT
  }

  const potBeforeCall =
    state.totalPot + state.players.reduce((s, p) => s + p.currentBet, 0)
  const potAfterCall = potBeforeCall + toCall
  const requiredEquity = toCall === 0 ? 0.0 : toCall / potAfterCall

  if (equityValue >= requiredEquity) return ACTION_CHECK_CALL
  if (canCheck) return ACTION_CHECK_CALL
  return ACTION_FOLD
}

// ── Action-index → concrete action ────────────────────────────────
//
// Mirrors training/games/holdem._action_for_idx. Keep in lock-step.

const POT_FRACTIONS = {
  [ACTION_RAISE_HALF_POT]: 0.5,
  [ACTION_RAISE_POT]: 1.0,
  [ACTION_RAISE_TWO_POT]: 2.0,
}

export function actionForIdx(state, idx) {
  const seat = state.actionOnSeat
  if (seat < 0) return null
  const actor = state.players[seat]
  if (actor.status !== 'active') return null

  const toCall = state.currentStreetBet - actor.currentBet
  const canCheck = toCall === 0

  if (idx === ACTION_FOLD) {
    return canCheck ? null : { type: 'fold' }
  }
  if (idx === ACTION_CHECK_CALL) {
    return canCheck ? { type: 'check' } : { type: 'call' }
  }
  if (state.maxRaiseAmount <= 0 || state.maxRaiseAmount <= state.currentStreetBet) {
    return null
  }
  const lo = state.minRaiseAmount
  const hi = state.maxRaiseAmount

  if (idx === ACTION_RAISE_MIN) return { type: 'raise', amount: lo }
  if (idx === ACTION_ALL_IN) return { type: 'raise', amount: hi }

  const fraction = POT_FRACTIONS[idx]
  if (fraction === undefined) return null

  const potAtAction =
    state.totalPot + state.players.reduce((s, p) => s + p.currentBet, 0)
  const potAfterCall = potAtAction + toCall
  const raiseTo = state.currentStreetBet + Math.round(fraction * potAfterCall)
  const clamped = Math.max(lo, Math.min(hi, raiseTo))
  return { type: 'raise', amount: clamped }
}

// ── Public entry point — matches the project's bestMove(state) contract ──

export function bestMove(state, seed = 1, numSamples = DEFAULT_NUM_SAMPLES) {
  const seat = state.actionOnSeat
  if (seat < 0) return { type: 'check' }
  const rng = makeLCG(seed)
  const eq = equity(state, seat, numSamples, rng)
  const idx = chooseActionIdx(state, eq)
  let action = actionForIdx(state, idx)
  if (action !== null) return action
  // Fallback chain.
  action = actionForIdx(state, ACTION_CHECK_CALL)
  if (action !== null) return action
  return actionForIdx(state, ACTION_FOLD)
}
