// Pure Face-Up Texas Hold'em rules — no React, no DOM.
// Face-up: every player's hole cards are visible to all. The rules themselves
// are identical to standard No-Limit Hold'em; only the UI differs.

import { evaluateHand, compareHandResults } from './handEvaluator.js'
import { calculateShowdownPots, getTotalPot } from './potCalculator.js'

export const SMALL_BLIND = 1
export const BIG_BLIND = 2
export const STARTING_STACK = 200
export const NUM_SEATS = 8
export const HUMAN_SEAT = 0

const BOT_NAMES = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace']
const SUITS = ['h', 'd', 'c', 's']
const RANK_LABELS = {
  2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
  10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A',
}

// ── Deck ──────────────────────────────────────────────────────────

export function cardLabel(card) {
  return `${RANK_LABELS[card.rank]}${card.suit}`
}

function createDeck() {
  const cards = []
  for (const suit of SUITS) {
    for (let rank = 2; rank <= 14; rank++) {
      cards.push({ rank, suit, id: `${RANK_LABELS[rank]}${suit}` })
    }
  }
  return cards
}

function shuffle(cards) {
  const a = cards.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function deal(deck, count) {
  return { dealt: deck.slice(0, count), remaining: deck.slice(count) }
}

// ── Game / player construction ────────────────────────────────────

function createPlayer(seat, name, isHuman) {
  return {
    seat,
    name,
    isHuman,
    chips: STARTING_STACK,
    status: 'active',
    holeCards: [],
    currentBet: 0,
    totalBetThisHand: 0,
    hasActedThisStreet: false,
    isDealer: false,
    isSB: false,
    isBB: false,
    handRank: undefined,
    handKickers: undefined,
  }
}

export function createGame() {
  const players = [
    createPlayer(0, 'You', true),
    ...BOT_NAMES.map((name, i) => createPlayer(i + 1, name, false)),
  ]
  const initialDealer = Math.floor(Math.random() * NUM_SEATS)
  return {
    handNumber: 0,
    phase: 'waiting',
    deck: [],
    communityCards: [],
    players,
    pots: [],
    totalPot: 0,
    dealerSeat: initialDealer,
    actionOnSeat: -1,
    lastAggressorSeat: -1,
    currentStreetBet: 0,
    lastRaiseSize: BIG_BLIND,
    callAmount: 0,
    canCheck: false,
    minRaiseAmount: 0,
    maxRaiseAmount: 0,
    winners: [],
  }
}

// ── Start a new hand ──────────────────────────────────────────────

export function startNewHand(prevState) {
  const state = structuredClone(prevState)

  for (const p of state.players) {
    p.holeCards = []
    p.handRank = undefined
    p.handKickers = undefined
    p.isDealer = false
    p.isSB = false
    p.isBB = false
    if (p.status !== 'eliminated') {
      p.status = 'active'
      p.currentBet = 0
      p.totalBetThisHand = 0
      p.hasActedThisStreet = false
    }
  }

  state.communityCards = []
  state.pots = []
  state.totalPot = 0
  state.winners = []
  state.handNumber++
  state.lastRaiseSize = BIG_BLIND

  // Advance dealer to next non-eliminated seat.
  // (On hand #1 the initial dealer was already chosen randomly in createGame;
  // we still advance once so the random pick is honored as "previous dealer".)
  state.dealerSeat = nextNonEliminatedSeat(state.players, state.dealerSeat)
  state.players[state.dealerSeat].isDealer = true

  const activeCount = state.players.filter((p) => p.status !== 'eliminated').length

  let sbSeat
  let bbSeat
  if (activeCount === 2) {
    // Heads-up: dealer posts SB
    sbSeat = state.dealerSeat
    bbSeat = nextNonEliminatedSeat(state.players, state.dealerSeat)
  } else {
    sbSeat = nextNonEliminatedSeat(state.players, state.dealerSeat)
    bbSeat = nextNonEliminatedSeat(state.players, sbSeat)
  }

  postBlind(state.players[sbSeat], SMALL_BLIND)
  state.players[sbSeat].isSB = true
  postBlind(state.players[bbSeat], BIG_BLIND)
  state.players[bbSeat].isBB = true

  // Shuffle and deal hole cards
  state.deck = shuffle(createDeck())
  for (const p of state.players) {
    if (p.status !== 'eliminated') {
      const result = deal(state.deck, 2)
      p.holeCards = result.dealt
      state.deck = result.remaining
    }
  }

  state.phase = 'preflop'
  state.currentStreetBet = BIG_BLIND
  state.lastAggressorSeat = bbSeat

  // First to act preflop: first active player after BB
  const firstToAct = nextActiveSeatForBetting(state.players, bbSeat)
  if (firstToAct === -1) {
    return dealRemainingAndShowdown(state)
  }

  state.actionOnSeat = firstToAct
  updateActionInfo(state)
  return state
}

// ── Apply an action (human or bot) ────────────────────────────────

export function applyAction(prevState, action) {
  const state = structuredClone(prevState)
  const seat = state.actionOnSeat
  const player = state.players[seat]

  switch (action.type) {
    case 'fold':
      player.status = 'folded'
      break

    case 'check':
      break

    case 'call': {
      const callAmount = Math.min(
        state.currentStreetBet - player.currentBet,
        player.chips,
      )
      player.chips -= callAmount
      player.currentBet += callAmount
      if (player.chips === 0) player.status = 'all_in'
      break
    }

    case 'raise': {
      // amount = total raise-to
      const raiseTotal = Math.min(action.amount, player.chips + player.currentBet)
      const cost = raiseTotal - player.currentBet
      player.chips -= cost
      state.lastRaiseSize = raiseTotal - state.currentStreetBet
      state.currentStreetBet = raiseTotal
      player.currentBet = raiseTotal
      state.lastAggressorSeat = seat

      // Re-open action for everyone else
      for (const p of state.players) {
        if (p.seat !== seat && p.status === 'active') {
          p.hasActedThisStreet = false
        }
      }
      if (player.chips === 0) player.status = 'all_in'
      break
    }
  }

  player.hasActedThisStreet = true
  return advanceAction(state)
}

// ── Legal actions for the current actor ───────────────────────────

export function legalActions(state) {
  if (state.actionOnSeat < 0) return []
  const player = state.players[state.actionOnSeat]
  if (!player || player.status !== 'active') return []

  const actions = []
  const toCall = state.currentStreetBet - player.currentBet

  if (toCall === 0) {
    actions.push({ type: 'check' })
  } else {
    actions.push({ type: 'fold' })
    actions.push({ type: 'call' })
  }

  if (state.maxRaiseAmount > 0 && state.maxRaiseAmount > state.currentStreetBet) {
    actions.push({ type: 'raise', amount: state.minRaiseAmount })
  }

  return actions
}

// ── Advance / end-of-street / showdown ────────────────────────────

function advanceAction(state) {
  const nonFolded = state.players.filter(
    (p) => p.status !== 'folded' && p.status !== 'eliminated',
  )
  if (nonFolded.length === 1) {
    return awardPotToLastPlayer(state)
  }

  const nextSeat = findNextSeatToAct(state)
  if (nextSeat === -1) {
    return endBettingRound(state)
  }

  state.actionOnSeat = nextSeat
  updateActionInfo(state)
  return state
}

function findNextSeatToAct(state) {
  const n = state.players.length
  let seat = (state.actionOnSeat + 1) % n
  for (let i = 0; i < n; i++) {
    const p = state.players[seat]
    if (p.status === 'active' && !p.hasActedThisStreet) return seat
    seat = (seat + 1) % n
  }
  return -1
}

function endBettingRound(state) {
  collectBetsIntoPot(state)

  const activePlayers = state.players.filter((p) => p.status === 'active')
  const nonFolded = state.players.filter(
    (p) => p.status !== 'folded' && p.status !== 'eliminated',
  )

  // If everyone left is all-in (≤1 still able to bet), run out the board.
  if (activePlayers.length <= 1 && nonFolded.length > 1) {
    return dealRemainingAndShowdown(state)
  }

  const nextP = nextPhase(state.phase)
  state.phase = nextP

  if (nextP === 'showdown') return runShowdown(state)

  dealStreetCards(state)

  state.currentStreetBet = 0
  state.lastAggressorSeat = -1
  state.lastRaiseSize = BIG_BLIND
  for (const p of state.players) p.hasActedThisStreet = false

  const firstToAct = nextActiveSeatForBetting(state.players, state.dealerSeat)
  if (firstToAct === -1) return dealRemainingAndShowdown(state)

  state.actionOnSeat = firstToAct
  updateActionInfo(state)
  return state
}

function collectBetsIntoPot(state) {
  for (const p of state.players) {
    state.totalPot += p.currentBet
    p.totalBetThisHand += p.currentBet
    p.currentBet = 0
  }
}

function nextPhase(current) {
  switch (current) {
    case 'preflop': return 'flop'
    case 'flop': return 'turn'
    case 'turn': return 'river'
    case 'river': return 'showdown'
    default: return 'showdown'
  }
}

function dealStreetCards(state) {
  let cardsToDeal
  switch (state.phase) {
    case 'flop': cardsToDeal = 3; break
    case 'turn':
    case 'river': cardsToDeal = 1; break
    default: return
  }
  // Burn one
  state.deck = state.deck.slice(1)
  const result = deal(state.deck, cardsToDeal)
  state.communityCards = [...state.communityCards, ...result.dealt]
  state.deck = result.remaining
}

function dealRemainingAndShowdown(state) {
  if (state.players.some((p) => p.currentBet > 0)) {
    collectBetsIntoPot(state)
  }
  const needed = 5 - state.communityCards.length
  for (let i = 0; i < needed; i++) {
    state.deck = state.deck.slice(1) // burn
    const result = deal(state.deck, 1)
    state.communityCards = [...state.communityCards, ...result.dealt]
    state.deck = result.remaining
  }
  state.phase = 'showdown'
  return runShowdown(state)
}

function runShowdown(state) {
  if (state.players.some((p) => p.currentBet > 0)) {
    collectBetsIntoPot(state)
  }

  for (const p of state.players) {
    if (p.status === 'active' || p.status === 'all_in') {
      const seven = [...p.holeCards, ...state.communityCards]
      const result = evaluateHand(seven)
      p.handRank = result.handRank
      p.handKickers = result.kickers
    }
  }

  state.pots = calculateShowdownPots(state.players)
  state.totalPot = getTotalPot(state.pots)
  state.winners = awardPots(state)

  finalizeHand(state)
  return state
}

function awardPotToLastPlayer(state) {
  if (state.players.some((p) => p.currentBet > 0)) {
    collectBetsIntoPot(state)
  }
  const winner = state.players.find(
    (p) => p.status !== 'folded' && p.status !== 'eliminated',
  )
  winner.chips += state.totalPot
  state.winners = [{ seat: winner.seat, amountWon: state.totalPot }]
  state.totalPot = 0
  state.pots = []
  finalizeHand(state)
  return state
}

function awardPots(state) {
  const winningsMap = new Map()

  for (const pot of state.pots) {
    const eligible = pot.eligibleSeats
      .map((seat) => state.players[seat])
      .filter((p) => p.status !== 'folded' && p.handRank !== undefined)

    if (eligible.length === 0) continue

    let best = [eligible[0]]
    for (let i = 1; i < eligible.length; i++) {
      const cmp = compareHandResults(
        { handRank: eligible[i].handRank, kickers: eligible[i].handKickers },
        { handRank: best[0].handRank, kickers: best[0].handKickers },
      )
      if (cmp > 0) best = [eligible[i]]
      else if (cmp === 0) best.push(eligible[i])
    }

    const share = Math.floor(pot.amount / best.length)
    const remainder = pot.amount - share * best.length
    for (let i = 0; i < best.length; i++) {
      const p = best[i]
      const amount = share + (i === 0 ? remainder : 0)
      p.chips += amount
      winningsMap.set(p.seat, (winningsMap.get(p.seat) || 0) + amount)
    }
  }

  const winners = []
  for (const [seat, amountWon] of winningsMap) {
    winners.push({ seat, amountWon })
  }
  return winners.sort((a, b) => b.amountWon - a.amountWon)
}

function finalizeHand(state) {
  for (const p of state.players) {
    if (p.chips === 0 && p.status !== 'eliminated') {
      p.status = 'eliminated'
    }
  }
  // Pot has been fully distributed to winners; clear the display total.
  state.totalPot = 0
  state.actionOnSeat = -1
  state.callAmount = 0
  state.canCheck = false
  state.minRaiseAmount = 0
  state.maxRaiseAmount = 0

  const human = state.players[HUMAN_SEAT]
  const activeBots = state.players.filter(
    (p) => !p.isHuman && p.status !== 'eliminated',
  )
  if (human.status === 'eliminated' || activeBots.length === 0) {
    state.phase = 'game_over'
  } else {
    state.phase = 'hand_complete'
  }
}

// ── Helpers ───────────────────────────────────────────────────────

function updateActionInfo(state) {
  const player = state.players[state.actionOnSeat]
  const toCall = state.currentStreetBet - player.currentBet

  state.canCheck = toCall === 0
  state.callAmount = Math.min(toCall, player.chips)

  const minRaiseIncrement = Math.max(state.lastRaiseSize, BIG_BLIND)
  const minRaiseTo = state.currentStreetBet + minRaiseIncrement
  const maxRaiseTo = player.chips + player.currentBet

  if (maxRaiseTo > state.currentStreetBet && player.chips > toCall) {
    state.minRaiseAmount = Math.min(minRaiseTo, maxRaiseTo)
    state.maxRaiseAmount = maxRaiseTo
  } else {
    state.minRaiseAmount = 0
    state.maxRaiseAmount = 0
  }
}

function postBlind(player, amount) {
  const actual = Math.min(amount, player.chips)
  player.chips -= actual
  player.currentBet = actual
  if (player.chips === 0) player.status = 'all_in'
}

function nextNonEliminatedSeat(players, currentSeat) {
  let seat = (currentSeat + 1) % players.length
  while (players[seat].status === 'eliminated') {
    seat = (seat + 1) % players.length
  }
  return seat
}

function nextActiveSeatForBetting(players, currentSeat) {
  const n = players.length
  let seat = (currentSeat + 1) % n
  for (let i = 0; i < n; i++) {
    if (players[seat].status === 'active') return seat
    seat = (seat + 1) % n
  }
  return -1
}

// ── Public read-only helpers ──────────────────────────────────────

export function isHumansTurn(state) {
  return state.actionOnSeat === HUMAN_SEAT
}

export function isBotsTurn(state) {
  return (
    state.actionOnSeat >= 0 &&
    state.actionOnSeat !== HUMAN_SEAT &&
    state.players[state.actionOnSeat]?.status === 'active'
  )
}
