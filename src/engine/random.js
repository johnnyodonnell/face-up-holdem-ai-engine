// The v1 AI engine: pick a uniform-random legal action.
//
// Contract — same shape as fox-lite-ai-engine's engines:
//   bestMove(state) -> a BetAction the current actor can legally take
// Caller must only invoke when it's a bot's turn (actionOnSeat !== HUMAN_SEAT).

import { legalActions } from './game.js'

export function bestMove(state) {
  const legal = legalActions(state)
  if (legal.length === 0) {
    return { type: 'check' }
  }
  const choice = { ...legal[Math.floor(Math.random() * legal.length)] }
  if (choice.type === 'raise') {
    const lo = state.minRaiseAmount
    const hi = state.maxRaiseAmount
    choice.amount = Math.floor(lo + Math.random() * (hi - lo + 1))
  }
  return choice
}
