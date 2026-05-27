import { useEffect, useState } from 'react'
import ActionPanel from './components/ActionPanel.jsx'
import CommunityCards from './components/CommunityCards.jsx'
import HandResult from './components/HandResult.jsx'
import PlayerSeat from './components/PlayerSeat.jsx'
import PokerTable from './components/PokerTable.jsx'
import PotDisplay from './components/PotDisplay.jsx'
import {
  HUMAN_SEAT,
  applyAction,
  createGame,
  isBotsTurn,
  isHumansTurn,
  startNewHand,
} from './engine/game.js'
import { bestMove } from './engine/random.js'

export default function App() {
  const [state, setState] = useState(createGame)

  // Bots act instantly whenever it's their turn.
  useEffect(() => {
    if (!isBotsTurn(state)) return
    setState((s) => (isBotsTurn(s) ? applyAction(s, bestMove(s)) : s))
  }, [state])

  // Auto-deal the very first hand on mount.
  useEffect(() => {
    if (state.phase === 'waiting') {
      setState((s) => (s.phase === 'waiting' ? startNewHand(s) : s))
    }
  }, [state.phase])

  function handleHumanAction(action) {
    setState((s) => (isHumansTurn(s) ? applyAction(s, action) : s))
  }
  function handleNextHand() {
    setState((s) => (s.phase === 'hand_complete' ? startNewHand(s) : s))
  }
  function handleNewGame() {
    setState(createGame())
  }

  const human = state.players[HUMAN_SEAT]
  const winnerSeats = new Set(state.winners.map((w) => w.seat))
  const currentStreetBets = state.players.reduce(
    (sum, p) => sum + p.currentBet,
    0,
  )
  const liveTotalPot = state.totalPot + currentStreetBets

  const renderSeat = (seatIndex) => (
    <PlayerSeat
      player={state.players[seatIndex]}
      isWinner={winnerSeats.has(seatIndex)}
      isActing={state.actionOnSeat === seatIndex}
    />
  )

  const centerContent = (
    <>
      <CommunityCards cards={state.communityCards} phase={state.phase} />
      <PotDisplay pots={state.pots} totalPot={liveTotalPot} />
      {state.phase === 'hand_complete' && (
        <HandResult winners={state.winners} players={state.players} />
      )}
      {state.phase === 'game_over' && (
        <HandResult winners={state.winners} players={state.players} />
      )}
    </>
  )

  return (
    <div className="game-table">
      <PokerTable renderSeat={renderSeat} centerContent={centerContent} />

      <ActionPanel
        phase={state.phase}
        humanCanAct={isHumansTurn(state)}
        canCheck={state.canCheck}
        callAmount={state.callAmount}
        minRaiseAmount={state.minRaiseAmount}
        maxRaiseAmount={state.maxRaiseAmount}
        potForSizing={liveTotalPot}
        currentStreetBet={state.currentStreetBet}
        humanCurrentBet={human.currentBet}
        onAction={handleHumanAction}
        onNextHand={handleNextHand}
        onNewGame={handleNewGame}
      />
    </div>
  )
}
