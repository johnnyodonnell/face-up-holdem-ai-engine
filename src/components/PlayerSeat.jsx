import Card from './Card.jsx'

export default function PlayerSeat({ player, isActing, isWinner }) {
  const classNames = [
    'player-seat',
    player.isHuman ? 'is-human' : '',
    player.status === 'eliminated' ? 'eliminated' : '',
    player.status === 'folded' ? 'folded' : '',
    isWinner ? 'is-winner' : '',
    isActing ? 'is-acting' : '',
  ]
    .filter(Boolean)
    .join(' ')

  const showCards = player.status !== 'eliminated' && player.holeCards.length > 0

  return (
    <div className={classNames}>
      <div className="player-name">
        {player.name}
        {player.isDealer && <span className="dealer-badge">D</span>}
      </div>
      <div className="player-chips">${player.chips}</div>
      {showCards && (
        <div className="player-cards">
          {player.holeCards.map((card, i) => (
            <Card key={i} card={card} />
          ))}
        </div>
      )}
      {player.status === 'all_in' && (
        <div className="player-status all-in">ALL IN</div>
      )}
      {player.status === 'folded' && (
        <div className="player-status fold">FOLD</div>
      )}
      {player.currentBet > 0 && (
        <div className="player-bet">Bet: ${player.currentBet}</div>
      )}
    </div>
  )
}
