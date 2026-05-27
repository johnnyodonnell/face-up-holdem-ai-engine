export default function HandResult({ winners, players }) {
  if (!winners || winners.length === 0) return null
  return (
    <div className="hand-result">
      {winners.map((w, i) => {
        const player = players[w.seat]
        const verb = player.isHuman ? 'win' : 'wins'
        return (
          <div key={i} className="winner-line">
            {player.name} {verb} ${w.amountWon}
          </div>
        )
      })}
    </div>
  )
}
