const SUIT_SYMBOLS = { h: '♥', d: '♦', c: '♣', s: '♠' }
const RANK_DISPLAY = {
  2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
  10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A',
}

export default function Card({ card }) {
  if (!card) return <div className="card placeholder" />
  const isRed = card.suit === 'h' || card.suit === 'd'
  return (
    <div className={`card ${isRed ? 'red' : 'black'}`}>
      <span className="card-rank">{RANK_DISPLAY[card.rank]}</span>
      <span className="card-suit">{SUIT_SYMBOLS[card.suit]}</span>
    </div>
  )
}
