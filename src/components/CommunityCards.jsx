import Card from './Card.jsx'

export default function CommunityCards({ cards, phase }) {
  if (cards.length === 0) {
    return (
      <div className="community-cards">
        <div className="community-placeholder">
          {phase === 'preflop' ? 'Preflop betting…' : ''}
        </div>
      </div>
    )
  }
  return (
    <div className="community-cards">
      {cards.map((card, i) => (
        <Card key={i} card={card} />
      ))}
    </div>
  )
}
