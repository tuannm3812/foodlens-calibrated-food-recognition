type PredictionRankingProps = {
  predictions: Array<[string, number]>;
};

function formatPercent(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

export function PredictionRanking({ predictions }: PredictionRankingProps) {
  return (
    <div className="prediction-ranking">
      <h3>Top predictions</h3>
      <ol className="prediction-ranking__list">
        {predictions.map(([label, confidence], index) => (
          <li key={`${label}-${index}`} className="prediction-ranking__row">
            <span>{`${label} · ${formatPercent(confidence)}`}</span>
            <meter min={0} max={1} value={confidence} aria-label={`${label} confidence`} />
          </li>
        ))}
      </ol>
    </div>
  );
}
