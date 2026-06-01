import type { AnalyzerResult, DecisionBand } from "../api/types";
import { PredictionRanking } from "./PredictionRanking";

type DecisionSummaryProps = {
  result: AnalyzerResult | null;
  resultSourceLabel?: string | null;
};

const DECISION_LABELS: Record<DecisionBand, string> = {
  auto_accept: "Auto accept",
  suggest: "Suggest",
  confirm: "Confirm",
  review: "Review",
};

function formatConfidence(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

export function DecisionSummary({
  result,
  resultSourceLabel,
}: DecisionSummaryProps) {
  if (!result) {
    return (
      <section className="decision-card decision-card--empty" aria-label="Decision summary">
        <p className="eyebrow">Decision</p>
        <h2>No input selected</h2>
        <p className="muted-copy">
          Upload a plated dish image, select a short video, or load the sample to
          review FoodLens crop decisions.
        </p>
      </section>
    );
  }

  const confidence = formatConfidence(result.strongestConfidence);
  const regionCount = result.regions.length;
  const regionLabel = `${regionCount} ${regionCount === 1 ? "region" : "regions"} detected`;

  return (
    <section className="decision-card" aria-label="Decision summary">
      <div className="decision-card__topline">
        <div>
          <span className="metric-label">Decision</span>
          <span className={`decision-badge decision-badge--${result.decisionBand}`}>
            {DECISION_LABELS[result.decisionBand]}
          </span>
        </div>
        <div className="confidence-metric">
          <span className="metric-label">Confidence</span>
          <strong>{confidence}</strong>
        </div>
      </div>

      <div className="decision-card__prediction">
        <h2>{result.strongestLabel}</h2>
        <p>{result.actionCopy}</p>
      </div>

      <div className="decision-context" aria-label="Result context">
        <span>{regionLabel}</span>
        {resultSourceLabel ? <span>Source: {resultSourceLabel}</span> : null}
      </div>

      <div className="confidence-track" aria-hidden="true">
        <span style={{ width: confidence }} />
      </div>

      <PredictionRanking predictions={result.topPredictions} />

      <dl className="model-metadata">
        <div>
          <dt>Model</dt>
          <dd>{result.modelName}</dd>
        </div>
        <div>
          <dt>Temperature</dt>
          <dd>{result.temperature.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Detector</dt>
          <dd>{result.detectorStatusLabel}</dd>
        </div>
        <div>
          <dt>Artifacts</dt>
          <dd>{result.artifactStatusLabel}</dd>
        </div>
        {result.fallbackReasonLabel ? (
          <div>
            <dt>Fallback</dt>
            <dd>{result.fallbackReasonLabel}</dd>
          </div>
        ) : null}
      </dl>
    </section>
  );
}
