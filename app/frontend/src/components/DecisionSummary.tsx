import type { AnalyzerResult, DecisionBand } from "../api/types";
import { PredictionRanking } from "./PredictionRanking";

type DecisionSummaryProps = {
  result: AnalyzerResult | null;
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

export function DecisionSummary({ result }: DecisionSummaryProps) {
  if (!result) {
    return (
      <section className="decision-card" aria-label="Decision summary">
        <p className="eyebrow">Decision</p>
        <h2>No input selected</h2>
        <p className="muted-copy">
          Upload a plated dish image or load the sample to review FoodLens crop
          decisions.
        </p>
      </section>
    );
  }

  return (
    <section className="decision-card" aria-label="Decision summary">
      <div className="decision-card__headline">
        <div>
          <p className="eyebrow">Decision</p>
          <h2>{result.strongestLabel}</h2>
        </div>
        <span className={`decision-badge decision-badge--${result.decisionBand}`}>
          {DECISION_LABELS[result.decisionBand]}
        </span>
      </div>
      <p className="confidence-line">
        {formatConfidence(result.strongestConfidence)} confidence
      </p>
      <p className="action-copy">{result.actionCopy}</p>
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
          <dd>{result.detectorStatus}</dd>
        </div>
        <div>
          <dt>Artifacts</dt>
          <dd>{result.artifactStatus}</dd>
        </div>
      </dl>
    </section>
  );
}
