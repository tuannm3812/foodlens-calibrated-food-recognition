import type { AnalyzerResult } from "../api/types";

type CropReviewGridProps = {
  result: AnalyzerResult | null;
};

function formatPercent(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

export function CropReviewGrid({ result }: CropReviewGridProps) {
  const regions = result?.regions ?? [];

  return (
    <section className="crop-review" aria-labelledby="crop-review-title">
      <div className="section-heading">
        <p className="eyebrow">Crops</p>
        <h2 id="crop-review-title">Detected regions</h2>
      </div>
      {regions.length === 0 ? (
        <p className="muted-copy">Detected crop cards will appear here.</p>
      ) : (
        <div className="crop-review__grid">
          {regions.map((region) => (
            <article key={`${region.source_id}-${region.detection_index}`} className="crop-card">
              <div className="crop-card__media">
                {region.artifacts.crop_data_url ? (
                  <img
                    src={region.artifacts.crop_data_url}
                    alt={`Crop ${region.displayIndex}`}
                  />
                ) : (
                  <span>Crop {region.displayIndex}</span>
                )}
              </div>
              <div className="crop-card__body">
                <h3>{`Region ${region.displayIndex}: ${region.foodlens.top_label}`}</h3>
                <p>{`${formatPercent(region.foodlens.top_confidence)} confidence`}</p>
                <p>{`${region.detector.label} · ${region.detector.proposal_role}`}</p>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
