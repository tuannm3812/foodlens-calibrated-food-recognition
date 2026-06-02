import type { AnalyzerResult, UiRegionPrediction } from "../api/types";

type CropReviewGridProps = {
  result: AnalyzerResult | null;
  selectedRegionKey?: string | null;
  onSelectRegion?: (regionKey: string) => void;
};

function formatPercent(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

function regionKey(region: UiRegionPrediction): string {
  return `${region.source_id}-${region.detection_index}`;
}

function bboxCopy(region: UiRegionPrediction): string {
  if (region.detector.label === "whole_image") {
    return "Whole image fallback";
  }

  if (!region.bbox) {
    return "No bounding box";
  }

  return `${region.bbox.x1}, ${region.bbox.y1} to ${region.bbox.x2}, ${region.bbox.y2}`;
}

export function CropReviewGrid({
  result,
  selectedRegionKey,
  onSelectRegion,
}: CropReviewGridProps) {
  const regions = result?.regions ?? [];
  const selectedRegion =
    regions.find((region) => regionKey(region) === selectedRegionKey) ?? regions[0];

  return (
    <section className="crop-review" aria-labelledby="crop-review-title">
      <div className="section-heading">
        <p className="eyebrow">Crops</p>
        <h2 id="crop-review-title">Detected regions</h2>
      </div>
      {regions.length === 0 ? (
        <p className="muted-copy">Detected crop cards will appear here.</p>
      ) : (
        <div className="crop-review__body">
          <div className="crop-grid">
            {regions.map((region) => {
              const key = regionKey(region);
              const selected = key === selectedRegionKey;

              return (
                <article
                  key={key}
                  className={`crop-card${selected ? " crop-card--selected" : ""}`}
                >
                  <button
                    type="button"
                    className="crop-card__button"
                    aria-pressed={selected}
                    onClick={() => onSelectRegion?.(key)}
                  >
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
                      <p>{`${region.detectorLabel} · ${region.detectorRoleLabel}`}</p>
                      <p>{region.regionStatusLabel}</p>
                    </div>
                  </button>
                </article>
              );
            })}
          </div>
          {selectedRegion ? (
            <aside className="crop-detail" aria-label="Selected crop details">
              <span className="metric-label">Selected crop</span>
              <strong>{`Region ${selectedRegion.displayIndex}`}</strong>
              <dl className="crop-detail__list">
                <div>
                  <dt>Classifier</dt>
                  <dd>{selectedRegion.foodlens.top_label}</dd>
                </div>
                <div>
                  <dt>Detector</dt>
                  <dd>{selectedRegion.detectorLabel}</dd>
                </div>
                <div>
                  <dt>Review type</dt>
                  <dd>{selectedRegion.detectorRoleLabel}</dd>
                </div>
                <div>
                  <dt>Bounds</dt>
                  <dd>{bboxCopy(selectedRegion)}</dd>
                </div>
              </dl>
            </aside>
          ) : null}
        </div>
      )}
    </section>
  );
}
