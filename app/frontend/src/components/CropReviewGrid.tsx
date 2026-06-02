import type { AnalyzerResult, UiRegionPrediction } from "../api/types";

type CropReviewGridProps = {
  result: AnalyzerResult | null;
  selectedRegionKey?: string | null;
  onSelectRegion?: (regionKey: string) => void;
};

function formatPercent(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

function confidenceWidth(value: number): string {
  return `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%`;
}

function regionKey(region: UiRegionPrediction): string {
  return `${region.source_id}-${region.detection_index}`;
}

function strongestRegion(regions: UiRegionPrediction[]): UiRegionPrediction | undefined {
  return regions
    .slice()
    .sort((left, right) => right.foodlens.top_confidence - left.foodlens.top_confidence)[0];
}

function sourceLabel(region: UiRegionPrediction): string | null {
  if (region.source_id.startsWith("video frame ")) {
    const frameLabel = region.source_id.replace("video frame ", "Frame ");
    if (typeof region.sourceTimeSeconds === "number") {
      return `${frameLabel} · ${formatTime(region.sourceTimeSeconds)}`;
    }

    return frameLabel;
  }

  return null;
}

function formatTime(value: number): string {
  return `${Math.round(value * 10) / 10}s`;
}

function statusPillClass(region: UiRegionPrediction): string {
  return region.regionStatusLabel.toLowerCase().includes("fallback")
    ? "crop-status-pill crop-status-pill--fallback"
    : "crop-status-pill";
}

function statusPillLabel(region: UiRegionPrediction, isVideoReview: boolean): string {
  if (isVideoReview && region.regionStatusLabel.toLowerCase().includes("fallback")) {
    return "Frame-level review";
  }

  return region.regionStatusLabel;
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
  const fallbackRegion = strongestRegion(regions);
  const selectedRegion =
    regions.find((region) => regionKey(region) === selectedRegionKey) ?? fallbackRegion;
  const activeSelectedRegionKey = selectedRegion ? regionKey(selectedRegion) : null;
  const balancedGrid = regions.length === 4;
  const isVideoReview = result?.source === "video_review";
  const sectionTitle = isVideoReview ? "Sampled frame regions" : "Detected regions";
  const emptyCopy = isVideoReview
    ? "Sampled video frame cards will appear here."
    : "Detected crop cards will appear here.";
  const reviewClassName = `crop-review${isVideoReview ? " crop-review--video" : ""}`;

  return (
    <section className={reviewClassName} aria-labelledby="crop-review-title">
      <div className="section-heading">
        <p className="eyebrow">Crops</p>
        <h2 id="crop-review-title">{sectionTitle}</h2>
      </div>
      {regions.length === 0 ? (
        <p className="muted-copy">{emptyCopy}</p>
      ) : (
        <div className="crop-review__body">
          <div className={balancedGrid ? "crop-grid crop-grid--balanced" : "crop-grid"}>
            {regions.map((region) => {
              const key = regionKey(region);
              const selected = key === activeSelectedRegionKey;
              const regionSourceLabel = sourceLabel(region);

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
                      {regionSourceLabel ? (
                        <span className="crop-source-pill">{regionSourceLabel}</span>
                      ) : null}
                      <h3>{`Region ${region.displayIndex}: ${region.foodlens.top_label}`}</h3>
                      <div className="crop-confidence">
                        <span>{`${formatPercent(region.foodlens.top_confidence)} confidence`}</span>
                        <div
                          className="crop-confidence__track"
                          aria-label={`Region ${region.displayIndex} confidence ${formatPercent(
                            region.foodlens.top_confidence,
                          )}`}
                        >
                          <span
                            style={{
                              width: confidenceWidth(region.foodlens.top_confidence),
                            }}
                          />
                        </div>
                      </div>
                      <p>{`${region.detectorLabel} · ${region.detectorRoleLabel}`}</p>
                      <span className={statusPillClass(region)}>
                        {statusPillLabel(region, isVideoReview)}
                      </span>
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
              <div className="crop-detail__summary">
                <span className={statusPillClass(selectedRegion)}>
                  {statusPillLabel(selectedRegion, isVideoReview)}
                </span>
                <span>{`${formatPercent(selectedRegion.foodlens.top_confidence)} confidence`}</span>
              </div>
              <div
                className="crop-confidence__track crop-confidence__track--detail"
                aria-label={`Selected crop confidence ${formatPercent(
                  selectedRegion.foodlens.top_confidence,
                )}`}
              >
                <span
                  style={{
                    width: confidenceWidth(selectedRegion.foodlens.top_confidence),
                  }}
                />
              </div>
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
                {sourceLabel(selectedRegion) ? (
                  <div>
                    <dt>Source</dt>
                    <dd>{sourceLabel(selectedRegion)}</dd>
                  </div>
                ) : null}
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
