import { useEffect, useMemo, useState } from "react";

import { fetchRuntimeStatus } from "../api/foodlensClient";
import type { RuntimeStatusSummary } from "../api/types";
import { CropReviewGrid } from "./CropReviewGrid";
import { DecisionPolicyStrip } from "./DecisionPolicyStrip";
import { DecisionSummary } from "./DecisionSummary";
import { PreviewStage } from "./PreviewStage";
import { ProductHeader, type ProductView } from "./ProductHeader";
import { RuntimeStatusPanel } from "./RuntimeStatusPanel";
import { StatusNotice } from "./StatusNotice";
import { UploadControls } from "./UploadControls";
import { useAnalyzer } from "../state/useAnalyzer";
import type { AnalyzerResult } from "../api/types";

const VIEW_COPY: Record<
  ProductView,
  { eyebrow: string; title: string; subtitle: string; overviewLabel: string }
> = {
  analyze: {
    eyebrow: "Workspace",
    title: "Food Recognition",
    subtitle: "Live API · Image/video upload · Calibrated crop review",
    overviewLabel: "Food recognition overview",
  },
  review: {
    eyebrow: "Review",
    title: "Review Queue",
    subtitle: "Confirm predictions and inspect crop-level evidence",
    overviewLabel: "Review overview",
  },
  models: {
    eyebrow: "Models",
    title: "Model Runtime",
    subtitle: "Classifier, detector, artifact, and fallback status",
    overviewLabel: "Model overview",
  },
};

const DECISION_COPY: Record<AnalyzerResult["decisionBand"], string> = {
  auto_accept: "Auto accept",
  suggest: "Suggest",
  confirm: "Confirm",
  review: "Review",
};

function regionKey(region: AnalyzerResult["regions"][number]): string {
  return `${region.source_id}-${region.detection_index}`;
}

function defaultRegionKey(result: AnalyzerResult | null): string | null {
  const strongestRegion = result?.regions
    .slice()
    .sort((left, right) => right.foodlens.top_confidence - left.foodlens.top_confidence)[0];

  return strongestRegion ? regionKey(strongestRegion) : null;
}

function formatConfidence(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

function detectorMetadata(result: AnalyzerResult): { label: string; details: string | null } {
  if (result.source === "video_review") {
    return {
      label: "Video aggregation",
      details: result.detectorStatusLabel,
    };
  }

  return {
    label: result.detectorStatusLabel,
    details: null,
  };
}

type SharedResultProps = {
  result: AnalyzerResult | null;
  selectedRegionKey: string | null;
  onSelectRegion: (regionKey: string) => void;
};

function AnalyzeView({
  analyzer,
  runtimeStatus,
  selectedRegionKey,
  onSelectRegion,
}: SharedResultProps & {
  analyzer: ReturnType<typeof useAnalyzer>;
  runtimeStatus: RuntimeStatusSummary | null;
}) {
  return (
    <>
      <section className="workbench-layout" aria-label="FoodLens analysis workspace">
        <div className="workbench-primary">
          <PreviewStage
            mode={analyzer.mode}
            previewUrl={analyzer.previewUrl}
            result={analyzer.result}
            selectedRegionKey={selectedRegionKey}
          />
          <UploadControls
            mode={analyzer.mode}
            status={analyzer.status}
            sourceContextLabel={analyzer.resultSourceContextLabel}
            onModeChange={analyzer.setMode}
            onUploadImage={(file) => {
              void analyzer.analyzeImage(file);
            }}
            onVideoSelected={(file) => {
              void analyzer.analyzeVideo(file);
            }}
            onUrlSubmit={(url) => {
              if (analyzer.mode === "video") {
                void analyzer.analyzeYoutubeUrl(url);
                return;
              }
              void analyzer.analyzeImageUrl(url);
            }}
            onSample={analyzer.loadSample}
            onClear={analyzer.clear}
          />
        </div>
        <DecisionSummary
          result={analyzer.result}
          resultSourceLabel={analyzer.resultSourceLabel}
        />
      </section>

      <DecisionPolicyStrip />
      <RuntimeStatusPanel status={runtimeStatus} />
      <CropReviewGrid
        result={analyzer.result}
        selectedRegionKey={selectedRegionKey}
        onSelectRegion={onSelectRegion}
      />
    </>
  );
}

function ReviewQueuePanel({ result }: { result: AnalyzerResult | null }) {
  if (!result) {
    return (
      <section className="secondary-panel" aria-label="Review queue">
        <p className="eyebrow">Queue</p>
        <h2>No result ready</h2>
        <p className="muted-copy">
          Run an image or video analysis to populate the review queue.
        </p>
      </section>
    );
  }

  const detector = detectorMetadata(result);

  return (
    <section className="secondary-panel" aria-label="Review queue">
      <div className="section-heading">
        <p className="eyebrow">Queue</p>
        <h2>Current result</h2>
      </div>
      <dl className="review-summary">
        <div>
          <dt>Prediction</dt>
          <dd>{result.strongestLabel}</dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd>{formatConfidence(result.strongestConfidence)}</dd>
        </div>
        <div>
          <dt>Decision</dt>
          <dd>{DECISION_COPY[result.decisionBand]}</dd>
        </div>
        <div>
          <dt>Detector</dt>
          <dd>{detector.label}</dd>
        </div>
        {detector.details ? (
          <div>
            <dt>Detector details</dt>
            <dd>{detector.details}</dd>
          </div>
        ) : null}
        <div>
          <dt>Artifacts</dt>
          <dd>{result.artifactStatusLabel}</dd>
        </div>
      </dl>
    </section>
  );
}

function ReviewView({ result, selectedRegionKey, onSelectRegion }: SharedResultProps) {
  if (!result) {
    return <ReviewQueuePanel result={result} />;
  }

  return (
    <>
      <ReviewQueuePanel result={result} />
      <CropReviewGrid
        result={result}
        selectedRegionKey={selectedRegionKey}
        onSelectRegion={onSelectRegion}
      />
    </>
  );
}

function ModelsView({
  runtimeStatus,
  result,
}: {
  runtimeStatus: RuntimeStatusSummary | null;
  result: AnalyzerResult | null;
}) {
  const detector = result ? detectorMetadata(result) : null;

  return (
    <section className="models-view" aria-label="Model runtime details">
      <RuntimeStatusPanel status={runtimeStatus} />
      <section className="secondary-panel" aria-label="Model metadata">
        <div className="section-heading">
          <p className="eyebrow">Metadata</p>
          <h2>Current artifact context</h2>
        </div>
        {result ? (
          <dl className="model-metadata model-metadata--standalone">
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
              <dd>{detector?.label}</dd>
            </div>
            {detector?.details ? (
              <div>
                <dt>Detector details</dt>
                <dd>{detector.details}</dd>
              </div>
            ) : null}
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
        ) : (
          <p className="muted-copy">
            Run an analysis to show the model metadata for the current result.
          </p>
        )}
      </section>
    </section>
  );
}

export function AnalyzerWorkbench() {
  const analyzer = useAnalyzer();
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatusSummary | null>(null);
  const [selectedRegionKey, setSelectedRegionKey] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<ProductView>("analyze");
  const fallbackRegionKey = useMemo(
    () => defaultRegionKey(analyzer.result),
    [analyzer.result],
  );
  const selectedRegionExists = useMemo(
    () =>
      Boolean(
        selectedRegionKey &&
          analyzer.result?.regions.some((region) => regionKey(region) === selectedRegionKey),
      ),
    [analyzer.result, selectedRegionKey],
  );
  const activeRegionKey = selectedRegionExists ? selectedRegionKey : fallbackRegionKey;

  useEffect(() => {
    let mounted = true;

    fetchRuntimeStatus()
      .then((status) => {
        if (mounted) {
          setRuntimeStatus(status);
        }
      })
      .catch(() => {
        if (mounted) {
          setRuntimeStatus(null);
        }
      });

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    setSelectedRegionKey(null);
  }, [analyzer.result]);

  const viewCopy = VIEW_COPY[activeView];

  return (
    <div className="app-shell">
      <ProductHeader activeView={activeView} onViewChange={setActiveView} />
      <main className="workbench-shell">
        <section className="workbench-title-row" aria-label={viewCopy.overviewLabel}>
          <div>
            <p className="eyebrow">{viewCopy.eyebrow}</p>
            <h1>{viewCopy.title}</h1>
            <p className="workbench-subtitle">{viewCopy.subtitle}</p>
          </div>
          <StatusNotice
            status={analyzer.status}
            message={analyzer.message}
            source={analyzer.result?.source}
          />
        </section>

        {activeView === "analyze" ? (
          <AnalyzeView
            analyzer={analyzer}
            runtimeStatus={runtimeStatus}
            result={analyzer.result}
            selectedRegionKey={activeRegionKey}
            onSelectRegion={setSelectedRegionKey}
          />
        ) : null}
        {activeView === "review" ? (
          <ReviewView
            result={analyzer.result}
            selectedRegionKey={activeRegionKey}
            onSelectRegion={setSelectedRegionKey}
          />
        ) : null}
        {activeView === "models" ? (
          <ModelsView runtimeStatus={runtimeStatus} result={analyzer.result} />
        ) : null}
      </main>
    </div>
  );
}
