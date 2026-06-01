import { useEffect, useMemo, useState } from "react";

import { fetchRuntimeStatus } from "../api/foodlensClient";
import type { RuntimeStatusSummary } from "../api/types";
import { CropReviewGrid } from "./CropReviewGrid";
import { DecisionPolicyStrip } from "./DecisionPolicyStrip";
import { DecisionSummary } from "./DecisionSummary";
import { PreviewStage } from "./PreviewStage";
import { ProductHeader } from "./ProductHeader";
import { RuntimeStatusPanel } from "./RuntimeStatusPanel";
import { StatusNotice } from "./StatusNotice";
import { UploadControls } from "./UploadControls";
import { useAnalyzer } from "../state/useAnalyzer";

export function AnalyzerWorkbench() {
  const analyzer = useAnalyzer();
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatusSummary | null>(null);
  const [selectedRegionKey, setSelectedRegionKey] = useState<string | null>(null);
  const firstRegionKey = useMemo(() => {
    const firstRegion = analyzer.result?.regions[0];
    if (!firstRegion) {
      return null;
    }

    return `${firstRegion.source_id}-${firstRegion.detection_index}`;
  }, [analyzer.result]);
  const activeRegionKey = selectedRegionKey ?? firstRegionKey;

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

  return (
    <div className="app-shell">
      <ProductHeader />
      <main className="workbench-shell">
        <section className="workbench-title-row" aria-label="Analysis overview">
          <div>
            <p className="eyebrow">Analyze</p>
            <h1>Analysis Result</h1>
            <p className="workbench-subtitle">
              Live API · Image/video upload · Calibrated crop review
            </p>
          </div>
          <StatusNotice
            status={analyzer.status}
            message={analyzer.message}
            source={analyzer.result?.source}
          />
        </section>

        <section className="workbench-layout" aria-label="FoodLens analysis workspace">
          <div className="workbench-primary">
            <PreviewStage
              mode={analyzer.mode}
              previewUrl={analyzer.previewUrl}
              result={analyzer.result}
              selectedRegionKey={activeRegionKey}
            />
            <UploadControls
              mode={analyzer.mode}
              status={analyzer.status}
              onModeChange={analyzer.setMode}
              onUploadImage={(file) => {
                void analyzer.analyzeImage(file);
              }}
              onVideoSelected={(file) => {
                void analyzer.analyzeVideo(file);
              }}
              onSample={analyzer.loadSample}
              onClear={analyzer.clear}
            />
          </div>
          <DecisionSummary result={analyzer.result} />
        </section>

        <DecisionPolicyStrip />
        <RuntimeStatusPanel status={runtimeStatus} />
        <CropReviewGrid
          result={analyzer.result}
          selectedRegionKey={activeRegionKey}
          onSelectRegion={setSelectedRegionKey}
        />
      </main>
    </div>
  );
}
