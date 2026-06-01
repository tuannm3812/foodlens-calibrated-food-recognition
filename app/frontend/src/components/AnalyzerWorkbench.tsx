import { CropReviewGrid } from "./CropReviewGrid";
import { DecisionPolicyStrip } from "./DecisionPolicyStrip";
import { DecisionSummary } from "./DecisionSummary";
import { PreviewStage } from "./PreviewStage";
import { ProductHeader } from "./ProductHeader";
import { StatusNotice } from "./StatusNotice";
import { UploadControls } from "./UploadControls";
import { useAnalyzer } from "../state/useAnalyzer";

export function AnalyzerWorkbench() {
  const analyzer = useAnalyzer();

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
        <CropReviewGrid result={analyzer.result} />
      </main>
    </div>
  );
}
