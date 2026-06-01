import { CropReviewGrid } from "./CropReviewGrid";
import { DecisionSummary } from "./DecisionSummary";
import { PreviewStage } from "./PreviewStage";
import { StatusNotice } from "./StatusNotice";
import { UploadControls } from "./UploadControls";
import { useAnalyzer } from "../state/useAnalyzer";

export function AnalyzerWorkbench() {
  const analyzer = useAnalyzer();

  return (
    <main className="app-shell">
      <section className="workbench-header">
        <div>
          <p className="eyebrow">FoodLens</p>
          <h1>Analyzer Workbench</h1>
        </div>
        <StatusNotice
          status={analyzer.status}
          message={analyzer.message}
          source={analyzer.result?.source}
        />
      </section>
      <section className="workbench-layout">
        <div className="workbench-primary">
          <PreviewStage previewUrl={analyzer.previewUrl} result={analyzer.result} />
          <UploadControls
            mode={analyzer.mode}
            status={analyzer.status}
            onModeChange={analyzer.setMode}
            onUploadImage={(file) => {
              void analyzer.analyzeImage(file);
            }}
            onSample={analyzer.loadSample}
            onClear={analyzer.clear}
          />
        </div>
        <DecisionSummary result={analyzer.result} />
      </section>
      <CropReviewGrid result={analyzer.result} />
    </main>
  );
}
