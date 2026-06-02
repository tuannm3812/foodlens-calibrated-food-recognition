import type { RuntimeStatusSummary } from "../api/types";

type RuntimeStatusPanelProps = {
  status: RuntimeStatusSummary | null;
};

export function RuntimeStatusPanel({ status }: RuntimeStatusPanelProps) {
  if (!status) {
    return null;
  }

  return (
    <section className="runtime-status" aria-label="Runtime status">
      <div>
        <span className="metric-label">Runtime</span>
        <strong>{status.title}</strong>
      </div>
      <div className="runtime-status__checks">
        <span>{status.classifierLabel}</span>
        <span>{status.detectorLabel}</span>
        <span>{status.modeLabel}</span>
      </div>
    </section>
  );
}
