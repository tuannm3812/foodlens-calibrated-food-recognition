import type { ResultSource } from "../api/types";
import type { AnalyzerStatus } from "../state/useAnalyzer";

const SOURCE_LABELS: Record<ResultSource, string> = {
  live: "Live API",
  backend_fallback: "Backend fallback",
  local_demo: "Local demo",
  video_review: "Video review",
};

const STATUS_LABELS: Record<AnalyzerStatus, string> = {
  idle: "Ready",
  loading: "Analyzing",
  ready: "Result ready",
  error: "Needs attention",
};

type StatusNoticeProps = {
  status: AnalyzerStatus;
  message: string;
  source?: ResultSource;
};

export function StatusNotice({ status, message, source }: StatusNoticeProps) {
  const sourceLabel = source ? SOURCE_LABELS[source] : "Workbench";

  return (
    <aside className={`status-notice status-notice--${status}`} aria-live="polite">
      <span className="status-notice__label">{STATUS_LABELS[status]}</span>
      <strong>{sourceLabel}</strong>
      <span className="status-notice__message">{message}</span>
    </aside>
  );
}
