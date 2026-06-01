import type { ResultSource } from "../api/types";
import type { AnalyzerStatus } from "../state/useAnalyzer";

const SOURCE_LABELS: Record<ResultSource, string> = {
  live: "Live API",
  backend_fallback: "Backend fallback",
  local_demo: "Local demo",
};

type StatusNoticeProps = {
  status: AnalyzerStatus;
  message: string;
  source?: ResultSource;
};

export function StatusNotice({ status, message, source }: StatusNoticeProps) {
  const label = source ? SOURCE_LABELS[source] : "Workbench";

  return (
    <aside className={`status-notice status-notice--${status}`} aria-live="polite">
      <span className="status-notice__label">{label}</span>
      <span className="status-notice__message">{message}</span>
    </aside>
  );
}
