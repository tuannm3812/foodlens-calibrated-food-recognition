import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

function cssRule(selector: string): string {
  const css = readFileSync(resolve(__dirname, "styles.css"), "utf8");
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = css.match(new RegExp(`${escapedSelector}\\s*\\{([^}]*)\\}`));

  return match?.[1] ?? "";
}

describe("decision card visual density", () => {
  it("keeps model metadata compact under top predictions", () => {
    expect(cssRule(".model-metadata")).toContain("gap: 6px");
    expect(cssRule(".model-metadata")).toContain("padding-top: 10px");
    expect(cssRule(".model-metadata")).toContain("grid-template-columns: 1fr");
    expect(cssRule(".model-metadata dt")).toContain("font-size: 0.62rem");
    expect(cssRule(".model-metadata dd")).toContain("font-size: 0.78rem");
  });
});

describe("preview media treatment", () => {
  it("uses a dedicated player surface for video previews", () => {
    expect(cssRule(".preview-stage__frame--video")).toContain("padding: 12px");
    expect(cssRule(".preview-stage__frame--video")).toContain("background: #101412");
    expect(cssRule(".preview-image-layer--video")).toContain("width: min(100%, 960px)");
    expect(cssRule(".preview-image-layer--video")).toContain("max-height: 100%");
  });
});

describe("workbench layout alignment", () => {
  it("uses the same column grid for the title/status row and analysis row", () => {
    expect(cssRule(".app-shell")).toContain(
      "--workbench-columns: minmax(0, 1.45fr) minmax(320px, 0.55fr)",
    );
    expect(cssRule(".workbench-title-row")).toContain(
      "grid-template-columns: var(--workbench-columns)",
    );
    expect(cssRule(".workbench-layout")).toContain(
      "grid-template-columns: var(--workbench-columns)",
    );
  });

  it("aligns crop cards and selected crop details in a shared review body", () => {
    expect(cssRule(".crop-review__body")).toContain(
      "grid-template-columns: minmax(0, 1fr) minmax(260px, 340px)",
    );
    expect(cssRule(".crop-detail")).toContain("align-self: start");
  });

  it("keeps crop review dense and selected details structured", () => {
    expect(cssRule(".crop-grid--balanced")).toContain(
      "grid-template-columns: repeat(2, minmax(0, 1fr))",
    );
    expect(cssRule(".crop-card__media")).toContain("height: clamp(190px, 16vw, 240px)");
    expect(cssRule(".crop-card__media")).toContain("overflow: hidden");
    expect(cssRule(".crop-card__body")).toContain("min-height: 126px");
    expect(cssRule(".crop-card--selected")).toContain(
      "border-color: rgba(45, 90, 39, 0.62)",
    );
    expect(cssRule(".crop-confidence__track")).toContain("height: 6px");
    expect(cssRule(".crop-status-pill")).toContain("border-radius: 999px");
    expect(cssRule(".crop-status-pill--fallback")).toContain("color: var(--lab-gold)");
    expect(cssRule(".crop-source-pill")).toContain("font-size: 0.68rem");
    expect(cssRule(".crop-detail__summary")).toContain("display: flex");
    expect(cssRule(".crop-detail__list div")).toContain(
      "grid-template-columns: 92px minmax(0, 1fr)",
    );
  });

  it("renders runtime readiness as compact status chips", () => {
    expect(cssRule(".runtime-status")).toContain(
      "grid-template-columns: minmax(180px, 1fr) auto",
    );
    expect(cssRule(".runtime-status")).toContain("padding: 9px 12px");
    expect(cssRule(".runtime-status__checks")).toContain("justify-content: flex-end");
    expect(cssRule(".runtime-status__checks span")).toContain("border-radius: 999px");
    expect(cssRule(".runtime-status__checks span")).toContain("font-size: 0.74rem");
  });
});
