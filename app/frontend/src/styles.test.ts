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
});
