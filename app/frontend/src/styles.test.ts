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
    expect(cssRule(".model-metadata dt")).toContain("font-size: 0.62rem");
    expect(cssRule(".model-metadata dd")).toContain("font-size: 0.78rem");
  });
});
