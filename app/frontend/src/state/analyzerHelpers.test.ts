import { afterEach, describe, expect, it, vi } from "vitest";

import { createPreviewUrl, sourceHost, videoSampleTimes, waitForEvent } from "./analyzerHelpers";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("videoSampleTimes", () => {
  it("samples at 20%, 50% and 80% of a long duration, spaced apart", () => {
    expect(videoSampleTimes(20)).toEqual([4, 10, 16]);
  });

  it("samples further apart as duration grows, while staying under the cap", () => {
    const cap = 3600 - 0.05;
    const times = videoSampleTimes(3600);

    expect(times).toEqual([720, 1800, 2880]);
    expect(times.every((time) => time <= cap)).toBe(true);
  });

  it("clamps every sample to zero right at the duration - 0.05 boundary", () => {
    // At duration 0.05, the cap (duration - 0.05) is exactly 0, so every
    // raw sample position (which is positive) is clamped down to it.
    expect(videoSampleTimes(0.05)).toEqual([0, 0, 0]);
  });

  it("collapses all three samples to the same clamped time just past the boundary", () => {
    // At duration 0.06 the cap is ~0.01, and even the 20% position (0.012)
    // already exceeds it, so all three samples clamp to the same value.
    const times = videoSampleTimes(0.06);
    const cap = 0.06 - 0.05;

    expect(times).toEqual([cap, cap, cap]);
  });

  it("clamps only the samples that exceed duration - 0.05", () => {
    const times = videoSampleTimes(0.3);

    expect(times[0]).toBeCloseTo(0.06, 10);
    expect(times[1]).toBeCloseTo(0.15, 10);
    expect(times[2]).toBeCloseTo(0.24, 10);
  });

  it("returns a single zero sample for a zero duration", () => {
    expect(videoSampleTimes(0)).toEqual([0]);
  });

  it("returns a single zero sample for a negative duration", () => {
    expect(videoSampleTimes(-5)).toEqual([0]);
  });

  it("returns a single zero sample for a NaN duration", () => {
    expect(videoSampleTimes(Number.NaN)).toEqual([0]);
  });

  it("returns a single zero sample for an infinite duration", () => {
    expect(videoSampleTimes(Number.POSITIVE_INFINITY)).toEqual([0]);
  });
});

describe("sourceHost", () => {
  it("extracts the hostname from a normal URL and strips a leading www.", () => {
    expect(sourceHost("https://www.example.com/path?query=1")).toBe("example.com");
  });

  it("extracts the hostname from a URL that specifies a port", () => {
    expect(sourceHost("https://example.com:8080/path")).toBe("example.com");
  });

  it("keeps a non-www hostname unchanged", () => {
    expect(sourceHost("https://cdn.example.com/video.mp4")).toBe("cdn.example.com");
  });

  it("falls back to the original string for a malformed URL", () => {
    expect(sourceHost("not a url")).toBe("not a url");
  });
});

describe("createPreviewUrl", () => {
  it("returns an object URL when URL.createObjectURL is available", () => {
    const createObjectURL = vi.fn(() => "blob:preview");
    vi.stubGlobal("URL", { ...URL, createObjectURL });

    const file = new File(["data"], "photo.jpg", { type: "image/jpeg" });
    expect(createPreviewUrl(file)).toBe("blob:preview");
    expect(createObjectURL).toHaveBeenCalledWith(file);
  });

  it("returns null when URL.createObjectURL is unavailable", () => {
    vi.stubGlobal("URL", { ...URL, createObjectURL: undefined });

    const file = new File(["data"], "photo.jpg", { type: "image/jpeg" });
    expect(createPreviewUrl(file)).toBeNull();
  });
});

describe("waitForEvent", () => {
  it("resolves with the event once it fires", async () => {
    const target = new EventTarget();
    const pending = waitForEvent(target, "loadedmetadata");

    const event = new Event("loadedmetadata");
    target.dispatchEvent(event);

    await expect(pending).resolves.toBe(event);
  });

  it("rejects when the target emits an error first", async () => {
    const target = new EventTarget();
    const pending = waitForEvent(target, "seeked");

    target.dispatchEvent(new Event("error"));

    await expect(pending).rejects.toThrow("Video failed while waiting for seeked.");
  });
});
