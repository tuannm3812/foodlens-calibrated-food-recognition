import { act, fireEvent, render, renderHook, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AnalyzerWorkbench } from "./AnalyzerWorkbench";
import { CropReviewGrid } from "./CropReviewGrid";
import { DecisionSummary } from "./DecisionSummary";
import { PreviewStage } from "./PreviewStage";
import { StatusNotice } from "./StatusNotice";
import { UploadControls } from "./UploadControls";
import type { AnalyzerResult } from "../api/types";
import {
  fetchRuntimeStatus,
  predictMultiFoodImage,
  predictMultiFoodImageUrl,
  predictMultiFoodYoutubeUrl,
} from "../api/foodlensClient";
import { useAnalyzer } from "../state/useAnalyzer";

vi.mock("../api/foodlensClient", () => ({
  combineFrameResults: (results: AnalyzerResult[]) => results[0] ?? createResult("fallback", 0.1),
  fetchRuntimeStatus: vi.fn(async () => ({
    ready: true,
    title: "System ready",
    classifierLabel: "Classifier ready",
    detectorLabel: "Detector ready",
    modeLabel: "Live detector + classifier",
  })),
  predictMultiFoodImage: vi.fn(),
  predictMultiFoodImageUrl: vi.fn(),
  predictMultiFoodYoutubeUrl: vi.fn(),
  toLocalDemoResult: () => createResult("ravioli", 0.972, "local_demo"),
}));

function createResult(
  label: string,
  confidence: number,
  source: AnalyzerResult["source"] = "live",
): AnalyzerResult {
  return {
    modelName: "test-model · Multi-food",
    temperature: 1,
    detectorStatus: source,
    detectorStatusLabel: source === "live" ? "Live detector + classifier" : "Local demo",
    artifactStatus: "mock",
    artifactStatusLabel: "Classifier fallback",
    fallbackReason: source === "local_demo" ? "frontend_local_demo" : undefined,
    fallbackReasonLabel: source === "local_demo" ? "Local demo response" : undefined,
    source,
    strongestLabel: label,
    strongestConfidence: confidence,
    decisionBand: confidence > 0.85 ? "auto_accept" : "suggest",
    actionCopy: "Test action copy.",
    topPredictions: [[label, confidence]],
    regions: [
      {
        source_id: `source-${label}`,
        detection_index: 0,
        displayIndex: 1,
        bbox: {
          x1: 10,
          y1: 10,
          x2: 90,
          y2: 90,
          source_width: 100,
          source_height: 100,
        },
        detector: {
          label: "plate",
          proposal_role: "direct_food",
          confidence: 0.8,
          crop_area_ratio: 0.64,
        },
        foodlens: {
          top_label: label,
          top_confidence: confidence,
          decision_band: confidence > 0.85 ? "auto_accept" : "suggest",
          top_k_predictions: [[label, confidence]],
        },
        artifacts: {
          crop_path: `${label}.jpg`,
          crop_artifact_path: `${label}.jpg`,
          figure_path: `${label}-figure.jpg`,
        },
        regionStatusLabel: "Detector crop",
        detectorLabel: "plate",
        detectorRoleLabel: "Food region",
      },
    ],
  };
}

function installVideoSamplingMocks() {
  const originalCreateElement = document.createElement.bind(document);

  vi.stubGlobal("URL", {
    createObjectURL: vi.fn(() => "blob:sample-video"),
    revokeObjectURL: vi.fn(),
  });

  vi.spyOn(document, "createElement").mockImplementation((tagName, options) => {
    if (tagName === "video") {
      const video = originalCreateElement("video", options) as HTMLVideoElement;
      let currentTime = 0;
      let src = "";

      Object.defineProperties(video, {
        duration: { configurable: true, value: 3 },
        videoWidth: { configurable: true, value: 160 },
        videoHeight: { configurable: true, value: 120 },
        readyState: {
          configurable: true,
          value: HTMLMediaElement.HAVE_CURRENT_DATA,
        },
        load: {
          configurable: true,
          value: vi.fn(),
        },
        currentTime: {
          configurable: true,
          get: () => currentTime,
          set: (nextTime: number) => {
            currentTime = nextTime;
            queueMicrotask(() => video.dispatchEvent(new Event("seeked")));
          },
        },
        src: {
          configurable: true,
          get: () => src,
          set: (nextSrc: string) => {
            src = nextSrc;
            queueMicrotask(() => video.dispatchEvent(new Event("loadedmetadata")));
          },
        },
      });

      return video;
    }

    if (tagName === "canvas") {
      const canvas = originalCreateElement("canvas", options) as HTMLCanvasElement;
      Object.defineProperties(canvas, {
        getContext: {
          configurable: true,
          value: () => ({ drawImage: vi.fn() }),
        },
        toBlob: {
          configurable: true,
          value: (callback: BlobCallback) => {
            callback(new Blob(["frame"], { type: "image/jpeg" }));
          },
        },
      });

      return canvas;
    }

    return originalCreateElement(tagName, options);
  });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, resolve, reject };
}

describe("AnalyzerWorkbench", () => {
  beforeEach(() => {
    vi.mocked(fetchRuntimeStatus).mockClear();
    vi.mocked(predictMultiFoodImage).mockReset();
    vi.mocked(predictMultiFoodImageUrl).mockReset();
    vi.mocked(predictMultiFoodYoutubeUrl).mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("labels aggregated video review results without calling them backend fallback", () => {
    render(
      <StatusNotice
        status="ready"
        message="Video review complete"
        source="video_review"
      />,
    );

    expect(screen.getByText("Result ready")).toBeInTheDocument();
    expect(screen.getByText("Video review")).toBeInTheDocument();
    expect(screen.queryByText("Backend fallback")).not.toBeInTheDocument();
  });

  it("renders the Precision Lab product shell", async () => {
    render(<AnalyzerWorkbench />);
    const decisionPolicy = screen.getByLabelText("Decision policy");

    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "FoodLens home" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Product navigation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyze" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("button", { name: "Review" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Models" })).not.toBeDisabled();
    expect(screen.getByRole("heading", { name: "Food Recognition" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Analysis Result" })).not.toBeInTheDocument();
    expect(screen.getByText("Live API · Image/video upload · Calibrated crop review")).toBeInTheDocument();
    expect(decisionPolicy).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Auto-accept")).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Suggest")).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Confirm")).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Review")).toBeInTheDocument();
    expect(screen.getByText("No input selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sample" })).toBeInTheDocument();
    expect(await screen.findByText("System ready")).toBeInTheDocument();
  });

  it("switches to the review queue tab", async () => {
    const user = userEvent.setup();
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Review" }));

    expect(screen.getByRole("button", { name: "Review" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("heading", { name: "Review Queue" })).toBeInTheDocument();
    expect(screen.getByText("No result ready")).toBeInTheDocument();
  });

  it("shows the current result in the review queue", async () => {
    const user = userEvent.setup();
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Sample" }));
    await user.click(screen.getByRole("button", { name: "Review" }));

    expect(screen.getByRole("heading", { name: "Review Queue" })).toBeInTheDocument();
    expect(screen.getByText("Current result")).toBeInTheDocument();
    expect(screen.getAllByText("ravioli").length).toBeGreaterThan(0);
    expect(screen.getByText("Region 1")).toBeInTheDocument();
  });

  it("switches to the models tab with runtime metadata", async () => {
    const user = userEvent.setup();
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Models" }));

    expect(screen.getByRole("button", { name: "Models" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("heading", { name: "Model Runtime" })).toBeInTheDocument();
    expect(await screen.findByText("Classifier ready")).toBeInTheDocument();
    expect(screen.getByText("Detector ready")).toBeInTheDocument();
    expect(screen.getByText("Live detector + classifier")).toBeInTheDocument();
  });

  it("renders local demo fallback after selecting sample", async () => {
    const user = userEvent.setup();
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Sample" }));

    expect(screen.getAllByText("ravioli").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Local demo").length).toBeGreaterThan(0);
    expect(screen.getByText("Detected regions")).toBeInTheDocument();
  });

  it("keeps the result card hierarchy after loading the sample", async () => {
    const user = userEvent.setup();
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Sample" }));

    expect(screen.getByLabelText("Decision summary")).toHaveClass("decision-card");
    expect(screen.getAllByText("Local demo").length).toBeGreaterThan(0);
    expect(screen.getAllByText("ravioli").length).toBeGreaterThan(0);
    expect(screen.getByText("97.2%")).toBeInTheDocument();
    expect(screen.getByText("Detected regions")).toBeInTheDocument();
    expect(screen.getByText("Decision")).toBeInTheDocument();
    expect(screen.getByText("Confidence")).toBeInTheDocument();
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getAllByText("Detector").length).toBeGreaterThan(0);
  });

  it("selects crop cards and highlights the matching preview box", async () => {
    const user = userEvent.setup();
    const result = createResult("ravioli", 0.91);
    result.regions = [
      result.regions[0],
      {
        ...result.regions[0],
        source_id: "source-ramen",
        displayIndex: 2,
        detection_index: 1,
        foodlens: {
          ...result.regions[0].foodlens,
          top_label: "ramen",
          top_confidence: 0.61,
        },
        detector: {
          ...result.regions[0].detector,
          label: "bowl",
          proposal_role: "serving_container",
        },
        detectorLabel: "bowl",
        detectorRoleLabel: "Serving area",
        regionStatusLabel: "Detector crop",
      },
    ];
    vi.mocked(predictMultiFoodImage).mockResolvedValue(result);
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:food-preview"),
      revokeObjectURL: vi.fn(),
    });
    const file = new File(["image"], "food.jpg", { type: "image/jpeg" });

    render(<AnalyzerWorkbench />);
    await user.upload(screen.getByLabelText("Upload"), file);
    await screen.findAllByText("ravioli");

    await user.click(screen.getByRole("button", { name: /Region 2: ramen/i }));

    const selectedCropDetails = screen.getByLabelText("Selected crop details");
    expect(within(selectedCropDetails).getByText("Selected crop")).toBeInTheDocument();
    expect(within(selectedCropDetails).getByText("Classifier")).toBeInTheDocument();
    expect(within(selectedCropDetails).getByText("ramen")).toBeInTheDocument();
    expect(within(selectedCropDetails).getByText("Detector")).toBeInTheDocument();
    expect(within(selectedCropDetails).getByText("bowl")).toBeInTheDocument();
    expect(within(selectedCropDetails).getByText("Detector crop")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Region 2 confidence 61%"),
    ).toBeInTheDocument();
    expect(
      within(selectedCropDetails).getByLabelText("Selected crop confidence 61%"),
    ).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Region 2: ramen" })).toHaveClass(
      "bbox-overlay--selected",
    );
    expect(
      screen.getByLabelText("Detected regions").querySelector(".crop-review__body"),
    ).toBeInTheDocument();
  });

  it("defaults crop review and preview selection to the strongest region", async () => {
    const user = userEvent.setup();
    const result = createResult("baby_back_ribs", 0.54);
    result.regions = [
      {
        ...result.regions[0],
        source_id: "video frame 1",
        sourceTimeSeconds: 0.6,
        displayIndex: 1,
        foodlens: {
          ...result.regions[0].foodlens,
          top_label: "baby_back_ribs",
          top_confidence: 0.54,
        },
        regionStatusLabel: "Whole image fallback",
        detectorLabel: "Whole image",
        detectorRoleLabel: "Whole image review",
      },
      {
        ...result.regions[0],
        source_id: "video frame 2",
        sourceTimeSeconds: 2.4,
        displayIndex: 2,
        detection_index: 1,
        foodlens: {
          ...result.regions[0].foodlens,
          top_label: "hamburger",
          top_confidence: 0.971,
        },
        detectorLabel: "sandwich",
        detectorRoleLabel: "Food region",
        regionStatusLabel: "Detector crop",
      },
    ];
    result.strongestLabel = "hamburger";
    result.strongestConfidence = 0.971;
    result.source = "video_review";
    vi.mocked(predictMultiFoodImage).mockResolvedValue(result);
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:food-preview"),
      revokeObjectURL: vi.fn(),
    });
    const file = new File(["image"], "food.jpg", { type: "image/jpeg" });

    render(<AnalyzerWorkbench />);
    await user.upload(screen.getByLabelText("Upload"), file);
    await screen.findAllByText("hamburger");

    expect(screen.getByRole("heading", { name: "Sampled frame regions" })).toBeInTheDocument();
    expect(screen.getAllByText("Frame 2 · 2.4s").length).toBeGreaterThan(0);
    const selectedCropDetails = screen.getByLabelText("Selected crop details");
    expect(within(selectedCropDetails).getByText("Region 2")).toBeInTheDocument();
    expect(within(selectedCropDetails).getByText("hamburger")).toBeInTheDocument();
    expect(within(selectedCropDetails).getByText("Frame 2 · 2.4s")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Region 2: hamburger/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("img", { name: "Region 2: hamburger" })).toHaveClass(
      "bbox-overlay--selected",
    );
  });

  it("loads the bundled burger video when sample is selected in video mode", async () => {
    const user = userEvent.setup();
    installVideoSamplingMocks();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(new Blob(["video"], { type: "video/mp4" }))),
    );
    vi.mocked(predictMultiFoodImage).mockResolvedValue(createResult("hamburger", 0.971));

    render(<AnalyzerWorkbench />);
    await user.click(screen.getByRole("button", { name: "Video" }));
    await user.click(screen.getByRole("button", { name: "Sample" }));

    expect(fetch).toHaveBeenCalledWith("/demo/burger-making-demo.mp4");
    expect(await screen.findByText("Video review complete")).toBeInTheDocument();
    expect(screen.getAllByText("hamburger").length).toBeGreaterThan(0);
    expect(screen.getByLabelText("Selected food video")).toBeInTheDocument();
  });

  it("does not render raw detector proposal role tokens", () => {
    const result = createResult("burger", 0.91);
    result.regions[0] = {
      ...result.regions[0],
      detector: {
        ...result.regions[0].detector,
        label: "whole_image",
        proposal_role: "fallback_region",
      },
      regionStatusLabel: "Whole image fallback",
      detectorLabel: "Whole image",
      detectorRoleLabel: "Whole image review",
    };

    render(<CropReviewGrid result={result} />);

    expect(screen.queryByText(/fallback_region/i)).not.toBeInTheDocument();
    const selectedCropDetails = screen.getByLabelText("Selected crop details");
    expect(within(selectedCropDetails).getByText("Review type")).toBeInTheDocument();
    expect(within(selectedCropDetails).getByText("Whole image review")).toBeInTheDocument();
  });

  it("keeps stale image analysis from overwriting the current request", async () => {
    const first = deferred<AnalyzerResult>();
    const second = deferred<AnalyzerResult>();
    vi.mocked(predictMultiFoodImage)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const firstFile = new File(["first"], "first.jpg", { type: "image/jpeg" });
    const secondFile = new File(["second"], "second.jpg", { type: "image/jpeg" });
    const { result } = renderHook(() => useAnalyzer());

    await act(async () => {
      void result.current.analyzeImage(firstFile);
    });
    await act(async () => {
      void result.current.analyzeImage(secondFile);
    });

    await act(async () => {
      second.resolve(createResult("second_upload", 0.91));
      await second.promise;
    });
    expect(result.current.result?.strongestLabel).toBe("second_upload");

    await act(async () => {
      first.resolve(createResult("first_upload", 0.99));
      await first.promise;
    });
    expect(result.current.result?.strongestLabel).toBe("second_upload");
  });

  it("analyzes direct image URLs through analyzer state", async () => {
    vi.mocked(predictMultiFoodImageUrl).mockResolvedValue(createResult("pizza", 0.94));
    const { result } = renderHook(() => useAnalyzer());

    await act(async () => {
      await result.current.analyzeImageUrl("https://example.com/pizza.jpg");
    });

    expect(predictMultiFoodImageUrl).toHaveBeenCalledWith(
      "https://example.com/pizza.jpg",
    );
    expect(result.current.status).toBe("ready");
    expect(result.current.message).toBe("Image URL analysis complete");
    expect(result.current.previewUrl).toBe("https://example.com/pizza.jpg");
    expect(result.current.resultSourceLabel).toBe("Image URL");
    expect(result.current.result?.strongestLabel).toBe("pizza");
  });

  it("analyzes YouTube URLs through analyzer state", async () => {
    vi.mocked(predictMultiFoodYoutubeUrl).mockResolvedValue(
      createResult("hamburger", 0.91),
    );
    const { result } = renderHook(() => useAnalyzer());

    await act(async () => {
      await result.current.analyzeYoutubeUrl(
        "https://www.youtube.com/watch?v=abc123",
      );
    });

    expect(predictMultiFoodYoutubeUrl).toHaveBeenCalledWith(
      "https://www.youtube.com/watch?v=abc123",
    );
    expect(result.current.status).toBe("ready");
    expect(result.current.message).toBe("Video URL review complete");
    expect(result.current.previewUrl).toBeNull();
    expect(result.current.resultSourceLabel).toBe("YouTube URL");
    expect(result.current.result?.strongestLabel).toBe("hamburger");
  });

  it("submits a direct image URL from the workbench controls", async () => {
    const user = userEvent.setup();
    vi.mocked(predictMultiFoodImageUrl).mockResolvedValue(createResult("pizza", 0.94));
    render(<AnalyzerWorkbench />);

    await user.type(
      screen.getByLabelText("Image URL"),
      "https://example.com/pizza.jpg",
    );
    await user.click(screen.getByRole("button", { name: "Analyze URL" }));

    expect(predictMultiFoodImageUrl).toHaveBeenCalledWith(
      "https://example.com/pizza.jpg",
    );
    expect(await screen.findByText("Image URL analysis complete")).toBeInTheDocument();
    expect(screen.getAllByText("pizza").length).toBeGreaterThan(0);
    expect(screen.getByText("1 region detected")).toBeInTheDocument();
    expect(screen.getByText("Source: Image URL")).toBeInTheDocument();
  });

  it("submits a YouTube URL from video mode controls", async () => {
    const user = userEvent.setup();
    vi.mocked(predictMultiFoodYoutubeUrl).mockResolvedValue(
      createResult("hamburger", 0.91),
    );
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Video" }));
    await user.type(
      screen.getByLabelText("YouTube URL"),
      "https://www.youtube.com/watch?v=abc123",
    );
    await user.click(screen.getByRole("button", { name: "Analyze URL" }));

    expect(predictMultiFoodYoutubeUrl).toHaveBeenCalledWith(
      "https://www.youtube.com/watch?v=abc123",
    );
    expect(await screen.findByText("Video URL review complete")).toBeInTheDocument();
    expect(screen.getAllByText("hamburger").length).toBeGreaterThan(0);
  });
});

describe("UploadControls", () => {
  it("groups mode selection and URL entry in one input row", () => {
    const { container } = render(
      <UploadControls
        mode="image"
        status="idle"
        onModeChange={vi.fn()}
        onUploadImage={vi.fn()}
        onVideoSelected={vi.fn()}
        onUrlSubmit={vi.fn()}
        onSample={vi.fn()}
        onClear={vi.fn()}
      />,
    );

    const inputRow = container.querySelector(".upload-controls__input-row");
    const actionRow = container.querySelector(".upload-controls__action-row");

    expect(inputRow).toContainElement(screen.getByRole("group", { name: "Input mode" }));
    expect(inputRow).toContainElement(screen.getByRole("form", { name: "URL analysis" }));
    expect(actionRow).toContainElement(screen.getByRole("button", { name: "Sample" }));
  });

  it("routes selected video files to the video upload callback", async () => {
    const user = userEvent.setup();
    const onUploadImage = vi.fn();
    const onVideoSelected = vi.fn();
    const file = new File(["video"], "demo.mp4", { type: "video/mp4" });

    render(
      <UploadControls
        mode="video"
        status="idle"
        onModeChange={vi.fn()}
        onUploadImage={onUploadImage}
        onVideoSelected={onVideoSelected}
        onUrlSubmit={vi.fn()}
        onSample={vi.fn()}
        onClear={vi.fn()}
      />,
    );

    const input = screen.getByLabelText("Upload") as HTMLInputElement;
    expect(input).toHaveAttribute("accept", "video/*");
    expect(input).not.toBeDisabled();

    await user.upload(input, file);

    expect(onVideoSelected).toHaveBeenCalledWith(file);
    expect(onUploadImage).not.toHaveBeenCalled();
  });

  it("submits image URLs from image mode", async () => {
    const user = userEvent.setup();
    const onUrlSubmit = vi.fn();

    render(
      <UploadControls
        mode="image"
        status="idle"
        onModeChange={vi.fn()}
        onUploadImage={vi.fn()}
        onVideoSelected={vi.fn()}
        onUrlSubmit={onUrlSubmit}
        onSample={vi.fn()}
        onClear={vi.fn()}
      />,
    );

    await user.type(screen.getByLabelText("Image URL"), "https://example.com/plate.jpg");
    await user.click(screen.getByRole("button", { name: "Analyze URL" }));

    expect(onUrlSubmit).toHaveBeenCalledWith("https://example.com/plate.jpg");
  });

  it("submits YouTube URLs from video mode", async () => {
    const user = userEvent.setup();
    const onUrlSubmit = vi.fn();

    render(
      <UploadControls
        mode="video"
        status="idle"
        onModeChange={vi.fn()}
        onUploadImage={vi.fn()}
        onVideoSelected={vi.fn()}
        onUrlSubmit={onUrlSubmit}
        onSample={vi.fn()}
        onClear={vi.fn()}
      />,
    );

    expect(screen.getByPlaceholderText("Paste a YouTube URL")).toBeInTheDocument();

    await user.type(
      screen.getByLabelText("YouTube URL"),
      "https://www.youtube.com/watch?v=abc123",
    );
    await user.click(screen.getByRole("button", { name: "Analyze URL" }));

    expect(onUrlSubmit).toHaveBeenCalledWith(
      "https://www.youtube.com/watch?v=abc123",
    );
  });
});

describe("PreviewStage", () => {
  it("fits tall image overlay planes by height", () => {
    const result = createResult("portrait_source", 0.91);
    result.regions[0].bbox = {
      x1: 20,
      y1: 40,
      x2: 80,
      y2: 160,
      source_width: 100,
      source_height: 200,
    };
    const { container } = render(
      <PreviewStage mode="image" previewUrl="blob:image-preview" result={result} />,
    );

    const layer = container.querySelector(".preview-image-layer") as HTMLElement;
    expect(layer).toHaveStyle({
      aspectRatio: "100 / 200",
      width: "auto",
      height: "100%",
    });
    expect(screen.getByLabelText("Region 1: portrait source")).toHaveTextContent(
      "1 portrait source",
    );
  });

  it("renders video previews with a video element", () => {
    const { container } = render(
      <PreviewStage mode="video" previewUrl="blob:video-preview" result={null} />,
    );

    const frame = container.querySelector(".preview-stage__frame") as HTMLElement;
    const layer = container.querySelector(".preview-image-layer") as HTMLElement;
    const video = screen.getByLabelText("Selected food video") as HTMLVideoElement;

    expect(frame).toHaveClass("preview-stage__frame--video");
    expect(layer).toHaveClass("preview-image-layer--video");
    expect(video.tagName).toBe("VIDEO");
    expect(video).toHaveAttribute("src", "blob:video-preview");
    expect(video).toHaveAttribute("controls");
  });

  it("uses video metadata to preserve the player aspect ratio", () => {
    const { container } = render(
      <PreviewStage mode="video" previewUrl="blob:video-preview" result={null} />,
    );

    const layer = container.querySelector(".preview-image-layer") as HTMLElement;
    const video = screen.getByLabelText("Selected food video") as HTMLVideoElement;
    Object.defineProperties(video, {
      videoWidth: { configurable: true, value: 640 },
      videoHeight: { configurable: true, value: 360 },
    });

    fireEvent.loadedMetadata(video);

    expect(layer).toHaveStyle({
      aspectRatio: "640 / 360",
      width: "100%",
      height: "auto",
    });
  });
});

describe("DecisionSummary", () => {
  it("surfaces video frame context in the decision context row", () => {
    const result = createResult("hamburger", 0.971, "video_review");
    result.detectorStatusLabel = "3 frames · 4 regions · 1 fallback frame";

    render(<DecisionSummary result={result} resultSourceLabel="Uploaded video" />);

    const resultContext = screen.getByLabelText("Result context");
    expect(
      within(resultContext).getByText("3 frames · 4 regions · 1 fallback frame"),
    ).toBeInTheDocument();
    expect(within(resultContext).getByText("Source: Uploaded video")).toBeInTheDocument();
    expect(screen.getByText("Video aggregation")).toBeInTheDocument();
  });
});
