import { useCallback, useEffect, useRef, useState } from "react";

import {
  combineFrameResults,
  predictMultiFoodImage,
  toLocalDemoResult,
} from "../api/foodlensClient";
import type { AnalyzerResult } from "../api/types";

export type AnalyzerMode = "image" | "video";
export type AnalyzerStatus = "idle" | "loading" | "ready" | "error";

type AnalyzerState = {
  mode: AnalyzerMode;
  previewUrl: string | null;
  status: AnalyzerStatus;
  result: AnalyzerResult | null;
  message: string;
  setMode: (mode: AnalyzerMode) => void;
  clear: () => void;
  loadSample: () => void;
  analyzeImage: (file: File) => Promise<void>;
  analyzeVideo: (file: File) => Promise<void>;
};

function createPreviewUrl(file: File): string | null {
  if (typeof URL.createObjectURL !== "function") {
    return null;
  }

  return URL.createObjectURL(file);
}

function waitForEvent(target: EventTarget, eventName: string): Promise<Event> {
  return new Promise((resolve, reject) => {
    function cleanup() {
      target.removeEventListener(eventName, handleEvent);
      target.removeEventListener("error", handleError);
    }

    function handleEvent(event: Event) {
      cleanup();
      resolve(event);
    }

    function handleError() {
      cleanup();
      reject(new Error(`Video failed while waiting for ${eventName}.`));
    }

    target.addEventListener(eventName, handleEvent, { once: true });
    target.addEventListener("error", handleError, { once: true });
  });
}

async function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  if (
    Math.abs(video.currentTime - time) < 0.01 &&
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    return;
  }

  const seeked = waitForEvent(video, "seeked");
  video.currentTime = time;
  await seeked;
}

async function frameToFile(
  video: HTMLVideoElement,
  frameIndex: number,
): Promise<File> {
  const canvas = document.createElement("canvas");
  const width = video.videoWidth;
  const height = video.videoHeight;

  if (width <= 0 || height <= 0) {
    throw new Error("Video frame has no drawable dimensions.");
  }

  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas rendering is unavailable.");
  }

  context.drawImage(video, 0, 0, width, height);

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((nextBlob) => {
      if (nextBlob) {
        resolve(nextBlob);
      } else {
        reject(new Error("Video frame export failed."));
      }
    }, "image/jpeg", 0.9);
  });

  return new File([blob], `video-frame-${frameIndex + 1}.jpg`, {
    type: "image/jpeg",
  });
}

function videoSampleTimes(duration: number): number[] {
  if (!Number.isFinite(duration) || duration <= 0) {
    return [0];
  }

  return [0.2, 0.5, 0.8].map((position) =>
    Math.min(Math.max(duration * position, 0), Math.max(duration - 0.05, 0)),
  );
}

export function useAnalyzer(): AnalyzerState {
  const [mode, setMode] = useState<AnalyzerMode>("image");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<AnalyzerStatus>("idle");
  const [result, setResult] = useState<AnalyzerResult | null>(null);
  const [message, setMessage] = useState("Ready for input");
  const objectUrlRef = useRef<string | null>(null);
  const requestSequenceRef = useRef(0);

  const revokePreview = useCallback(() => {
    if (objectUrlRef.current && typeof URL.revokeObjectURL === "function") {
      URL.revokeObjectURL(objectUrlRef.current);
    }
    objectUrlRef.current = null;
  }, []);

  const replacePreview = useCallback(
    (nextPreviewUrl: string | null) => {
      revokePreview();
      objectUrlRef.current = nextPreviewUrl;
      setPreviewUrl(nextPreviewUrl);
    },
    [revokePreview],
  );

  const clear = useCallback(() => {
    requestSequenceRef.current += 1;
    replacePreview(null);
    setStatus("idle");
    setResult(null);
    setMessage("Ready for input");
  }, [replacePreview]);

  const loadSample = useCallback(() => {
    requestSequenceRef.current += 1;
    replacePreview(null);
    setStatus("ready");
    setResult(toLocalDemoResult());
    setMessage("Local demo data loaded");
  }, [replacePreview]);

  const analyzeImage = useCallback(
    async (file: File) => {
      requestSequenceRef.current += 1;
      const requestSequence = requestSequenceRef.current;
      const nextPreviewUrl = createPreviewUrl(file);
      replacePreview(nextPreviewUrl);
      setStatus("loading");
      setResult(null);
      setMessage("Analyzing image");

      try {
        const nextResult = await predictMultiFoodImage(file);
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        setResult(nextResult);
        setStatus("ready");
        setMessage("Analysis complete");
      } catch (error) {
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        setResult(toLocalDemoResult());
        setStatus("ready");
        setMessage(
          error instanceof Error
            ? `Using local demo fallback: ${error.message}`
            : "Using local demo fallback",
        );
      }
    },
    [replacePreview],
  );

  const analyzeVideo = useCallback(
    async (file: File) => {
      requestSequenceRef.current += 1;
      const requestSequence = requestSequenceRef.current;
      const nextPreviewUrl = createPreviewUrl(file);
      replacePreview(nextPreviewUrl);
      setStatus("loading");
      setResult(null);
      setMessage("Sampling video frames");

      try {
        if (!nextPreviewUrl) {
          throw new Error("Video preview URLs are unavailable.");
        }

        const video = document.createElement("video");
        video.muted = true;
        video.playsInline = true;
        video.preload = "metadata";
        video.src = nextPreviewUrl;

        await waitForEvent(video, "loadedmetadata");
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }

        const frameFiles: File[] = [];
        for (const [index, sampleTime] of videoSampleTimes(video.duration).entries()) {
          await seekVideo(video, sampleTime);
          if (requestSequence !== requestSequenceRef.current) {
            return;
          }
          frameFiles.push(await frameToFile(video, index));
        }

        video.removeAttribute("src");
        video.load();

        setMessage("Analyzing sampled frames");
        const frameResults: AnalyzerResult[] = [];
        for (const frameFile of frameFiles) {
          frameResults.push(await predictMultiFoodImage(frameFile));
          if (requestSequence !== requestSequenceRef.current) {
            return;
          }
        }

        const nextResult = combineFrameResults(frameResults);
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        setResult(nextResult);
        setStatus("ready");
        setMessage("Video review complete");
      } catch (error) {
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        setResult(toLocalDemoResult());
        setStatus("ready");
        setMessage(
          error instanceof Error
            ? `Using local demo fallback: ${error.message}`
            : "Using local demo fallback",
        );
      }
    },
    [replacePreview],
  );

  useEffect(() => revokePreview, [revokePreview]);

  return {
    mode,
    previewUrl,
    status,
    result,
    message,
    setMode,
    clear,
    loadSample,
    analyzeImage,
    analyzeVideo,
  };
}
