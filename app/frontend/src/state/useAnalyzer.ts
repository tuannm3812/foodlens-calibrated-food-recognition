import { useCallback, useEffect, useRef, useState } from "react";

import {
  combineFrameResults,
  isUserInputApiError,
  predictMultiFoodImage,
  predictMultiFoodImageUrl,
  predictMultiFoodYoutubeUrl,
  toLocalDemoResult,
} from "../api/foodlensClient";
import type { AnalyzerResult } from "../api/types";
import {
  DEMO_VIDEO_NAME,
  createPreviewUrl,
  fetchDemoVideoFile,
  frameToFile,
  seekVideo,
  sourceHost,
  videoSampleTimes,
  waitForEvent,
} from "./analyzerHelpers";

export type AnalyzerMode = "image" | "video";
export type AnalyzerStatus = "idle" | "loading" | "ready" | "error";

type AnalyzerState = {
  mode: AnalyzerMode;
  previewUrl: string | null;
  status: AnalyzerStatus;
  result: AnalyzerResult | null;
  resultSourceLabel: string | null;
  resultSourceContextLabel: string | null;
  message: string;
  setMode: (mode: AnalyzerMode) => void;
  clear: () => void;
  loadSample: () => Promise<void> | void;
  analyzeImage: (file: File) => Promise<void>;
  analyzeVideo: (file: File) => Promise<void>;
  analyzeImageUrl: (url: string) => Promise<void>;
  analyzeYoutubeUrl: (url: string) => Promise<void>;
};

export function useAnalyzer(): AnalyzerState {
  const [mode, setMode] = useState<AnalyzerMode>("image");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<AnalyzerStatus>("idle");
  const [result, setResult] = useState<AnalyzerResult | null>(null);
  const [resultSourceLabel, setResultSourceLabel] = useState<string | null>(null);
  const [resultSourceContextLabel, setResultSourceContextLabel] = useState<string | null>(
    null,
  );
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

  const replaceExternalPreview = useCallback(
    (nextPreviewUrl: string | null) => {
      revokePreview();
      objectUrlRef.current = null;
      setPreviewUrl(nextPreviewUrl);
    },
    [revokePreview],
  );

  const clear = useCallback(() => {
    requestSequenceRef.current += 1;
    replacePreview(null);
    setStatus("idle");
    setResult(null);
    setResultSourceLabel(null);
    setResultSourceContextLabel(null);
    setMessage("Ready for input");
  }, [replacePreview]);

  const analyzeImage = useCallback(
    async (file: File) => {
      requestSequenceRef.current += 1;
      const requestSequence = requestSequenceRef.current;
      const nextPreviewUrl = createPreviewUrl(file);
      replacePreview(nextPreviewUrl);
      setStatus("loading");
      setResult(null);
      setResultSourceLabel("Uploaded image");
      setResultSourceContextLabel(`Uploaded image · ${file.name}`);
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
    async (
      file: File,
      sourceLabel = "Uploaded video",
      sourceContextLabel = `Uploaded video · ${file.name}`,
    ) => {
      requestSequenceRef.current += 1;
      const requestSequence = requestSequenceRef.current;
      const nextPreviewUrl = createPreviewUrl(file);
      replacePreview(nextPreviewUrl);
      setStatus("loading");
      setResult(null);
      setResultSourceLabel(sourceLabel);
      setResultSourceContextLabel(sourceContextLabel);
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

        const sampleTimes = videoSampleTimes(video.duration);
        const frameFiles: File[] = [];
        for (const [index, sampleTime] of sampleTimes.entries()) {
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

        const nextResult = combineFrameResults(frameResults, sampleTimes);
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

  const analyzeImageUrl = useCallback(
    async (url: string) => {
      requestSequenceRef.current += 1;
      const requestSequence = requestSequenceRef.current;
      replaceExternalPreview(url);
      setStatus("loading");
      setResult(null);
      setResultSourceLabel("Image URL");
      setResultSourceContextLabel(`Image URL · ${sourceHost(url)}`);
      setMessage("Analyzing image URL");

      try {
        const nextResult = await predictMultiFoodImageUrl(url);
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        setResult(nextResult);
        setStatus("ready");
        setMessage("Image URL analysis complete");
      } catch (error) {
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        if (isUserInputApiError(error)) {
          setResult(null);
          setResultSourceLabel(null);
          setResultSourceContextLabel(null);
          setStatus("error");
          setMessage(error.message);
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
    [replaceExternalPreview],
  );

  const analyzeYoutubeUrl = useCallback(
    async (url: string) => {
      requestSequenceRef.current += 1;
      const requestSequence = requestSequenceRef.current;
      replaceExternalPreview(null);
      setStatus("loading");
      setResult(null);
      setResultSourceLabel("YouTube URL");
      setResultSourceContextLabel(`YouTube · ${sourceHost(url)}`);
      setMessage("Downloading YouTube video");

      try {
        const nextResult = await predictMultiFoodYoutubeUrl(url);
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        setResult(nextResult);
        setStatus("ready");
        setMessage("Video URL review complete");
      } catch (error) {
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        if (isUserInputApiError(error)) {
          setResult(null);
          setResultSourceLabel(null);
          setResultSourceContextLabel(null);
          setStatus("error");
          setMessage(error.message);
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
    [replaceExternalPreview],
  );

  const loadSample = useCallback(async () => {
    if (mode !== "video") {
      requestSequenceRef.current += 1;
      replacePreview(null);
      setStatus("ready");
      setResult(toLocalDemoResult());
      setResultSourceLabel("Sample");
      setResultSourceContextLabel("Sample · local demo");
      setMessage("Local demo data loaded");
      return;
    }

    requestSequenceRef.current += 1;
    const requestSequence = requestSequenceRef.current;
    replacePreview(null);
    setStatus("loading");
    setResult(null);
    setResultSourceLabel("Sample video");
    setResultSourceContextLabel(`Sample video · ${DEMO_VIDEO_NAME}`);
    setMessage("Loading sample video");

    try {
      const sampleFile = await fetchDemoVideoFile();
      if (requestSequence !== requestSequenceRef.current) {
        return;
      }
      await analyzeVideo(sampleFile, "Sample video", `Sample video · ${DEMO_VIDEO_NAME}`);
    } catch (error) {
      if (requestSequence !== requestSequenceRef.current) {
        return;
      }
      setResult(toLocalDemoResult());
      setResultSourceLabel("Sample video");
      setResultSourceContextLabel(`Sample video · ${DEMO_VIDEO_NAME}`);
      setStatus("ready");
      setMessage(
        error instanceof Error
          ? `Using local demo fallback: ${error.message}`
          : "Using local demo fallback",
      );
    }
  }, [analyzeVideo, mode, replacePreview]);

  useEffect(() => revokePreview, [revokePreview]);

  return {
    mode,
    previewUrl,
    status,
    result,
    resultSourceLabel,
    resultSourceContextLabel,
    message,
    setMode,
    clear,
    loadSample,
    analyzeImage,
    analyzeVideo,
    analyzeImageUrl,
    analyzeYoutubeUrl,
  };
}
