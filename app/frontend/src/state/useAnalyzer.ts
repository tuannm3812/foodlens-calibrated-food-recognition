import { useCallback, useEffect, useRef, useState } from "react";

import {
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
};

function createPreviewUrl(file: File): string | null {
  if (typeof URL.createObjectURL !== "function") {
    return null;
  }

  return URL.createObjectURL(file);
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
  };
}
