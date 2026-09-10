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

function videoSampleTimes(duration: number): number[] {
  if (!Number.isFinite(duration) || duration <= 0) {
    return [0];
  }

  return [0.2, 0.5, 0.8].map((position) =>
    Math.min(Math.max(duration * position, 0), Math.max(duration - 0.05, 0)),
  );
}

function sourceHost(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

export { createPreviewUrl, waitForEvent, videoSampleTimes, sourceHost };
