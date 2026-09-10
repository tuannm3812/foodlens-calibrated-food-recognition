const DEMO_VIDEO_PATH = "/demo/burger-making-demo.mp4";
const DEMO_VIDEO_NAME = "burger-making-demo.mp4";

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

async function fetchDemoVideoFile(): Promise<File> {
  const response = await fetch(DEMO_VIDEO_PATH);
  if (!response.ok) {
    throw new Error(`Demo video returned ${response.status}`);
  }

  const blob = await response.blob();
  return new File([blob], DEMO_VIDEO_NAME, {
    type: blob.type || "video/mp4",
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

export {
  DEMO_VIDEO_NAME,
  createPreviewUrl,
  waitForEvent,
  videoSampleTimes,
  sourceHost,
  fetchDemoVideoFile,
  seekVideo,
  frameToFile,
};
