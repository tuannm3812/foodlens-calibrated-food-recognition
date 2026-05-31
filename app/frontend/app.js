const SAMPLE_IMAGE =
  "https://www.meatdistrictco.com.au/wp-content/uploads/2024/08/0O2A0384-1700x660.jpg";
const API_BASE_URL = "http://127.0.0.1:8000";

const MOCK_RESULTS = {
  image: {
    decision: "suggest",
    title: "Show suggestions",
    action: "Show ranked suggestions for user selection.",
    predictions: [
      ["steak", 0.7838],
      ["filet_mignon", 0.1543],
      ["prime_rib", 0.0223],
      ["baby_back_ribs", 0.0102],
      ["pork_chop", 0.0092],
    ],
  },
  video: {
    decision: "confirm",
    title: "Confirm dish",
    action: "Ask the user to confirm because sampled frames are not fully aligned.",
    predictions: [
      ["sushi", 0.6842],
      ["sashimi", 0.2015],
      ["ceviche", 0.0511],
      ["tuna_tartare", 0.0394],
      ["miso_soup", 0.0128],
    ],
  },
};

const MULTI_FOOD_SAMPLE = {
  model: "resnet50_ft_v2",
  temperature: 0.9581114053726196,
  crop_count: 23,
  predictions: [
    {
      source_id: "sample_05_prohibition_table",
      detector: { proposal_role: "serving_container", label: "bowl", confidence: 0.5368 },
      foodlens: {
        top_label: "ravioli",
        top_confidence: 0.972,
        decision_band: "auto_accept",
        top_k_predictions: [["ravioli", 0.972], ["gnocchi", 0.018], ["lasagna", 0.004]],
      },
    },
    {
      source_id: "sample_03_food_market",
      detector: { proposal_role: "serving_container", label: "bowl", confidence: 0.3046 },
      foodlens: {
        top_label: "lasagna",
        top_confidence: 0.92,
        decision_band: "auto_accept",
        top_k_predictions: [["lasagna", 0.92], ["ravioli", 0.033], ["pizza", 0.018]],
      },
    },
    {
      source_id: "sample_01_simplot_table",
      detector: { proposal_role: "serving_container", label: "bowl", confidence: 0.4445 },
      foodlens: {
        top_label: "ramen",
        top_confidence: 0.768,
        decision_band: "suggest",
        top_k_predictions: [["ramen", 0.768], ["pho", 0.034], ["miso_soup", 0.023]],
      },
    },
    {
      source_id: "sample_03_food_market",
      detector: { proposal_role: "serving_container", label: "bowl", confidence: 0.3983 },
      foodlens: {
        top_label: "french_fries",
        top_confidence: 0.752,
        decision_band: "suggest",
        top_k_predictions: [["french_fries", 0.752], ["fish_and_chips", 0.146], ["onion_rings", 0.026]],
      },
    },
    {
      source_id: "sample_02_party_food",
      detector: { proposal_role: "direct_food", label: "cake", confidence: 0.5763 },
      foodlens: {
        top_label: "falafel",
        top_confidence: 0.241,
        decision_band: "confirm",
        top_k_predictions: [["falafel", 0.241], ["donuts", 0.195], ["garlic_bread", 0.112]],
      },
    },
    {
      source_id: "sample_04_orchard_table",
      detector: { proposal_role: "serving_container", label: "bowl", confidence: 0.3118 },
      foodlens: {
        top_label: "pork_chop",
        top_confidence: 0.144,
        decision_band: "confirm",
        top_k_predictions: [["pork_chop", 0.144], ["steak", 0.128], ["prime_rib", 0.097]],
      },
    },
  ],
};

const DECISION_LABELS = {
  auto_accept: "Auto-accept",
  suggest: "Suggest",
  confirm: "Confirm",
  review: "Review",
};

const BADGE_CLASSES = {
  auto_accept: "badge-auto",
  suggest: "badge-suggest",
  confirm: "badge-confirm",
  review: "badge-review",
};

const imageInput = document.querySelector("#imageInput");
const videoInput = document.querySelector("#videoInput");
const imagePreview = document.querySelector("#imagePreview");
const videoPreview = document.querySelector("#videoPreview");
const detectionOverlay = document.querySelector("#detectionOverlay");
const previewEmpty = document.querySelector("#previewEmpty");
const previewHelp = document.querySelector("#previewHelp");
const imageZone = document.querySelector("#imageZone");
const videoZone = document.querySelector("#videoZone");
const fileHint = document.querySelector("#fileHint");
const sampleButton = document.querySelector("#sampleButton");
const clearButton = document.querySelector("#clearButton");
const cameraButton = document.querySelector("#cameraButton");
const notificationButton = document.querySelector("#notificationButton");
const profileButton = document.querySelector("#profileButton");
const decisionTitle = document.querySelector("#decisionTitle");
const decisionBadge = document.querySelector("#decisionBadge");
const topPrediction = document.querySelector("#topPrediction");
const confidenceValue = document.querySelector("#confidenceValue");
const confidenceFill = document.querySelector("#confidenceFill");
const actionCopy = document.querySelector("#actionCopy");
const predictionList = document.querySelector("#predictionList");
const multiFoodPanel = document.querySelector("#multiFoodPanel");
const multiFoodGrid = document.querySelector("#multiFoodGrid");
const multiFoodCount = document.querySelector("#multiFoodCount");
const modelName = document.querySelector("#modelName");
const temperatureValue = document.querySelector("#temperatureValue");
const artifactStatus = document.querySelector("#artifactStatus");
const decisionAction = document.querySelector("#decisionTitle");
const modeTabs = document.querySelectorAll(".mode-tab");
const decisionTiles = document.querySelectorAll(".decision-tile");
const navLinks = document.querySelectorAll(".top-nav a[data-view]");
const brandLink = document.querySelector(".brand");
const analysisLayout = document.querySelector(".analysis-layout");
const decisionStrip = document.querySelector(".decision-strip");
const secondaryView = document.querySelector("#secondaryView");
const pageHeading = document.querySelector(".page-title h1");
const scanMeta = document.querySelector("#scanMeta");

let activeMode = "image";
let noticeTimer;
let currentObjectUrl;
let latestMultiFoodResult;

const FILE_HINTS = {
  image: "Images: JPG, PNG, WebP, HEIC. One clear dish works best.",
  video: "Videos: MP4, MOV, WebM. FoodLens samples key frames and reviews detected crops.",
};

const PREVIEW_HELP = {
  image: "Upload a food image to begin analysis.",
  video: "Upload a short food video to sample key frames.",
};

const SECONDARY_VIEWS = {
  archive: {
    title: "Archive",
    meta: "Prototype demo data",
    copy: "A sample archive view showing how saved predictions and review status could be organized.",
    cards: [
      ["Demo analyses", "4"],
      ["Decision bands", "4 states"],
      ["Review queue", "Prototype"],
    ],
    rows: [
      ["Pho image", "pho", "34.91%", "Confirm"],
      ["Steak sample", "filet mignon", "78.38%", "Suggest"],
      ["Dessert upload", "ice cream", "54.76%", "Suggest"],
      ["Salmon plate", "grilled salmon", "62.51%", "Confirm"],
    ],
  },
  database: {
    title: "Database",
    meta: "Food-101 reference data",
    copy: "A reference view based on the project scope: supported Food-101 labels and model risk concepts.",
    cards: [
      ["Classes", "101 food categories"],
      ["Hard classes", "15 calibrated classes"],
      ["Confusion pairs", "30 monitored pairs"],
    ],
    rows: [
      ["Fine-grained meats", "filet mignon, steak, prime rib", "High", "Review"],
      ["Japanese dishes", "sushi, sashimi, miso soup", "Medium", "Confirm"],
      ["Noodle soups", "pho, ramen, wonton soup", "Medium", "Confirm"],
      ["Desserts", "ice cream, cheesecake, tiramisu", "Low", "Suggest"],
    ],
  },
  settings: {
    title: "Settings",
    meta: "Local prototype settings",
    copy: "Current app-facing defaults for the local FoodLens prototype.",
    cards: [
      ["Model", "ResNet50 FT-V2"],
      ["Temperature", "0.958111"],
      ["Runtime", "Local API"],
    ],
    settings: [
      ["Auto-accept threshold", "95%"],
      ["Suggest band", "75-95%"],
      ["Confirm band", "50-75%"],
      ["Review band", "< 50%"],
    ],
  },
};

function formatLabel(label) {
  return label.replaceAll("_", " ");
}

function formatConfidence(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function showNotice(message) {
  const existingNotice = document.querySelector(".notice");
  if (existingNotice) {
    existingNotice.remove();
  }

  const notice = document.createElement("div");
  notice.className = "notice";
  notice.textContent = message;
  document.body.append(notice);

  window.clearTimeout(noticeTimer);
  noticeTimer = window.setTimeout(() => notice.remove(), 2600);
}

function renderTabRows(view) {
  if (view.settings) {
    return `
      <div class="settings-panel">
        ${view.settings
          .map(
            ([label, value]) => `
              <article class="setting-item">
                <label>${label}</label>
                <strong>${value}</strong>
              </article>
            `,
          )
          .join("")}
      </div>
    `;
  }

  return `
    <div class="tab-table" role="table">
      <div class="tab-row header" role="row">
        <span>Item</span>
        <span>Focus</span>
        <span>Score</span>
        <span>Status</span>
      </div>
      ${view.rows
        .map(
          ([item, focus, score, status]) => `
            <div class="tab-row" role="row">
              <strong>${item}</strong>
              <span>${focus}</span>
              <span>${score}</span>
              <span class="status-pill">${status}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function showView(viewName) {
  navLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.view === viewName);
  });

  if (viewName === "analysis") {
    pageHeading.textContent = "Analysis Result";
    scanMeta.textContent = "FoodLens · ResNet50 FT-V2 · Calibrated";
    analysisLayout.classList.remove("hidden");
    decisionStrip.classList.remove("hidden");
    secondaryView.classList.add("hidden");
    secondaryView.innerHTML = "";
    return;
  }

  const view = SECONDARY_VIEWS[viewName];
  if (!view) {
    return;
  }

  pageHeading.textContent = view.title;
  scanMeta.textContent = view.meta;
  analysisLayout.classList.add("hidden");
  decisionStrip.classList.add("hidden");
  secondaryView.classList.remove("hidden");
  secondaryView.innerHTML = `
    <div class="tab-dashboard">
      <div class="tab-hero">
        <div>
          <h2>${view.title}</h2>
          <p>${view.copy}</p>
        </div>
        <div class="tab-actions">
          <button class="tab-action" type="button">Export</button>
          <button class="tab-action" type="button">Refresh</button>
        </div>
      </div>
      <div class="secondary-grid">
        ${view.cards
          .map(
            ([label, value]) => `
              <article class="secondary-card">
                <span>${label}</span>
                <strong>${value}</strong>
              </article>
            `,
          )
          .join("")}
      </div>
      ${renderTabRows(view)}
    </div>
  `;

  secondaryView.querySelectorAll(".tab-action").forEach((button) => {
    button.addEventListener("click", () => showNotice(`${button.textContent} is ready.`));
  });
}

function setPreview(type, source) {
  previewEmpty.style.display = "none";
  imagePreview.style.display = type === "image" ? "block" : "none";
  videoPreview.style.display = type === "video" ? "block" : "none";
  detectionOverlay.innerHTML = "";

  if (type === "image") {
    imagePreview.src = source;
    videoPreview.removeAttribute("src");
  } else {
    videoPreview.src = source;
    imagePreview.removeAttribute("src");
  }
}

function clearPreview() {
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = undefined;
  }
  imagePreview.removeAttribute("src");
  videoPreview.removeAttribute("src");
  imagePreview.style.display = "none";
  videoPreview.style.display = "none";
  detectionOverlay.innerHTML = "";
  previewEmpty.style.display = "grid";
}

function renderPredictions(result) {
  const [label, confidence] = result.predictions[0];
  const decision = result.decision;

  decisionTitle.textContent = result.title;
  decisionAction.className = `confirm-button action-${decision}`;
  decisionBadge.textContent = DECISION_LABELS[decision];
  decisionBadge.className = `decision-badge ${BADGE_CLASSES[decision]}`;
  topPrediction.textContent = formatLabel(label);
  confidenceValue.textContent = formatConfidence(confidence);
  confidenceFill.style.width = formatConfidence(confidence);
  confidenceFill.className = confidence < 0.4 ? "low" : confidence < 0.75 ? "medium" : "";
  actionCopy.textContent = result.action;
  modelName.textContent = result.modelName || "ResNet50 FT-V2";
  temperatureValue.textContent = result.temperature
    ? Number(result.temperature).toFixed(6)
    : "0.958111";
  artifactStatus.textContent = result.artifactStatus || "Mock";

  predictionList.innerHTML = result.predictions
    .map(
      ([className, score]) => `
        <div class="prediction-row">
          <strong>${formatLabel(className)}</strong>
          <span>${formatConfidence(score)}</span>
        </div>
      `,
    )
    .join("");

  decisionTiles.forEach((tile) => {
    tile.classList.toggle("active", tile.dataset.band === decision);
  });
}

function summarizeMultiFoodResult(appResult) {
  const predictions = appResult.predictions || [];
  const accepted = predictions.filter(
    (prediction) => prediction.foodlens.decision_band === "auto_accept",
  ).length;
  const suggested = predictions.filter(
    (prediction) => prediction.foodlens.decision_band === "suggest",
  ).length;
  const confirmed = predictions.filter(
    (prediction) => prediction.foodlens.decision_band === "confirm",
  ).length;
  const reviewed = predictions.filter(
    (prediction) => prediction.foodlens.decision_band === "review",
  ).length;
  const strongestPrediction = predictions
    .slice()
    .sort((a, b) => b.foodlens.top_confidence - a.foodlens.top_confidence)[0];

  return {
    decision: accepted > 0 ? "suggest" : "confirm",
    title: accepted > 0 ? "Review regions" : "Confirm crops",
    action: `${predictions.length} crops analyzed. ${accepted} auto-accept, ${suggested} suggest, ${confirmed} confirm, ${reviewed} review.`,
    modelName: `${appResult.model || "ResNet50 FT-V2"} · Multi-food`,
    temperature: appResult.temperature,
    artifactStatus: appResult.detector_status || appResult.artifact_status || "JSON ready",
    predictions: strongestPrediction
      ? strongestPrediction.foodlens.top_k_predictions
      : [["no_detection", 0]],
  };
}

function imageDisplayMetrics() {
  const stageRect = imagePreview.parentElement.getBoundingClientRect();
  const naturalWidth = imagePreview.naturalWidth;
  const naturalHeight = imagePreview.naturalHeight;

  if (!naturalWidth || !naturalHeight || !stageRect.width || !stageRect.height) {
    return undefined;
  }

  const imageRatio = naturalWidth / naturalHeight;
  const stageRatio = stageRect.width / stageRect.height;
  const displayWidth = imageRatio > stageRatio ? stageRect.width : stageRect.height * imageRatio;
  const displayHeight = imageRatio > stageRatio ? stageRect.width / imageRatio : stageRect.height;

  return {
    left: (stageRect.width - displayWidth) / 2,
    top: (stageRect.height - displayHeight) / 2,
    width: displayWidth,
    height: displayHeight,
    naturalWidth,
    naturalHeight,
  };
}

function renderDetectionOverlay(appResult) {
  const predictions = (appResult.predictions || [])
    .slice()
    .sort((a, b) => b.foodlens.top_confidence - a.foodlens.top_confidence);
  const metrics = imageDisplayMetrics();
  detectionOverlay.innerHTML = "";

  if (!metrics || predictions.length === 0 || imagePreview.style.display === "none") {
    return;
  }

  detectionOverlay.style.left = `${metrics.left}px`;
  detectionOverlay.style.top = `${metrics.top}px`;
  detectionOverlay.style.width = `${metrics.width}px`;
  detectionOverlay.style.height = `${metrics.height}px`;

  predictions.slice(0, 8).forEach((prediction, index) => {
    if (!prediction.bbox) {
      return;
    }

    const bbox = prediction.bbox;
    const sourceWidth = bbox.source_width || metrics.naturalWidth;
    const sourceHeight = bbox.source_height || metrics.naturalHeight;
    const box = document.createElement("div");

    box.className = "detection-box";
    box.style.left = `${(bbox.x1 / sourceWidth) * 100}%`;
    box.style.top = `${(bbox.y1 / sourceHeight) * 100}%`;
    box.style.width = `${((bbox.x2 - bbox.x1) / sourceWidth) * 100}%`;
    box.style.height = `${((bbox.y2 - bbox.y1) / sourceHeight) * 100}%`;
    box.innerHTML = `<span>${String(index + 1).padStart(2, "0")}</span>`;
    detectionOverlay.append(box);
  });
}

function renderMultiFoodResults(appResult) {
  latestMultiFoodResult = appResult;
  const predictions = (appResult.predictions || [])
    .slice()
    .sort((a, b) => b.foodlens.top_confidence - a.foodlens.top_confidence);

  multiFoodPanel.classList.toggle("hidden", predictions.length === 0);
  multiFoodCount.textContent = `${predictions.length} crops`;
  renderDetectionOverlay(appResult);
  multiFoodGrid.innerHTML = predictions
    .map((prediction, index) => {
      const decision = prediction.foodlens.decision_band;
      const label = formatLabel(prediction.foodlens.top_label);
      const confidence = formatConfidence(prediction.foodlens.top_confidence);
      const detectorLabel = formatLabel(prediction.detector.label);
      const cropPreview = prediction.artifacts?.crop_data_url;
      const topAlternatives = prediction.foodlens.top_k_predictions
        .slice(1, 3)
        .map(([className, score]) => `${formatLabel(className)} ${formatConfidence(score)}`)
        .join(" · ");

      return `
        <article class="multi-food-card">
          <div class="crop-thumb ${cropPreview ? "has-image" : ""}">
            ${cropPreview ? `<img src="${cropPreview}" alt="${label} crop preview">` : ""}
            <span>${String(index + 1).padStart(2, "0")}</span>
            <strong>${label}</strong>
          </div>
          <div class="crop-body">
            <div class="crop-topline">
              <span class="decision-badge ${BADGE_CLASSES[decision]}">${DECISION_LABELS[decision]}</span>
              <strong>${confidence}</strong>
            </div>
            <h3>${label}</h3>
            <p>${prediction.source_id.replaceAll("_", " ")} · region proposal · ${detectorLabel}</p>
            <small>${topAlternatives || "No alternatives"}</small>
          </div>
        </article>
      `;
    })
    .join("");

  renderPredictions(summarizeMultiFoodResult(appResult));
}

function normalizeApiResult(apiResult) {
  return {
    decision: apiResult.decision.band,
    title: apiResult.decision.title,
    action: apiResult.decision.recommended_action,
    modelName: apiResult.model_name,
    temperature: apiResult.temperature,
    artifactStatus: apiResult.artifact_status,
    predictions: apiResult.top_predictions.map((prediction) => [
      prediction.class_name,
      prediction.confidence,
    ]),
  };
}

async function predictWithBackend(file, type) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict/${type}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`FoodLens API returned ${response.status}`);
  }

  return normalizeApiResult(await response.json());
}

async function predictMultiFoodWithBackend(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict/multi-food/image`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`FoodLens multi-food API returned ${response.status}`);
  }

  return response.json();
}

function waitForEvent(element, eventName) {
  return new Promise((resolve, reject) => {
    const onSuccess = () => {
      element.removeEventListener(eventName, onSuccess);
      element.removeEventListener("error", onError);
      resolve();
    };
    const onError = () => {
      element.removeEventListener(eventName, onSuccess);
      element.removeEventListener("error", onError);
      reject(new Error(`Video ${eventName} failed.`));
    };
    element.addEventListener(eventName, onSuccess, { once: true });
    element.addEventListener("error", onError, { once: true });
  });
}

async function seekVideo(video, seconds) {
  video.currentTime = seconds;
  await waitForEvent(video, "seeked");
}

async function frameToBlob(video) {
  const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("Could not extract frame."));
        }
      },
      "image/jpeg",
      0.88,
    );
  });
}

function buildVideoMultiFoodResult(frameResults) {
  const predictions = frameResults.flatMap((frameResult, frameIndex) =>
    (frameResult.result.predictions || []).map((prediction) => ({
      ...prediction,
      source_id: `video frame ${frameIndex + 1} at ${frameResult.timeLabel}`,
    })),
  );
  const firstResult = frameResults[0]?.result || {};
  const liveFrameCount = frameResults.filter(
    (frameResult) => frameResult.result.detector_status === "live_yolo",
  ).length;

  return {
    model: firstResult.model || "resnet50_ft_v2",
    temperature: firstResult.temperature || 0.958111,
    top_k: firstResult.top_k || 5,
    decision_thresholds: firstResult.decision_thresholds || {
      auto_accept: 0.85,
      suggest: 0.5,
    },
    detector_status:
      liveFrameCount > 0
        ? `live_yolo · ${liveFrameCount}/${frameResults.length} frames`
        : "fallback_demo",
    crop_count: predictions.length,
    predictions,
    artifact_status: firstResult.artifact_status || "ready",
  };
}

async function predictVideoMultiFoodFrames(source) {
  const video = document.createElement("video");
  video.src = source;
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";

  await waitForEvent(video, "loadedmetadata");
  const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
  const frameCount = Math.min(3, Math.max(1, Math.ceil(duration / 2)));
  const frameTimes = Array.from({ length: frameCount }, (_, index) => {
    if (frameCount === 1) {
      return Math.min(0.1, duration * 0.5);
    }
    return Math.min(duration - 0.05, (duration * (index + 1)) / (frameCount + 1));
  });

  const frameResults = [];
  for (const [index, frameTime] of frameTimes.entries()) {
    const safeFrameTime = Math.max(0, frameTime);
    actionCopy.textContent = `Analyzing video frame ${index + 1} of ${frameTimes.length}.`;
    await seekVideo(video, safeFrameTime);
    const frameBlob = await frameToBlob(video);
    const frameFile = new File([frameBlob], `foodlens_video_frame_${index + 1}.jpg`, {
      type: "image/jpeg",
    });
    frameResults.push({
      timeLabel: `${safeFrameTime.toFixed(1)}s`,
      result: await predictMultiFoodWithBackend(frameFile),
    });
  }

  return buildVideoMultiFoodResult(frameResults);
}

function resetResult() {
  latestMultiFoodResult = undefined;
  decisionTitle.textContent = "Ready";
  decisionAction.className = "confirm-button";
  decisionBadge.textContent = "Waiting";
  decisionBadge.className = "decision-badge badge-neutral";
  topPrediction.textContent = "No image selected";
  confidenceValue.textContent = "0.00%";
  confidenceFill.style.width = "0%";
  confidenceFill.className = "";
  modelName.textContent = "ResNet50 FT-V2";
  temperatureValue.textContent = "0.958111";
  artifactStatus.textContent = "Ready";
  actionCopy.textContent =
    "FoodLens will show calibrated top-k predictions after an input is selected.";
  predictionList.innerHTML = "";
  multiFoodPanel.classList.add("hidden");
  multiFoodGrid.innerHTML = "";
  multiFoodCount.textContent = "0 crops";
  detectionOverlay.innerHTML = "";
  decisionTiles.forEach((tile) => tile.classList.remove("active"));
}

function setMode(mode) {
  activeMode = mode;
  modeTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  imageZone.classList.toggle("hidden", mode !== "image");
  videoZone.classList.toggle("hidden", mode !== "video");
  fileHint.textContent = FILE_HINTS[mode];
  previewHelp.textContent = PREVIEW_HELP[mode];
  clearPreview();
  resetResult();
}

async function handleFile(file, type) {
  if (!file) {
    return;
  }
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
  }
  const source = URL.createObjectURL(file);
  currentObjectUrl = source;
  setPreview(type, source);
  actionCopy.textContent =
    type === "video"
      ? "Sampling video frames and running FoodLens crop review."
      : "Running FoodLens classification.";
  decisionBadge.textContent = "Analyzing";
  decisionBadge.className = "decision-badge badge-neutral";

  try {
    if (type === "image") {
      const apiResult = await predictMultiFoodWithBackend(file);
      renderMultiFoodResults(apiResult);
    } else {
      const apiResult = await predictVideoMultiFoodFrames(source);
      renderMultiFoodResults(apiResult);
    }
  } catch (error) {
    console.warn("Using mock predictions because backend is unavailable.", error);
    if (type === "image") {
      renderMultiFoodResults(MULTI_FOOD_SAMPLE);
    } else {
      renderPredictions(MOCK_RESULTS[type]);
    }
  }
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

navLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    showView(link.dataset.view);
  });
});

imageInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0], "image");
});

imagePreview.addEventListener("load", () => {
  if (latestMultiFoodResult) {
    renderDetectionOverlay(latestMultiFoodResult);
  }
});

window.addEventListener("resize", () => {
  if (latestMultiFoodResult) {
    renderDetectionOverlay(latestMultiFoodResult);
  }
});

videoInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0], "video");
});

cameraButton.addEventListener("click", () => {
  showView("analysis");
  if (activeMode === "video") {
    videoInput.click();
  } else {
    imageInput.click();
  }
});

notificationButton.addEventListener("click", () => {
  showNotice("No new analysis alerts.");
});

profileButton.addEventListener("click", () => {
  showNotice("Local analyst profile is ready.");
});

sampleButton.addEventListener("click", () => {
  showView("analysis");
  if (activeMode !== "image") {
    setMode("image");
  }
  setPreview("image", SAMPLE_IMAGE);
  renderMultiFoodResults(MULTI_FOOD_SAMPLE);
});

clearButton.addEventListener("click", () => {
  showView("analysis");
  imageInput.value = "";
  videoInput.value = "";
  clearPreview();
  resetResult();
});

decisionAction.addEventListener("click", () => {
  showNotice("Decision captured for this local prototype.");
});

brandLink.addEventListener("click", (event) => {
  event.preventDefault();
  showView("analysis");
});

resetResult();
