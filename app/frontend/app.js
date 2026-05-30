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
const previewEmpty = document.querySelector("#previewEmpty");
const imageZone = document.querySelector("#imageZone");
const videoZone = document.querySelector("#videoZone");
const sampleButton = document.querySelector("#sampleButton");
const clearButton = document.querySelector("#clearButton");
const decisionTitle = document.querySelector("#decisionTitle");
const decisionBadge = document.querySelector("#decisionBadge");
const topPrediction = document.querySelector("#topPrediction");
const confidenceValue = document.querySelector("#confidenceValue");
const confidenceFill = document.querySelector("#confidenceFill");
const actionCopy = document.querySelector("#actionCopy");
const predictionList = document.querySelector("#predictionList");
const modeTabs = document.querySelectorAll(".mode-tab");
const decisionTiles = document.querySelectorAll(".decision-tile");

let activeMode = "image";

function formatLabel(label) {
  return label.replaceAll("_", " ");
}

function formatConfidence(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function setPreview(type, source) {
  previewEmpty.style.display = "none";
  imagePreview.style.display = type === "image" ? "block" : "none";
  videoPreview.style.display = type === "video" ? "block" : "none";

  if (type === "image") {
    imagePreview.src = source;
    videoPreview.removeAttribute("src");
  } else {
    videoPreview.src = source;
    imagePreview.removeAttribute("src");
  }
}

function clearPreview() {
  imagePreview.removeAttribute("src");
  videoPreview.removeAttribute("src");
  imagePreview.style.display = "none";
  videoPreview.style.display = "none";
  previewEmpty.style.display = "grid";
}

function renderPredictions(result) {
  const [label, confidence] = result.predictions[0];
  const decision = result.decision;

  decisionTitle.textContent = result.title;
  decisionBadge.textContent = DECISION_LABELS[decision];
  decisionBadge.className = `decision-badge ${BADGE_CLASSES[decision]}`;
  topPrediction.textContent = formatLabel(label);
  confidenceValue.textContent = formatConfidence(confidence);
  confidenceFill.style.width = formatConfidence(confidence);
  actionCopy.textContent = result.action;

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

function normalizeApiResult(apiResult) {
  return {
    decision: apiResult.decision.band,
    title: apiResult.decision.title,
    action: apiResult.decision.recommended_action,
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

function resetResult() {
  decisionTitle.textContent = "Ready";
  decisionBadge.textContent = "Waiting";
  decisionBadge.className = "decision-badge badge-neutral";
  topPrediction.textContent = "No image selected";
  confidenceValue.textContent = "0.00%";
  confidenceFill.style.width = "0%";
  actionCopy.textContent =
    "FoodLens will show calibrated top-k predictions after an input is selected.";
  predictionList.innerHTML = "";
  decisionTiles.forEach((tile) => tile.classList.remove("active"));
}

function setMode(mode) {
  activeMode = mode;
  modeTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  imageZone.classList.toggle("hidden", mode !== "image");
  videoZone.classList.toggle("hidden", mode !== "video");
  clearPreview();
  resetResult();
}

async function handleFile(file, type) {
  if (!file) {
    return;
  }
  const source = URL.createObjectURL(file);
  setPreview(type, source);

  try {
    const apiResult = await predictWithBackend(file, type);
    renderPredictions(apiResult);
  } catch (error) {
    console.warn("Using mock predictions because backend is unavailable.", error);
    renderPredictions(MOCK_RESULTS[type]);
  }
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

imageInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0], "image");
});

videoInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0], "video");
});

sampleButton.addEventListener("click", () => {
  setPreview("image", SAMPLE_IMAGE);
  renderPredictions(MOCK_RESULTS[activeMode]);
});

clearButton.addEventListener("click", () => {
  imageInput.value = "";
  videoInput.value = "";
  clearPreview();
  resetResult();
});

resetResult();
