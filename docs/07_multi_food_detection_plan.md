# Multi-Food Detection Plan

## 1. Objective

FoodLens currently uses a **Food-101 image classifier**. It predicts one label
for the whole image or for sampled video frames. This is useful for single-dish
images, but it cannot locate, count, or classify multiple foods in one scene.

The next capability is **multi-food detection**:

- Detect visible food regions in an image or sampled video frame.
- Crop each detected region.
- Classify each crop with the existing FoodLens classifier.
- Return a structured result with one prediction per food region.

## 2. Recommended Direction

The most practical next step is a **detector-plus-classifier pipeline**:

1. Use **YOLO** or **RT-DETR** to propose food/object bounding boxes.
2. Crop each detected region.
3. Run the current **ResNet50 FT-V2 Food-101 classifier** on each crop.
4. Apply the existing calibrated decision layer per crop.
5. Aggregate detections for image-level and video-level summaries.

This lets the project reuse the current champion classifier while adding
localization.

## 3. Why Not Replace Everything Immediately

Food-101 provides class labels but not bounding boxes. Training a full custom
food detector requires a detection dataset with bounding-box or segmentation
annotations. Until that dataset is selected, a pretrained detector is best used
as an exploration tool.

## 4. Notebook Plan

| Notebook | Purpose |
| --- | --- |
| `archive/07_multi_food_detection_exploration.ipynb` | Historical YOLO detection/segmentation exploration for food images and videos. |
| `08_detection_to_foodlens_pipeline.ipynb` | Connect detector crops to the existing FoodLens classifier and decision layer, producing per-food predictions. |

## 5. Success Criteria

- The notebook can process one image with multiple visible foods.
- The notebook exports bounding-box metadata and crop images.
- Each crop receives top-k Food-101 predictions.
- Video mode samples frames and runs the same detection/classification logic.
- The output format can later be consumed by the FoodLens app.

## 6. Future Dataset Options

Potential datasets for a stronger detector phase:

- **UECFOOD-256** for food categories and bounding boxes.
- **FoodSeg103** for food segmentation masks.
- **Open Images food-related classes** for broader detection pretraining.
- A custom labelled set for product-specific dishes.
