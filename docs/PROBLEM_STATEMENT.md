# Problem Statement

## Title
AI Based Crop Detection - Automated crop and disease identification from leaf images.

## Background
Agriculture supports over half of the rural workforce in developing economies.
Plant diseases (blight, rust, leaf spot, mildew, mosaic virus) destroy a large share of
production every season. Detection today depends on visual inspection by experts,
which is slow, costly and inconsistent.

## Objective
Build a deep learning system that classifies a crop leaf image into
`<crop>___<condition>` (e.g. `Tomato___Late_blight`) with >95% validation accuracy,
serve it through a REST API, and expose it in a simple web app usable on a phone.

## Scope
- 14 crops, 38 classes (PlantVillage).
- Image input: JPEG/PNG, any resolution, resized to 224x224.
- Output: class, confidence, top-3 alternatives, remedy text.

## Functional Requirements
1. Upload image from browser or mobile camera.
2. Preprocess with OpenCV (resize, denoise, colour-space normalisation).
3. Inference under 1 second per image on CPU.
4. Return confidence and treatment advice.
5. Log every prediction for future retraining.

## Non-Functional Requirements
- Stateless container, horizontally scalable on Cloud Run.
- Model < 30 MB (MobileNetV2 transfer learning).
- HTTPS only, image size capped at 10 MB.

## Success Metrics
| Metric | Target |
|---|---|
| Validation accuracy | >= 95% |
| Macro F1 | >= 0.93 |
| P95 latency | <= 1000 ms |

## Out of Scope
Pest insect detection, yield prediction, soil analysis.
