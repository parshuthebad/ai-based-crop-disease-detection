# AI Based Crop Detection (Crop & Disease Detection)

An end-to-end AI system that detects crop type and crop disease from leaf/field images.

## Problem Statement
Farmers lose 20-40% of yield every year to plant diseases that are detected too late.
Manual scouting is slow, subjective and needs expert agronomists who are not available
in most rural regions. This project builds an **AI based crop detection system** that
takes a simple phone photo of a leaf and returns:

1. The crop / plant species.
2. Whether the plant is healthy or diseased.
3. The disease name with a confidence score.
4. A short treatment / remedy recommendation.

## Tech Stack
| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras (primary), PyTorch (alternative trainer) |
| Image Processing | OpenCV |
| Backend API | FastAPI (Flask variant included) |
| Frontend | React + Vite |
| Hosting | Google Cloud Run (AWS Elastic Beanstalk / ECS alternative) |
| Dataset | PlantVillage (54,305 images, 38 classes) |

## Repository Structure
```
backend/            FastAPI + Flask inference servers
ml/                 Training, preprocessing, evaluation, inference
frontend/           React upload + prediction UI
deployment/         Dockerfile, Cloud Run & AWS deploy scripts
docs/               Problem statement, architecture, dataset notes
```

## Quickstart
```bash
# 1. Train (or download weights into ml/models/)
pip install -r requirements.txt
python ml/train_tensorflow.py --data data/PlantVillage --epochs 15

# 2. Run API
uvicorn backend.main:app --reload --port 8000

# 3. Run UI
cd frontend && npm install && npm run dev
```

## API
`POST /api/predict` - multipart image file -> JSON prediction.
`GET /api/health` - service health.

## Dataset
PlantVillage: https://www.kaggle.com/datasets/emmarex/plantdisease

## License
MIT
