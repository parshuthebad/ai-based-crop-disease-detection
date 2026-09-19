# Architecture

```text
 [React UI]  --upload image-->  [FastAPI /api/predict]
     |                                |
     |                        OpenCV preprocessing
     |                                |
     |                     TensorFlow MobileNetV2 model
     |                                |
     <----- JSON: class, confidence, remedy ------
```

## Components
- **frontend/** React + Vite single page app; drag-and-drop upload, preview, result card.
- **backend/main.py** FastAPI app, CORS enabled, multipart upload, model loaded once at startup.
- **backend/app_flask.py** Equivalent Flask service for teams standardised on Flask.
- **ml/preprocess.py** OpenCV pipeline: decode, resize 224x224, CLAHE, normalise.
- **ml/train_tensorflow.py** Transfer learning on MobileNetV2 with augmentation.
- **ml/train_pytorch.py** ResNet18 alternative trainer.
- **ml/inference.py** Shared predictor used by both servers.

## Deployment
Docker image -> Google Artifact Registry -> Cloud Run (min 0, max 10 instances).
AWS path: same image -> ECR -> ECS Fargate behind an ALB.
