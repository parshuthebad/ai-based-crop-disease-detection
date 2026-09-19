# Dataset - PlantVillage

- Source: https://www.kaggle.com/datasets/emmarex/plantdisease
- Size: ~54,305 colour leaf images
- Classes: 38 (`<Crop>___<Condition>`) across 14 crops
- Licence: CC0 / public domain research release

## Expected layout
```
data/PlantVillage/
  Tomato___Late_blight/
  Tomato___healthy/
  Potato___Early_blight/
  ...
```

## Download
```bash
pip install kaggle
kaggle datasets download -d emmarex/plantdisease -p data --unzip
```

## Split
80% train / 20% validation, stratified by folder, seed 1337.

## Known limitations
Images are mostly single leaves on a plain background. Field photos with soil,
hands and shadows degrade accuracy, which is why `ml/preprocess.py` applies
CLAHE lighting correction and optional green-mask segmentation, and the training
script uses aggressive augmentation.
