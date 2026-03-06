# YOLO-Based Banana Ripeness Detection

## Dataset Structure

### For Classification (train.py):
```
dataset/
├── train/
│   ├── unripe/
│   ├── ripe/
│   └── rotten/
├── val/
│   ├── unripe/
│   ├── ripe/
│   └── rotten/
└── test/
    ├── unripe/
    ├── ripe/
    └── rotten/
```

### For Detection (train_detection.py):
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your dataset according to the structure above

3. Train the model:
```bash
# For classification
python train.py

# For detection
python train_detection.py
```

4. Test the model:
```bash
python test_model.py
```

## Training Parameters

- **epochs**: 100 (adjust based on your dataset size)
- **imgsz**: 224 for classification, 640 for detection
- **batch**: 16 (reduce if GPU memory is limited)
- **patience**: 20 (early stopping)

## Output

Trained model will be saved in:
- Classification: `runs/classify/banana_ripeness/weights/best.pt`
- Detection: `runs/detect/banana_ripeness/weights/best.pt`
