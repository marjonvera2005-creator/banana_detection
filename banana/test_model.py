from ultralytics import YOLO
from PIL import Image

# Load trained model
model = YOLO('runs/classify/banana_ripeness/weights/best.pt')  # or runs/detect/...

# Test on a single image
results = model('path/to/test/image.jpg')

# Display results
for result in results:
    print(f"Class: {result.names[result.probs.top1]}")
    print(f"Confidence: {result.probs.top1conf:.2f}")
    result.show()
