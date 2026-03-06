from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n-cls.pt')  # nano model for classification

# Train the model
results = model.train(
    data='dataset',  # path to your dataset folder
    epochs=100,
    imgsz=224,
    batch=16,
    patience=20,  # early stopping
    save=True,
    project='runs/classify',
    name='banana_ripeness'
)

# Validate the model
metrics = model.val()

print(f"Training completed!")
print(f"Best model saved at: runs/classify/banana_ripeness/weights/best.pt")
