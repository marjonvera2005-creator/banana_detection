from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # nano model for detection

# Train the model
results = model.train(
    data='data.yaml',  # path to your dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    save=True,
    project='runs/detect',
    name='banana_ripeness'
)

# Validate the model
metrics = model.val()

print(f"Training completed!")
print(f"Best model saved at: runs/detect/banana_ripeness/weights/best.pt")
