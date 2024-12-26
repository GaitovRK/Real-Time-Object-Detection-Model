from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")
# Evaluate model performance on the validation set
metrics = model.val(data="config.yaml")
