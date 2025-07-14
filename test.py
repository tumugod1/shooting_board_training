from ultralytics import YOLO

model = YOLO("C:/Users/ROG/Desktop/Shooting_board_dataset/runs/detect/target-model/weights/best.pt")

model.predict(
    source="yolo_dataset/images/valid",
    conf=0.25,
    save=True,
    save_txt=True,
    device=0
)
