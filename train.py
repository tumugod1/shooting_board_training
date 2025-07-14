from ultralytics import YOLO
import torch

def main():
    print("CUDA Kullanılabilir mi?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    model.train(
        data="C:/Users/ROG/Desktop/Shooting_board_dataset/data.yaml",  # senin dataset yolun
        epochs=50,
        imgsz=640,
        batch=8,
        name="target-model",
        device=0
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # windows için zorunlu
    main()
