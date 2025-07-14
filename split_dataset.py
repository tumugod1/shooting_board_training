import os
import shutil
from sklearn.model_selection import train_test_split

img_folder = "C:/Users/ROG/Desktop/Shooting_board_dataset/dataset/images/augmented"
lbl_folder = "C:/Users/ROG/Desktop/Shooting_board_dataset/dataset/labels/augmented"

train_imgs, val_imgs = train_test_split(os.listdir(img_folder), test_size=0.2, random_state=42)

# Yeni klasörleri oluştur
for path in [
    "C:/Users/ROG/Desktop/Shooting_board_dataset/yolo_dataset/images/train",
    "C:/Users/ROG/Desktop/Shooting_board_dataset/yolo_dataset/images/valid",
    "C:/Users/ROG/Desktop/Shooting_board_dataset/yolo_dataset/labels/train",
    "C:/Users/ROG/Desktop/Shooting_board_dataset/yolo_dataset/labels/valid"
]:
    os.makedirs(path, exist_ok=True)

# Train klasörüne kopyala
for file in train_imgs:
    shutil.copy(f"{img_folder}/{file}", f"yolo_dataset/images/train/{file}")
    txt_name = file.replace(".jpg", ".txt").replace(".png", ".txt")
    shutil.copy(f"{lbl_folder}/{txt_name}", f"yolo_dataset/labels/train/{txt_name}")

# Valid klasörüne kopyala
for file in val_imgs:
    shutil.copy(f"{img_folder}/{file}", f"yolo_dataset/images/valid/{file}")
    txt_name = file.replace(".jpg", ".txt").replace(".png", ".txt")
    shutil.copy(f"{lbl_folder}/{txt_name}", f"yolo_dataset/labels/valid/{txt_name}")
