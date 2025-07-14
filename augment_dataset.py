import os
import cv2
import albumentations as A

#  Klasör yolları
input_img_folder = "C:\\Users\\ROG\\Desktop\\Shooting_board_dataset\\images"
input_lbl_folder = "C:\\Users\\ROG\\Desktop\\Shooting_board_dataset\\labels"
output_img_folder = "C:\\Users\\ROG\\Desktop\\Shooting_board_dataset\\dataset\\images\\augmented"
output_lbl_folder = "C:\\Users\\ROG\\Desktop\\Shooting_board_dataset\\dataset\\labels\\augmented"

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_lbl_folder, exist_ok=True)

#  Augmentasyon tanımı
transform = A.Compose([
    A.Rotate(limit=30, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.Resize(640, 640)
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.1
))

#  Görselleri işle
for filename in os.listdir(input_img_folder):
    if not filename.endswith((".jpg", ".png")):
        continue

    image = cv2.imread(os.path.join(input_img_folder, filename))
    basename = filename.rsplit(".", 1)[0]
    label_path = os.path.join(input_lbl_folder, basename + ".txt")

    if not os.path.exists(label_path):
        print(f"[WARN] Label bulunamadı, atlanıyor: {label_path}")
        continue

    bboxes = []
    class_labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[WARN] Label format hatası, atlanıyor: {label_path} satır: {line.strip()}")
                continue

            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            if w <= 0 or h <= 0:
                print(f"[WARN] Geçersiz bbox (w/h=0): {label_path} -> [{x}, {y}, {w}, {h}]")
                continue

            bboxes.append([x, y, w, h])
            class_labels.append(class_id)

    for i in range(10):  # Her görsel için 10 augmentasyon
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented.get('class_labels', [])
        except Exception as e:
            print(f"[WARN] Augment sırasında hata oluştu ({basename}_aug_{i}): {e}")
            continue

        # Hatalı bbox varsa atla
        valid_bboxes = []
        valid_labels = []
        for bbox, label in zip(aug_bboxes, aug_labels):
            x, y, w, h = bbox
            if w > 0 and h > 0:
                valid_bboxes.append([x, y, w, h])
                valid_labels.append(label)

        if len(valid_bboxes) == 0:
            continue

        out_img_name = f"{basename}_aug_{i}.jpg"
        out_lbl_name = f"{basename}_aug_{i}.txt"

        cv2.imwrite(os.path.join(output_img_folder, out_img_name), aug_img)

        with open(os.path.join(output_lbl_folder, out_lbl_name), "w") as f:
            for bbox, label in zip(valid_bboxes, valid_labels):
                x, y, w, h = bbox
                f.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
