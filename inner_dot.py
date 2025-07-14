import os
import cv2

def yolo_line_to_center(line, img_w, img_h):
    parts = line.strip().split()
    class_id = int(parts[0])
    if class_id != 1:  # sadece inner_dot
        return None
    x = float(parts[1]) * img_w
    y = float(parts[2]) * img_h
    return int(x), int(y)

input_dir = "runs/detect/predict"
label_dir = os.path.join(input_dir, "labels")
output_dir = "runs/detect/centered_results"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(input_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

    image = cv2.imread(image_path)
    if image is None or not os.path.exists(label_path):
        continue

    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            center = yolo_line_to_center(line, w, h)
            if center:
                cx, cy = center
                cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(image, f"({cx},{cy})", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, image)
    print(f"[✓] Kaydedildi: {out_path}")
