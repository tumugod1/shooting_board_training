import os
import cv2
import numpy as np

def yolo_to_pixel_bbox(line, img_width, img_height):
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1]) * img_width
    y_center = float(parts[2]) * img_height
    width = float(parts[3]) * img_width
    height = float(parts[4]) * img_height
    x = int(x_center - width / 2)
    y = int(y_center - height / 2)
    return class_id, x, y, int(width), int(height)

input_dir = "runs/detect/predict"
labels_dir = os.path.join(input_dir, "labels")
output_dir = "runs/detect/centered_results"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".jpg"):
        continue

    img_path = os.path.join(input_dir, filename)
    label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

    image = cv2.imread(img_path)
    if image is None or not os.path.exists(label_path):
        continue

    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        class_id, x, y, bw, bh = yolo_to_pixel_bbox(line, w, h)
        roi = image[y:y+bh, x:x+bw]  # hedefin kırpılmış görüntüsü

        # ROI'yi griye çevir ve yumuşat
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Daire tespiti yap
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=roi.shape[0]//8,
            param1=100,
            param2=20,
            minRadius=5,
            maxRadius=roi.shape[1]//2
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (cx, cy, r) in circles[0, :1]:  # yalnızca en büyük daireyi al
                # ROI içindeki koordinatı tüm resme çevir
                abs_cx = x + cx
                abs_cy = y + cy
                cv2.circle(image, (abs_cx, abs_cy), 5, (0, 0, 255), -1)
                cv2.putText(image, f"({abs_cx},{abs_cy})", (abs_cx + 10, abs_cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, image)

print("Daire merkezleri çizildi.")
