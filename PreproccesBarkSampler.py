import os
import cv2
import numpy as np
from glob import glob

TREE_TYPE = "red-gum"
IMAGE_DIR = f"C:/Users/zeynep/Desktop/bark crop/{TREE_TYPE}"
OUTPUT_DIR = f"C:/Users/zeynep/Desktop/bark crop/{TREE_TYPE}/crop/{TREE_TYPE}"

CROP_SIZE = 700   
OUTPUT_SIZE = 500  

UI_SCALE = 0.5 

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = []
for ext in IMAGE_EXTENSIONS:
    pattern = os.path.join(IMAGE_DIR, "**", ext)
    for p in glob(pattern, recursive=True):
        try:
            if os.path.commonpath([p, OUTPUT_DIR]) == OUTPUT_DIR:
                continue
        except ValueError:
            pass
        image_paths.append(p)

image_paths.sort()

if not image_paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

print(f"Found {len(image_paths)} images.")

current_points = []
current_image = None
display_image = None

def mouse_callback(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(x / UI_SCALE)
        real_y = int(y / UI_SCALE)
        current_points.append((real_x, real_y))
        print(f"Added point (original coords): ({real_x}, {real_y})")

def draw_points():
    global display_image, current_image, current_points
    
    h, w = current_image.shape[:2]
    new_w, new_h = int(w * UI_SCALE), int(h * UI_SCALE)
    display_image = cv2.resize(current_image, (new_w, new_h))
    
    half = (CROP_SIZE // 2) * UI_SCALE

    for (real_cx, real_cy) in current_points:
        cx, cy = int(real_cx * UI_SCALE), int(real_cy * UI_SCALE)
        
        x1 = int(max(cx - half, 0))
        y1 = int(max(cy - half, 0))
        x2 = int(min(cx + half, new_w - 1))
        y2 = int(min(cy + half, new_h - 1))

        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(display_image, (cx, cy), 3, (0, 0, 255), -1)

def get_next_index_for_folder(folder):
    if not os.path.exists(folder):
        return 1
    files = glob(os.path.join(folder, "*.jpg"))
    indices = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if name.isdigit():
            indices.append(int(name))
    return max(indices) + 1 if indices else 1

def save_crops(img_path):
    global current_points, current_image
    if not current_points:
        return

    h, w = current_image.shape[:2]
    half = CROP_SIZE // 2

    src_dir = os.path.dirname(img_path)
    rel_dir = os.path.relpath(src_dir, IMAGE_DIR)
    target_dir = os.path.join(OUTPUT_DIR, "" if rel_dir == "." else rel_dir)
    os.makedirs(target_dir, exist_ok=True)

    idx = get_next_index_for_folder(target_dir)

    for (cx, cy) in current_points:
        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w)
        y2 = min(cy + half, h)

        crop = current_image[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        crop_resized = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
        
        out_path = os.path.join(target_dir, f"{idx}.jpg")
        cv2.imwrite(out_path, crop_resized)
        print(f"Saved: {out_path}")
        idx += 1

def main():
    global current_image, display_image, current_points

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", mouse_callback)

    for img_idx, img_path in enumerate(image_paths):
        current_image = cv2.imread(img_path)
        if current_image is None:
            continue

        current_points = []
        print(f"\n[{img_idx + 1}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")

        while True:
            draw_points()
            cv2.imshow("image", display_image)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('c'):
                current_points = []
                print("Cleared points.")
            elif key == ord('n'):
                save_crops(img_path)
                break
            elif key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()