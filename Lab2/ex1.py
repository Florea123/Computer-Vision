import os
import cv2
import numpy as np

def First_Method(image):
    b = image[:, :, 0].astype(np.int32)
    g = image[:, :, 1].astype(np.int32)
    r = image[:, :, 2].astype(np.int32)

    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)

    cond = (
        (r > 95) & (g > 40) & (b > 20) &
        ((max_val - min_val) > 15) &
        (np.abs(r - g) > 15) &
        (r > g) & (r > b)
    )

    skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    skin_mask[cond] = 255
    return skin_mask

def Second_Method(image):
    hsv_image = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
    h_val = hsv_image[:, :, 0]
    s_val = hsv_image[:, :, 1]
    v_val = hsv_image[:, :, 2]

    cond = (
        (h_val >= 0.0) & (h_val <= 50.0) &
        (s_val >= 0.23) & (s_val <= 0.68) &
        (v_val >= 0.35) & (v_val <= 1.0)
    )

    skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    skin_mask[cond] = 255
    return skin_mask

def Third_Method(image):
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    cond = (y > 80) & (cb > 85) & (cb < 135) & (cr > 135) & (cr < 180)

    skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    skin_mask[cond] = 255
    return skin_mask

def Fourth_Method(image):
    b = image[:, :, 0].astype(np.int32)
    g = image[:, :, 1].astype(np.int32)
    r = image[:, :, 2].astype(np.int32)
    has_alpha = (len(image.shape) > 2 and image.shape[2] == 4)
    a = image[:, :, 3].astype(np.int32) if has_alpha else 255

    rgb_cond = (
        (r > 95) & (g > 40) & (b > 20) &
        (r > g) & (r > b) &
        (np.abs(r - g) > 15) &
        (a > 15)
    )

    hsv_image = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
    h_val = hsv_image[:, :, 0]
    s_val = hsv_image[:, :, 1]
    cond1 = rgb_cond & (h_val >= 0.0) & (h_val <= 50.0) & (s_val >= 0.23) & (s_val <= 0.68)

    rf = image[:, :, 2].astype(np.float32)
    gf = image[:, :, 1].astype(np.float32)
    bf = image[:, :, 0].astype(np.float32)
    y = 0.299 * rf + 0.587 * gf + 0.114 * bf
    cb = -0.1687 * rf - 0.3313 * gf + 0.5 * bf + 128
    cr = 0.5 * rf - 0.4187 * gf - 0.0813 * bf + 128

    cond2 = (
        rgb_cond &
        (cr > 135) & (cb > 85) & (y > 80) &
        (cr <= 1.5862 * cb + 20) &
        (cr >= 0.3448 * cb + 76.2069) &
        (cr >= -4.5652 * cb + 234.5652) &
        (cr <= -1.15 * cb + 301.75) &
        (cr <= -2.2857 * cb + 432.85)
    )

    skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    skin_mask[cond1 | cond2] = 255
    return skin_mask

def evaluate_dataset():
    methods = {
        "Method 1": First_Method,
        "Method 2": Second_Method,
        "Method 3": Third_Method,
        "Method 4": Fourth_Method
    }

    confusion_matrices = {
        name: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for name in methods
    }

    subsets = [
        ("FacePhoto", "GroundT_FacePhoto"),
        ("FamilyPhoto", "GroundT_FamilyPhoto")
    ]

    base_dir = "Face_Dataset"
    total_images = 0


    for img_subdir, gt_subdir in subsets:
        img_folder = os.path.join(base_dir, "Pratheepan_Dataset", img_subdir)
        gt_folder = os.path.join(base_dir, "Ground_Truth", gt_subdir)

        if not os.path.exists(img_folder):
            continue

        for file_name in os.listdir(img_folder):
            if file_name.startswith("."):
                continue

            img_path = os.path.join(img_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            gt_path = os.path.join(gt_folder, base_name + ".png")

            if not os.path.exists(gt_path):
                continue

            img = cv2.imread(img_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if img is None or gt is None:
                continue

            total_images += 1

            gt_skin = (gt == 255)

            for method_name, method_fn in methods.items():
                pred = method_fn(img)
                pred_skin = (pred == 255)

                tp = np.sum(gt_skin & pred_skin)
                fn = np.sum(gt_skin & ~pred_skin)
                fp = np.sum(~gt_skin & pred_skin)
                tn = np.sum(~gt_skin & ~pred_skin)

                confusion_matrices[method_name]["TP"] += int(tp)
                confusion_matrices[method_name]["FN"] += int(fn)
                confusion_matrices[method_name]["FP"] += int(fp)
                confusion_matrices[method_name]["TN"] += int(tn)


    for method_name, matrix in confusion_matrices.items():
        tp = matrix["TP"]
        fn = matrix["FN"]
        fp = matrix["FP"]
        tn = matrix["TN"]

        total_pixels = tp + fn + fp + tn
        accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0

        print(f"Method: {method_name}")
        print(f"Accuracy = (TP + TN) / (TP + FN + FP + TN) = {accuracy * 100:.2f}%")


def detect_face_square(image):
    skin_mask = Second_Method(image)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Nu a fost detectata nicio regiune de piele.")
        return image, clean_mask

    face_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(face_contour)

    side = max(w, h)
    cx = x + w // 2
    cy = y + h // 2

    img_h, img_w = image.shape[:2]

    sq_x = max(0, min(cx - side // 2, img_w - side))
    sq_y = max(0, min(cy - side // 2, img_h - side))
    side = min(side, img_w, img_h)

    result_image = image.copy()
    cv2.rectangle(result_image, (sq_x, sq_y), (sq_x + side, sq_y + side), (0, 255, 0), 2)

    return result_image, clean_mask



test_image=cv2.imread("group.jpg")
image = First_Method(test_image)
image1=Second_Method(test_image)
image2=Third_Method(test_image)
image3=Fourth_Method(test_image)
cv2.imshow("Original Image",test_image)
cv2.imshow("First Method",image)
cv2.imshow("Second Method",image1)
cv2.imshow("Third Method",image2)
cv2.imshow("Fourth Method",image3)
cv2.waitKey(0)
cv2.destroyAllWindows()

evaluate_dataset()


portrait_path = "Face_Dataset/Pratheepan_Dataset/FacePhoto/Aishwarya-Rai_20091229_aatheory.jpg"
portrait_image = cv2.imread(portrait_path)

if portrait_image is not None:
    face_detected, face_mask = detect_face_square(portrait_image)
    cv2.imshow("Original", portrait_image)
    cv2.imshow("Skin Mask", face_mask)
    cv2.imshow("Detected Face Square", face_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


