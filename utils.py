import cv2

def check_blur(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()

    print("Blur Value:", blur_value)

    # 🔥 STRONGER THRESHOLD
    if blur_value < 150:
        return False   # blurry
    return True