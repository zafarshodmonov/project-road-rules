import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np

def load_models():
    """YOLO va OCR modellarini yuklash."""
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    model_yolo = YOLO("yolo11n.pt")
    model_license_plate = YOLO('/content/sample_data/number_car.pt')
    return ocr, model_yolo, model_license_plate

def process_license_plate(frame, model_license_plate, ocr, result_image):
    """Avtomobil raqamini aniqlash va OCR orqali matnni olish."""
    results_license_plate = model_license_plate(frame)
    for box in results_license_plate[0].boxes:
        if box.conf >= 0.6:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop_img = frame[y1:y2, x1:x2]
            results_ocr = ocr.ocr(crop_img, cls=True)
            if results_ocr and results_ocr[0]:
                for line in results_ocr[0]:
                    text = line[1][0]
                    label = f"license plate: {text}"
                    draw_label(result_image, label, (x1, y1), (255, 0, 0), (255, 255, 255))
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return result_image


def detect_traffic_light_color(cropped_image):
    """
    Svetofor rasmini kiritib, qaysi chiroq yoqilganligini aniqlaydi.

    Parametrlar:
        cropped_image (numpy.ndarray): Svetofor rasmi.

    Natija:
        string: Yoqilgan rang ("Red", "Yellow", "Green").
    """
    # Tasvirni HSV formatga o'tkazish
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Ranglar uchun HSV oralig'ini belgilash
    red_lower = np.array([0, 70, 50])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    green_lower = np.array([40, 70, 70])
    green_upper = np.array([80, 255, 255])

    # Har bir rang bo'yicha niqob yaratish
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Rang maydonlarining piksel qiymatlarini hisoblash
    red_area = cv2.countNonZero(red_mask)
    yellow_area = cv2.countNonZero(yellow_mask)
    green_area = cv2.countNonZero(green_mask)

    # Eng katta maydonli rangni aniqlash
    if red_area > yellow_area and red_area > green_area:
        return "Red"
    elif yellow_area > red_area and yellow_area > green_area:
        return "Yellow"
    elif green_area > red_area and green_area > yellow_area:
        return "Green"
    else:
        return "Unknown"


def draw_traffic_light_status(image, bounding_box, light_color):
    """
    Svetaforning bounding box tagiga rangni yozadi.

    Parametrlar:
        image (numpy.ndarray): Asosiy rasm.
        bounding_box (tuple): (x_min, y_min, x_max, y_max) koordinatalari.
        light_color (string): "Red", "Yellow" yoki "Green".

    Natija:
        numpy.ndarray: Rang yozilgan rasm.
    """
    # Bounding box koordinatalarini oling
    x_min, y_min, x_max, y_max = bounding_box

    # Rang uchun matnni sozlash
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Rangni ko'rsatish uchun mos matn rangi
    color_map = {
        "Red": (0, 0, 255),
        "Yellow": (0, 255, 255),
        "Green": (0, 255, 0),
        "Unknown": (255, 255, 255)
    }

    # Yozuv rangi
    text_color = color_map.get(light_color, (255, 255, 255))

    # Yozuv joylashuvi (bounding box ostiga joylashadi)
    text_position = (x_min, y_max + 30)

    # Rasmga yozuvni joylashtirish
    cv2.putText(image, light_color, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA)

    return image


def process_yolo_objects(frame, model_yolo, result_image):
    """YOLO modeli orqali obyektlarni aniqlash va qayta ishlash."""
    results_yolo = model_yolo(frame)
    for box in results_yolo[0].boxes:
        class_id = int(box.cls[0])
        if class_id <= 11 and box.conf >= 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if class_id == 9:  # Svetafor
                crop_img = frame[y1:y2, x1:x2]
                svetafor_rangi = detect_traffic_light_color(crop_img)
                result_image = draw_traffic_light_status(result_image, (x1, y1, x2, y2), svetafor_rangi)
            else:
                label = f"{model_yolo.names[class_id]} {int(100 * box.conf.item())}%"
                draw_label(result_image, label, (x1, y1), (0, 0, 255), (255, 255, 255))
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return result_image

def draw_label(image, label, position, bg_color, text_color):
    """Obyekt ustiga matn va fon chizish."""
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), bg_color, -1)
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, lineType=cv2.LINE_AA)

def process_video(video_path, output_path, ocr, model_yolo, model_license_plate):
    """Videoga ishlov berish."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_image = frame.copy()
        result_image = process_license_plate(frame, model_license_plate, ocr, result_image)
        result_image = process_yolo_objects(frame, model_yolo, result_image)
        out.write(result_image)
        cv2.imshow("Result", result_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    video_path = "/content/svetafor_1.mp4"
    output_path = "output_video_1.mp4"
    ocr, model_yolo, model_license_plate = load_models()
    process_video(video_path, output_path, ocr, model_yolo, model_license_plate)

if __name__ == '__main__':
    main()
