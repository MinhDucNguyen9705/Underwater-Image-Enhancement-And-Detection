import os
from ultralytics import YOLO
import argparse
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in underwater images using a trained YOLO model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained YOLO model file (e.g., .pt).')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the input image for detection.')
    parser.add_argument('--output_path', type=str, default=None, help='Optional path to save the annotated image.')
    return parser.parse_args()

def detect(model_path, img_path):
    model = YOLO(model_path)
    results = model(img_path, conf=0.25)
    return results

def draw_bounding_boxes(image, detection_results, output_dir=None):
    """Draw YOLO detection boxes, labels, and confidences on an image.

    Args:
        image: Image path or numpy image array.
        detection_results: Results returned by detect()/YOLO inference.
        output_path: Optional path to save the annotated image.

    Returns:
        Annotated image as a BGR numpy array.
    """
    if isinstance(image, str):
        annotated_image = cv2.imread(image)
        if annotated_image is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    elif isinstance(image, np.ndarray):
        annotated_image = image.copy()
    else:
        raise TypeError("image must be a file path or a numpy.ndarray")

    if not isinstance(detection_results, (list, tuple)):
        detection_results = [detection_results]

    for result in detection_results:
        names = result.names
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().item())
            class_id = int(box.cls[0].cpu().item())
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            label = f"{class_name} {confidence:.2f}"

            color = (0, 255, 0)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_y = max(y1, text_size[1] + baseline + 4)
            cv2.rectangle(
                annotated_image,
                (x1, text_y - text_size[1] - baseline - 4),
                (x1 + text_size[0] + 4, text_y),
                color,
                -1,
            )
            cv2.putText(
                annotated_image,
                label,
                (x1 + 2, text_y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/annotated_image.jpg"
        cv2.imwrite(output_path, annotated_image)

    return annotated_image

if __name__ == "__main__":
    args = parse_args()
    detect_results = detect(args.model_path, args.img_path)
    if args.output_path:
        draw_bounding_boxes(args.img_path, detect_results, args.output_path)
