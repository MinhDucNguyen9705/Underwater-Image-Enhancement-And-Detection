from pathlib import Path
import torch
import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2
from detect import draw_bounding_boxes
from model import myModel
from utils.data_utils import pad_image, unpad_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENHANCE_MODEL_PATH = Path("weights/best_model.pth")
RAW_DETECT_MODEL_PATH = Path("weights/ruod-raw-yolo26n.pt")
CLEAN_DETECT_MODEL_PATH = Path("weights/ruod-clean-yolo26n.pt")

ENHANCE_MODEL = myModel(in_channels=3, feature_channels=32, use_white_balance=True).to(DEVICE)
ENHANCE_MODEL.load_state_dict(torch.load(ENHANCE_MODEL_PATH, map_location=DEVICE, weights_only=False))
RAW_DETECT_MODEL = YOLO(str(RAW_DETECT_MODEL_PATH)).to(DEVICE)
CLEAN_DETECT_MODEL = YOLO(str(CLEAN_DETECT_MODEL_PATH)).to(DEVICE)
ENHANCE_MODEL.eval()
CLEAN_DETECT_MODEL.eval()
RAW_DETECT_MODEL.eval()

def reset_for_upload(image):
    if image is None:
        return gr.update(value=None, visible=False), False, None, None, None, None

    height, width = image.shape[:2]
    return (
        gr.update(value=None, visible=False),
        False,
        image,
        None,
        gr.update(value=width),
        gr.update(value=height),
    )


def reset_for_clear():
    return (
        gr.update(value=None, visible=False),
        False,
        None,
        None,
        gr.update(value=None),
        gr.update(value=None),
    )


def resize_image_array(image, width, height):
    width = int(width)
    height = int(height)
    interpolation = cv2.INTER_AREA if width < image.shape[1] or height < image.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_current_image(image, width, height, is_clean_image, raw_image, clean_image):
    if image is None:
        return None, gr.update(value=None, visible=False), False, None, None

    if width is None or height is None or int(width) <= 0 or int(height) <= 0:
        raise gr.Error("Width and height must be positive numbers.")

    resized_image = resize_image_array(image, width, height)

    if is_clean_image:
        resized_raw = resize_image_array(raw_image, width, height) if raw_image is not None else resized_image
        resized_clean = resize_image_array(clean_image, width, height) if clean_image is not None else resized_image
        return resized_image, gr.update(value=(resized_raw, resized_clean), visible=True), True, resized_raw, resized_clean

    return resized_image, gr.update(value=None, visible=False), False, resized_image, None


def reduce_current_image_resolution(image, width, height, divisor, is_clean_image, raw_image, clean_image):
    if image is None:
        return None, gr.update(value=None, visible=False), False, None, None, None, None

    if divisor is None or float(divisor) <= 0:
        raise gr.Error("Reduce by must be a positive number.")

    current_height, current_width = image.shape[:2]
    width = current_width if width is None else int(width)
    height = current_height if height is None else int(height)

    if width <= 0 or height <= 0:
        raise gr.Error("Width and height must be positive numbers.")

    new_width = max(1, round(width / float(divisor)))
    new_height = max(1, round(height / float(divisor)))
    resized_image, output_update, is_clean, resized_raw, resized_clean = resize_current_image(
        image,
        new_width,
        new_height,
        is_clean_image,
        raw_image,
        clean_image,
    )

    return resized_image, output_update, is_clean, resized_raw, resized_clean, new_width, new_height


def run_enhancement(image):
    image_tensor, (orig_h, orig_w) = pad_image(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        enhanced_tensor = ENHANCE_MODEL(image_tensor)
    enhanced_image = unpad_image(enhanced_tensor.squeeze(0).cpu(), orig_h, orig_w).permute(1, 2, 0).numpy()
    return (np.clip(enhanced_image, 0, 1) * 255).astype(np.uint8)


def enhance_image(image):
    """Placeholder for image enhancement.

    Replace this function body with your enhancement model later.
    """
    if image is None:
        return None, gr.update(value=None, visible=False), False, None, None
    
    raw_image = image.copy()
    enhanced_image = run_enhancement(image)

    return enhanced_image, gr.update(value=(raw_image, enhanced_image), visible=True), True, raw_image, enhanced_image

def detect_objects(image, is_clean_image, raw_image, clean_image):
    if image is None:
        return gr.update(value=None, visible=False), False, None, None

    if is_clean_image:
        raw_for_detection = raw_image if raw_image is not None else image
        clean_for_detection = clean_image if clean_image is not None else image
    else:
        raw_for_detection = image
        clean_for_detection = run_enhancement(image)

    raw_results = RAW_DETECT_MODEL(raw_for_detection, conf=0.25)
    clean_results = CLEAN_DETECT_MODEL(clean_for_detection, conf=0.25)
    annotated_raw = draw_bounding_boxes(raw_for_detection, raw_results)
    annotated_clean = draw_bounding_boxes(clean_for_detection, clean_results)

    return gr.update(value=(annotated_raw, annotated_clean), visible=True), True, raw_for_detection, clean_for_detection

def remove_annotations(image, is_clean_image, raw_image, clean_image):
    if image is None:
        return gr.update(value=None, visible=False)

    if is_clean_image:
        raw_output = raw_image if raw_image is not None else image
        clean_output = clean_image if clean_image is not None else image
        return gr.update(value=(raw_output, clean_output), visible=True)

    return gr.update(value=image, visible=True)

with gr.Blocks(title="Underwater Image Enhancement and Detection") as demo:
    gr.Markdown("# Underwater Image Enhancement and Detection")
    is_clean_image = gr.State(False)
    raw_image_state = gr.State(None)
    clean_image_state = gr.State(None)

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="numpy")
        output_image = gr.ImageSlider(
            label="Output Image",
            type="numpy",
            visible=False,
            slider_position=50,
            max_height=500,
        )

    with gr.Row():
        resize_width = gr.Number(label="Width", precision=0, minimum=1)
        resize_height = gr.Number(label="Height", precision=0, minimum=1)
        resize_button = gr.Button("Resize Image")
        reduce_factor = gr.Number(label="Reduce by", value=2, minimum=0.01)
        reduce_button = gr.Button("Reduce & Resize")

    with gr.Row():
        enhance_button = gr.Button("Image Enhancement")
        detect_button = gr.Button("Object Detection")
        remove_annotations_button = gr.Button("Remove Annotations")

    resize_button.click(
        fn=resize_current_image,
        inputs=[input_image, resize_width, resize_height, is_clean_image, raw_image_state, clean_image_state],
        outputs=[input_image, output_image, is_clean_image, raw_image_state, clean_image_state],
    )
    reduce_button.click(
        fn=reduce_current_image_resolution,
        inputs=[
            input_image,
            resize_width,
            resize_height,
            reduce_factor,
            is_clean_image,
            raw_image_state,
            clean_image_state,
        ],
        outputs=[
            input_image,
            output_image,
            is_clean_image,
            raw_image_state,
            clean_image_state,
            resize_width,
            resize_height,
        ],
    )
    enhance_button.click(
        fn=enhance_image,
        inputs=input_image,
        outputs=[input_image, output_image, is_clean_image, raw_image_state, clean_image_state],
    )
    detect_button.click(
        fn=detect_objects,
        inputs=[input_image, is_clean_image, raw_image_state, clean_image_state],
        outputs=[output_image, is_clean_image, raw_image_state, clean_image_state],
    )
    remove_annotations_button.click(
        fn=remove_annotations,
        inputs=[input_image, is_clean_image, raw_image_state, clean_image_state],
        outputs=output_image,
    )
    input_image.upload(
        fn=reset_for_upload,
        inputs=input_image,
        outputs=[output_image, is_clean_image, raw_image_state, clean_image_state, resize_width, resize_height],
        show_progress="hidden",
    )
    input_image.clear(
        fn=reset_for_clear,
        inputs=None,
        outputs=[output_image, is_clean_image, raw_image_state, clean_image_state, resize_width, resize_height],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch()
