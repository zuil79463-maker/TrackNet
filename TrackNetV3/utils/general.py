"""Preprocessing and localization for the official TrackNet checkpoint."""
import cv2
import numpy as np
from PIL import Image

HEIGHT, WIDTH = 288, 512


def resize_rgb(frame):
    """Match upstream PIL RGB resizing, before normalization."""
    return np.asarray(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((WIDTH, HEIGHT)))


def stack_input(resized_frames, background):
    """Build the (C, H, W) float32 network input from already resized RGB images."""
    # The deployed checkpoint expects RGB background followed by eight RGB frames.
    images = [background] + list(resized_frames)
    return np.ascontiguousarray(
        np.concatenate([image.transpose(2, 0, 1) for image in images], axis=0), dtype=np.float32) / np.float32(255.0)


def prepare_input(frames, background):
    return stack_input([resize_rgb(frame) for frame in frames], background)


def locate(heatmap, image_shape, threshold=0.5):
    """Return visibility, original-image x/y and peak heatmap score."""
    mask = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    confidence = float(heatmap.max())
    if not contours:
        return 0, 0, 0, confidence
    boxes = [cv2.boundingRect(contour) for contour in contours]
    x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
    height, width = image_shape[:2]
    return (1, min(width - 1, int(int(x + w / 2) * width / WIDTH)),
            min(height - 1, int(int(y + h / 2) * height / HEIGHT)), confidence)
