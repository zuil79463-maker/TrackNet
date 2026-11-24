"""RealSense SDK color capture with the same read/get/release interface as OpenCV."""
import threading

import cv2
import numpy as np


def load_sdk():
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError('RealSense SDK missing. Install pyrealsense2 in the project .venv.') from error
    return rs


def device_info(device, rs):
    return {'name': device.get_info(rs.camera_info.name),
            'serial': device.get_info(rs.camera_info.serial_number),
            'usb': device.get_info(rs.camera_info.usb_type_descriptor)
            if device.supports(rs.camera_info.usb_type_descriptor) else 'unknown'}


def list_devices():
    rs = load_sdk()
    context = rs.context()
    return [device_info(device, rs) for device in context.query_devices()]


class RealSenseCapture:
    def __init__(self, width, height, fps, serial=None):
        self.rs = rs = load_sdk()
        self.context = rs.context()
        devices = [device_info(device, rs) for device in self.context.query_devices()]
        candidates = [device for device in devices if device['serial'] == serial] if serial else [
            device for device in devices if 'D435I' in device['name'].upper()]
        if not candidates:
            raise RuntimeError('Requested RealSense device not found. Connect D435i with USB and use --list-devices to check.')
        if len(candidates) > 1:
            raise RuntimeError('Multiple D435i cameras found. Select one with --serial SERIAL.')
        self.info = candidates[0]
        self.pipeline = rs.pipeline(self.context)
        self.lock = threading.Lock()
        self.running = False
        config = rs.config()
        config.enable_device(self.info['serial'])
        # Explicitly request only RGB color: depth and IMU are not used by TrackNet.
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            profile = self.pipeline.start(config)
            self.running = True
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.width, self.height, self.fps = color_profile.width(), color_profile.height(), color_profile.fps()
        except RuntimeError as error:
            self.release()
            raise RuntimeError(f'Cannot start RealSense color {width}x{height}@{fps}. '
                               'Close RealSense Viewer/other camera apps; check USB and supported resolution/FPS. '
                               f'SDK: {error}') from error

    def read(self):
        if not self.running:
            return False, None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError('No color frame received')
            # Own the pixels: SDK buffers must not be reused while retained in our 8-frame deque.
            return True, np.asanyarray(color.get_data()).copy()
        except RuntimeError as error:
            raise RuntimeError(f'RealSense color stream interrupted: {error}') from error

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: self.width, cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                cv2.CAP_PROP_FPS: self.fps}.get(prop, 0)

    def release(self):
        with self.lock:
            if self.running:
                self.running = False
                self.pipeline.stop()
