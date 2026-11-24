"""Regression tests with a simulated camera; no live camera is opened."""
import csv
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import predict
import realsense_camera
from utils.general import HEIGHT, WIDTH, locate, prepare_input


class FakeCapture:
    def __init__(self, fail_after=None):
        self.stopped = threading.Event()
        self.count = 0
        self.fail_after = fail_after

    def read(self):
        if self.stopped.wait(1 / 120):
            return False, None
        if self.fail_after is not None and self.count >= self.fail_after:
            return False, None
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + self.count % 200, 80), 3, (255, 255, 255), -1)
        self.count += 1
        return True, frame

    def release(self):
        self.stopped.set()

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 180,
                cv2.CAP_PROP_FPS: 120}.get(prop, 0)


class RealtimeTests(unittest.TestCase):
    def test_latest_window_is_bounded_and_consecutive(self):
        capture = FakeCapture()
        stream = predict.CameraStream(capture, 8).start()
        try:
            packets = stream.snapshot(timeout=2)
            self.assertEqual(len(packets), 8)
            previous = packets[-1][0]
            time.sleep(.15)  # Simulate inference falling behind capture.
            packets = stream.snapshot(after=previous, timeout=2)
            indices = [packet[0] for packet in packets]
            self.assertEqual(indices, list(range(indices[-1] - 7, indices[-1] + 1)))
            self.assertGreater(indices[-1], previous + 1)
            self.assertEqual(len(stream.frames), 8)
        finally:
            stream.close()
        self.assertFalse(stream.thread.is_alive())
        self.assertTrue(capture.stopped.is_set())

    def test_disconnect_propagates_and_releases_camera(self):
        capture = FakeCapture(fail_after=0)
        stream = predict.CameraStream(capture, 8).start()
        try:
            with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                stream.snapshot(timeout=2)
        finally:
            stream.close()
        self.assertTrue(capture.stopped.is_set())
        self.assertFalse(stream.thread.is_alive())

    def test_preprocessing_and_coordinates(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # BGR red -> first RGB input channel.
        background = np.zeros_like(frame)
        tensor = prepare_input([frame] * 8, background)
        self.assertEqual(tensor.shape, (27, HEIGHT, WIDTH))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(tensor[:3] == 0))
        self.assertTrue(np.all(tensor[3] == 1))
        self.assertTrue(np.all(tensor[4:6] == 0))
        heatmap = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        self.assertEqual(locate(heatmap, (576, 1024, 3))[:3], (0, 0, 0))
        heatmap[50:53, 100:103] = .9
        self.assertEqual(locate(heatmap, (576, 1024, 3))[:3], (1, 202, 102))

    def test_camera_arguments_reject_video_files(self):
        self.assertEqual(predict.parse_args([]).source, 'realsense')
        self.assertEqual(predict.camera_source('RealSense'), 'realsense')
        self.assertEqual(predict.camera_source('1'), 1)
        self.assertEqual(predict.camera_source('rtsp://localhost/live'), 'rtsp://localhost/live')
        with self.assertRaises(predict.argparse.ArgumentTypeError):
            predict.camera_source('test.mp4')

    def test_official_checkpoint_stream_and_csv(self):
        capture = FakeCapture()
        with tempfile.TemporaryDirectory(prefix='tracknet-live-') as directory:
            path = Path(directory) / 'coordinates.csv'
            args = predict.parse_args(['--headless', '--background-frames', '3',
                                       '--max-results', '2', '--csv', str(path)])
            with patch.object(predict, 'open_camera', return_value=capture):
                self.assertEqual(predict.run(args), 2)
            self.assertTrue(capture.stopped.is_set())
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertLess(int(rows[0]['Frame']), int(rows[1]['Frame']))
            for row in rows:
                self.assertIn(row['Visibility'], ('0', '1'))
                self.assertTrue(row['TimestampUTC'].endswith('+00:00'))
                self.assertGreater(float(row['InferenceMs']), 0)
                self.assertGreaterEqual(float(row['ResultAgeMs']), float(row['InferenceMs']))
                self.assertTrue(0 <= int(row['X']) < 320)
                self.assertTrue(0 <= int(row['Y']) < 180)


class RealSenseTests(unittest.TestCase):
    def sdk(self):
        rs = MagicMock()
        rs.camera_info = SimpleNamespace(name='name', serial_number='serial', usb_type_descriptor='usb')
        rs.stream.color = 'color'
        rs.format.bgr8 = 'bgr8'
        device = MagicMock()
        device.get_info.side_effect = {'name': 'RealSense D435I', 'serial': 'test-serial', 'usb': '3.2'}.__getitem__
        device.supports.return_value = True
        rs.context.return_value.query_devices.return_value = [device]
        pipeline = rs.pipeline.return_value
        profile = pipeline.start.return_value.get_stream.return_value.as_video_stream_profile.return_value
        profile.width.return_value, profile.height.return_value, profile.fps.return_value = 1280, 720, 30
        return rs

    def test_color_only_frame_ownership_and_release(self):
        rs = self.sdk()
        pixels = np.zeros((720, 1280, 3), dtype=np.uint8)
        rs.pipeline.return_value.wait_for_frames.return_value.get_color_frame.return_value.get_data.return_value = pixels
        with patch.object(realsense_camera, 'load_sdk', return_value=rs):
            capture = realsense_camera.RealSenseCapture(1280, 720, 30)
            rs.config.return_value.enable_device.assert_called_once_with('test-serial')
            rs.config.return_value.enable_stream.assert_called_once_with('color', 1280, 720, 'bgr8', 30)
            ok, frame = capture.read()
            self.assertTrue(ok)
            pixels[:] = 255
            self.assertTrue(np.all(frame == 0))
            rs.pipeline.return_value.wait_for_frames.assert_called_once_with(timeout_ms=3000)
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 30)
            capture.release()
            capture.release()
            rs.pipeline.return_value.stop.assert_called_once()
            self.assertEqual(capture.read(), (False, None))

    def test_missing_device_never_falls_back_to_webcam(self):
        rs = self.sdk()
        with patch.object(realsense_camera, 'load_sdk', return_value=rs):
            with self.assertRaisesRegex(RuntimeError, 'not found'):
                realsense_camera.RealSenseCapture(1280, 720, 30, serial='different-device')
            rs.pipeline.assert_not_called()

    def test_timeout_and_start_failure_report_sdk_error(self):
        rs = self.sdk()
        with patch.object(realsense_camera, 'load_sdk', return_value=rs):
            capture = realsense_camera.RealSenseCapture(1280, 720, 30)
            rs.pipeline.return_value.wait_for_frames.side_effect = RuntimeError('frame timeout')
            with self.assertRaisesRegex(RuntimeError, 'RealSense color stream interrupted'):
                capture.read()
            capture.release()
        rs = self.sdk()
        rs.pipeline.return_value.start.side_effect = RuntimeError('unsupported profile')
        with patch.object(realsense_camera, 'load_sdk', return_value=rs):
            with self.assertRaisesRegex(RuntimeError, 'Cannot start RealSense color'):
                realsense_camera.RealSenseCapture(1280, 720, 30)


if __name__ == '__main__':
    unittest.main(verbosity=2)
