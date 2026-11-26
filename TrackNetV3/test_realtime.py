"""Regression tests with a simulated camera and NPU; no live camera or Ascend device is opened."""
import csv
import ctypes
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import inference
import predict
import realsense_camera
from utils.general import HEIGHT, WIDTH, locate, prepare_input, resize_rgb, stack_input


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


def fake_acl(input_shape=(1, 27, HEIGHT, WIDTH)):
    """Minimal pyACL stand-in (device run mode, shared memory) backed by ctypes buffers."""
    memory, log = {}, []

    def malloc(size, policy=0):
        buffer = ctypes.create_string_buffer(size)
        memory[ctypes.addressof(buffer)] = buffer
        return ctypes.addressof(buffer), 0

    def free(pointer):
        memory.pop(pointer)
        return 0

    def memcpy(destination, destination_size, source, count, kind):
        assert count <= destination_size
        log.append(('memcpy', kind))
        ctypes.memmove(destination, source, count)
        return 0

    def add_dataset_buffer(dataset, data):
        dataset.append(data)
        return dataset, 0

    def execute(model_id, inputs, outputs):
        (in_ptr, in_size), (out_ptr, out_size) = inputs[0], outputs[0]
        source = np.ctypeslib.as_array((ctypes.c_float * (in_size // 4)).from_address(in_ptr))
        target = np.ctypeslib.as_array((ctypes.c_float * (out_size // 4)).from_address(out_ptr))
        # Echo the last input channel so the test can check the data path end to end.
        target[:] = source.reshape(-1, HEIGHT * WIDTH)[-1]
        return 0

    size = int(np.prod(input_shape)) * 4
    acl = SimpleNamespace(
        init=lambda: 0, finalize=lambda: log.append(('finalize',)) or 0,
        create_data_buffer=lambda pointer, size: (pointer, size), destroy_data_buffer=lambda data: 0,
        rt=SimpleNamespace(set_device=lambda i: 0, reset_device=lambda i: log.append(('reset',)) or 0,
                           create_context=lambda i: ('context', 0), destroy_context=lambda c: 0,
                           set_context=lambda c: 0, get_run_mode=lambda: (0, 0), malloc=malloc, free=free,
                           malloc_host=malloc, free_host=free, memcpy=memcpy),
        mdl=SimpleNamespace(load_from_file=lambda path: (7, 0), unload=lambda m: log.append(('unload',)) or 0,
                            create_desc=lambda: 'desc', destroy_desc=lambda d: 0, get_desc=lambda d, m: 0,
                            get_num_inputs=lambda d: 1, get_num_outputs=lambda d: 1,
                            get_input_dims=lambda d, i: ({'dims': list(input_shape)}, 0),
                            get_input_size_by_index=lambda d, i: size,
                            get_output_size_by_index=lambda d, i: HEIGHT * WIDTH * 4,
                            create_dataset=list, destroy_dataset=lambda d: 0,
                            add_dataset_buffer=add_dataset_buffer, execute=execute))
    return acl, memory, log


class EngineTests(unittest.TestCase):
    def test_cached_resize_matches_reference_preprocessing(self):
        rng = np.random.default_rng(1)
        frames = [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(8)]
        background = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        expected = prepare_input(frames, background)
        actual = stack_input([resize_rgb(frame) for frame in frames], background)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(actual.flags['C_CONTIGUOUS'])
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_engine_resolution(self):
        self.assertEqual(inference.resolve_engine('auto', Path('a.om')), 'om')
        self.assertEqual(inference.resolve_engine('auto', Path('a.onnx')), 'onnx')
        self.assertEqual(inference.resolve_engine('auto', Path('a.pt')), 'torch')
        self.assertEqual(inference.resolve_engine('onnx'), 'onnx')
        with patch.object(inference, 'acl_available', return_value=False):
            self.assertEqual(inference.resolve_engine('auto'), 'torch')
        args = predict.parse_args(['--engine', 'torch'])
        self.assertEqual(args.model, inference.DEFAULT_CHECKPOINT)

    def test_acl_engine_data_path_and_release(self):
        acl, memory, log = fake_acl()
        with patch.object(inference, 'load_acl', return_value=acl):
            engine = inference.AclEngine('model.om')
            self.assertEqual(engine.seq_len, 8)
            tensor = np.random.default_rng(2).random((27, HEIGHT, WIDTH), dtype=np.float32)
            heatmap = engine(tensor)
            self.assertEqual(heatmap.shape, (HEIGHT, WIDTH))
            np.testing.assert_array_equal(heatmap, tensor[-1])
            self.assertEqual({entry[1] for entry in log if entry[0] == 'memcpy'},
                             {inference.ACL_MEMCPY_DEVICE_TO_DEVICE})
            with self.assertRaises(ValueError):
                engine(tensor[:3])
            engine.close()
            engine.close()
        self.assertEqual(memory, {})
        self.assertEqual([entry[0] for entry in log if entry[0] != 'memcpy'], ['unload', 'reset', 'finalize'])

    def test_acl_engine_rejects_wrong_model_and_cleans_up(self):
        acl, memory, log = fake_acl(input_shape=(1, 27, 288, 288))
        with patch.object(inference, 'load_acl', return_value=acl):
            with self.assertRaisesRegex(ValueError, 'input must be'):
                inference.AclEngine('model.om')
        self.assertEqual(memory, {})
        self.assertIn(('finalize',), log)

    def test_tracker_uses_cached_images(self):
        engine = MagicMock(seq_len=8)
        engine.name = 'fake'
        heatmap = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        heatmap[100:103, 200:203] = .9
        engine.return_value = heatmap
        tracker = predict.Tracker(engine)
        frame = np.zeros((576, 1024, 3), dtype=np.uint8)
        packets = [(i, '', 0.0, frame, resize_rgb(frame)) for i in range(8)]
        background = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.assertEqual(tracker.infer(packets, background, .5)[:3], (1, 402, 202))
        self.assertEqual(engine.call_args[0][0].shape, (27, HEIGHT, WIDTH))
        tracker.close()
        engine.close.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),
                         'onnx/onnxruntime not installed')
    def test_onnx_export_matches_pytorch(self):
        import export_onnx
        with tempfile.TemporaryDirectory(prefix='tracknet-onnx-') as directory:
            path = Path(directory) / 'tracknet.onnx'
            model, seq_len = export_onnx.export(inference.DEFAULT_CHECKPOINT, path, 11)
            engine = inference.OnnxEngine(path)
            self.assertEqual(engine.seq_len, seq_len)
            torch_engine = inference.TorchEngine(inference.DEFAULT_CHECKPOINT, 'cpu')
            tensor = np.random.default_rng(3).random((27, HEIGHT, WIDTH), dtype=np.float32)
            np.testing.assert_allclose(engine(tensor), torch_engine(tensor), atol=1e-4)
            engine.close()


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
