"""Live camera shuttlecock tracking with the pretrained TrackNetV3 detector."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import torch

from model import TrackNet
from utils.general import HEIGHT, WIDTH, locate, prepare_input, resize_rgb

WINDOW = 'TrackNetV3 Camera | Q: quit | B: recalibrate background'


def camera_source(value):
    if value.lower() == 'realsense':
        return 'realsense'
    if value.isdecimal():
        return int(value)
    if value.lower().startswith(('rtsp://', 'rtsps://', 'http://', 'https://')):
        return value
    raise argparse.ArgumentTypeError('Use realsense, a camera index (0, 1, ...) or RTSP/HTTP stream URL')


def open_camera(source, backend, width, height, fps, serial=None):
    if source == 'realsense':
        from realsense_camera import RealSenseCapture
        return RealSenseCapture(width, height, fps, serial=serial)
    api = {'auto': cv2.CAP_ANY, 'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF}[backend]
    if backend == 'auto' and os.name == 'nt' and isinstance(source, int):
        api = cv2.CAP_DSHOW
    if isinstance(source, str):
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    else:
        capture = cv2.VideoCapture(source, api)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError('Cannot open camera. Check connection, camera index, permissions and other camera apps.')
    if isinstance(source, int):
        for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height),
                            (cv2.CAP_PROP_FPS, fps)):
            capture.set(prop, value)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class CameraStream:
    """Keep only the most recent consecutive sequence; never accumulate a backlog."""
    def __init__(self, capture, seq_len):
        self.capture = capture
        self.frames = deque(maxlen=seq_len)
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._read, name='camera-capture', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read(self):
        index = 0
        try:
            while not self.stopped.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError('Camera disconnected or frame read failed')
                # Timestamp is the host receipt time, not the sensor exposure time.
                packet = (index, datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                          time.perf_counter(), frame)
                with self.condition:
                    self.frames.append(packet)
                    self.condition.notify_all()
                index += 1
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.capture.release()

    def snapshot(self, after=-1, full=True, timeout=0.2):
        with self.condition:
            self.condition.wait_for(lambda: self.error is not None or self.stopped.is_set() or
                                    (len(self.frames) > 0 and self.frames[-1][0] > after and
                                     (not full or len(self.frames) == self.frames.maxlen)), timeout)
            if self.error:
                raise self.error
            if not self.frames or self.frames[-1][0] <= after:
                return None
            if full and len(self.frames) < self.frames.maxlen:
                return None
            return list(self.frames)

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            # Some camera drivers block inside read until capture is released.
            self.capture.release()
            self.thread.join(timeout=1)


class Tracker:
    def __init__(self, checkpoint, device_name='auto'):
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_name == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        self.device = torch.device(device_name)
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
        params = checkpoint['param_dict']
        self.seq_len = int(params['seq_len'])
        if params['bg_mode'] != 'concat':
            raise ValueError('This live entry requires the official bg_mode=concat checkpoint')
        self.model = TrackNet((self.seq_len + 1) * 3, self.seq_len)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device).eval()
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        # Complete CUDA initialization and convolution tuning before opening capture.
        with torch.inference_mode():
            dummy = torch.zeros(1, (self.seq_len + 1) * 3, HEIGHT, WIDTH, device=self.device)
            self.model(dummy)[0, -1, 0, 0].item()

    @torch.inference_mode()
    def infer(self, packets, background, threshold):
        frames = [packet[3] for packet in packets]
        tensor = torch.from_numpy(prepare_input(frames, background)).unsqueeze(0).to(self.device)
        # Use the latest output only; future frames and offline ensembles are not required.
        heatmap = self.model(tensor)[0, -1].float().cpu().numpy()
        return locate(heatmap, frames[-1].shape, threshold)


def key_pressed(headless):
    return -1 if headless else cv2.waitKey(1) & 0xff


def run(args):
    tracker = Tracker(args.checkpoint, args.device)
    print(f'Device: {tracker.device}; sequence: {tracker.seq_len} frames', flush=True)
    capture = open_camera(args.source, args.backend, args.width, args.height, args.fps, serial=args.serial)
    if hasattr(capture, 'info'):
        info = capture.info
        print(f"Camera: {info['name']}; serial: {info['serial']}; USB: {info['usb']}", flush=True)
    print(f'Capture: {capture.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x'
          f'{capture.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}, '
          f'reported FPS: {capture.get(cv2.CAP_PROP_FPS):.1f}', flush=True)
    stream = CameraStream(capture, tracker.seq_len).start()
    background_samples = []
    background = None
    trail = deque(maxlen=args.traj_len)
    last_frame, results = -1, 0
    previous_shape = None
    csv_file = None
    writer = None
    started = time.perf_counter()
    last_result = None
    try:
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_file = args.csv.open('w', newline='', encoding='utf-8')
            writer = csv.writer(csv_file)
            writer.writerow(['Frame', 'TimestampUTC', 'Visibility', 'X', 'Y', 'PeakScore',
                             'InferenceMs', 'ResultAgeMs'])
        print('Collecting background. Keep camera fixed; Q quits, B recalibrates.', flush=True)
        while True:
            packets = stream.snapshot(after=last_frame, full=background is not None)
            if not packets:
                if time.perf_counter() - started > 10:
                    raise RuntimeError('Camera has not provided a new frame for 10 seconds')
                if key_pressed(args.headless) in (ord('q'), 27):
                    break
                continue
            started = time.perf_counter()
            index, timestamp, received, frame = packets[-1]
            last_frame = index
            if previous_shape is not None and previous_shape != frame.shape:
                background, background_samples = None, []
                trail.clear()
            previous_shape = frame.shape
            display = frame.copy()
            if background is None:
                background_samples.append(resize_rgb(frame))
                label = f'Background: {len(background_samples)}/{args.background_frames}'
                if len(background_samples) >= args.background_frames:
                    background = np.median(np.stack(background_samples), axis=0).astype(np.uint8)
                    background_samples.clear()
                    print('Background ready.', flush=True)
            else:
                # Do not pass mixed image sizes to the network after camera renegotiation.
                if any(packet[3].shape != frame.shape for packet in packets):
                    continue
                begin = time.perf_counter()
                visible, x, y, score = tracker.infer(packets, background, args.threshold)
                finished = time.perf_counter()
                inference_ms, age_ms = (finished - begin) * 1000, (finished - received) * 1000
                rate = 0 if last_result is None else 1 / (finished - last_result)
                last_result = finished
                trail.append((x, y) if visible else None)
                for point in trail:
                    if point is not None:
                        cv2.circle(display, point, 3, (0, 255, 255), -1)
                if visible:
                    cv2.circle(display, (x, y), 9, (0, 0, 255), 2)
                label = f'{rate:.1f} results/s | inference {inference_ms:.0f} ms | age {age_ms:.0f} ms'
                row = [index, timestamp, visible, x, y, round(score, 4), round(inference_ms, 1), round(age_ms, 1)]
                if writer:
                    writer.writerow(row)
                    csv_file.flush()
                print(f'frame={index} visible={visible} x={x} y={y} score={score:.3f} age_ms={age_ms:.1f}', flush=True)
                results += 1
            if not args.headless:
                cv2.putText(display, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                cv2.imshow(WINDOW, display)
            key = key_pressed(args.headless)
            if key in (ord('q'), 27):
                break
            if not args.headless and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('b'):
                background, background_samples = None, []
                trail.clear()
                last_result = None
                print('Recalibrating background...', flush=True)
            if args.max_results and results >= args.max_results:
                break
    finally:
        stream.close()
        if csv_file:
            csv_file.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=camera_source, default='realsense', help='realsense (default), camera index or RTSP/HTTP URL')
    parser.add_argument('--serial', help='Select RealSense device by serial number')
    parser.add_argument('--list-devices', action='store_true', help='List RealSense devices and exit without opening streams')
    parser.add_argument('--backend', choices=['auto', 'dshow', 'msmf'], default='auto')
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parent / 'ckpts/TrackNet_best.pt')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30, help='Requested capture FPS; camera may choose a different rate')
    parser.add_argument('--background-frames', type=int, default=60)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--traj-len', type=int, default=16)
    parser.add_argument('--csv', type=Path, help='Optional coordinate output CSV, overwritten if it exists')
    parser.add_argument('--headless', action='store_true', help='Disable preview; Ctrl+C exits')
    parser.add_argument('--max-results', type=int, default=0, help='Stop after N predictions; 0 runs continuously')
    args = parser.parse_args(argv)
    if args.serial and args.source != 'realsense':
        parser.error('--serial is only valid for --source realsense')
    if min(args.width, args.height, args.fps, args.background_frames, args.traj_len) < 1 or args.max_results < 0:
        parser.error('Sizes, FPS and frame counts must be positive; max-results must be nonnegative')
    if not 0 < args.threshold < 1:
        parser.error('threshold must be between 0 and 1')
    if not args.list_devices and not args.checkpoint.is_file():
        parser.error('Checkpoint missing; run download_weights.py first')
    return args


def main():
    try:
        args = parse_args()
        if args.list_devices:
            from realsense_camera import list_devices
            devices = list_devices()
            for info in devices:
                print(f"{info['name']} | serial={info['serial']} | USB={info['usb']}")
            if not devices:
                print('No RealSense devices found.')
            return 0
        run(args)
    except KeyboardInterrupt:
        print('\nStopped.')
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
