"""TrackNet inference engines.

- torch: PyTorch on CUDA or CPU (Windows development PC).
- om:    Ascend OM model through pyACL on the Orange Pi AIpro NPU (Ascend 310B).
- onnx:  ONNX Runtime on CPU, a fallback for ARM boards without a converted OM model.

Every engine takes a (C, 288, 512) float32 array and returns the latest frame heatmap (288, 512).
"""
import ctypes
import importlib.util
from pathlib import Path

import numpy as np

from utils.general import HEIGHT, WIDTH

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / 'ckpts' / 'TrackNet_best.pt'
DEFAULT_ONNX = ROOT / 'ckpts' / 'TrackNet_best.onnx'
DEFAULT_OM = ROOT / 'ckpts' / 'TrackNet_best.om'
ENGINES = ('auto', 'torch', 'om', 'onnx')


def seq_len_from_channels(channels):
    channels = int(channels)
    if channels < 6 or channels % 3:
        raise ValueError(f'Unexpected TrackNet input channels: {channels}')
    return channels // 3 - 1


def check_input_shape(shape, source):
    shape = tuple(int(value) for value in shape)
    if len(shape) != 4 or shape[0] != 1 or shape[2:] != (HEIGHT, WIDTH):
        raise ValueError(f'{source} input must be [1, C, {HEIGHT}, {WIDTH}], got {list(shape)}')
    return seq_len_from_channels(shape[1])


def last_heatmap(output):
    """Accept full (1, 8, H, W) or last-frame-only (1, 1, H, W) outputs."""
    return np.asarray(output, dtype=np.float32).reshape(-1, HEIGHT, WIDTH)[-1]


def acl_available():
    return importlib.util.find_spec('acl') is not None


def resolve_engine(engine, model=None):
    """Pick a concrete engine. auto prefers the AIpro NPU when an OM model and pyACL exist."""
    if engine != 'auto':
        return engine
    if model is not None:
        suffix = Path(model).suffix.lower()
        return {'.om': 'om', '.onnx': 'onnx'}.get(suffix, 'torch')
    if DEFAULT_OM.is_file() and acl_available():
        return 'om'
    return 'torch'


def default_model(engine):
    return {'torch': DEFAULT_CHECKPOINT, 'om': DEFAULT_OM, 'onnx': DEFAULT_ONNX}[engine]


def load_torch_model(checkpoint):
    """Load the official bg_mode=concat checkpoint on CPU; shared by inference and ONNX export."""
    import torch
    from model import TrackNet
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    params = state['param_dict']
    seq_len = int(params['seq_len'])
    if params['bg_mode'] != 'concat':
        raise ValueError('This live entry requires the official bg_mode=concat checkpoint')
    network = TrackNet((seq_len + 1) * 3, seq_len)
    network.load_state_dict(state['model'])
    return network.eval(), seq_len


class TorchEngine:
    def __init__(self, checkpoint, device_name='auto', threads=0):
        import torch
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_name == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        self.device = torch.device(device_name)
        network, self.seq_len = load_torch_model(checkpoint)
        self.model = network.to(self.device)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        self.name = f'torch-{self.device.type}'

    def __call__(self, tensor):
        with self.torch.inference_mode():
            batch = self.torch.from_numpy(tensor).unsqueeze(0).to(self.device)
            return self.model(batch)[0, -1].float().cpu().numpy()

    def close(self):
        pass


class OnnxEngine:
    def __init__(self, path, threads=0):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError('onnxruntime missing. Install it with: pip install onnxruntime') from error
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.seq_len = check_input_shape(model_input.shape, 'ONNX model')
        self.name = 'onnxruntime-cpu'

    def __call__(self, tensor):
        return last_heatmap(self.session.run(None, {self.input_name: tensor[np.newaxis]})[0])

    def close(self):
        self.session = None


# pyACL constants (CANN acl.h).
ACL_SUCCESS = 0
ACL_DEVICE = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


def load_acl():
    try:
        import acl
    except ImportError as error:
        raise RuntimeError('pyACL missing. Install CANN and run: '
                           'source /usr/local/Ascend/ascend-toolkit/set_env.sh') from error
    return acl


def acl_check(ret, action):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f'Ascend ACL {action} failed with error code {ret}')


class AclEngine:
    """Run an ATC-converted OM model on the Ascend NPU with pyACL (single float32 NCHW input)."""

    def __init__(self, path, device_id=0):
        self.acl = acl = load_acl()
        self.device_id = device_id
        self.initialized = self.device_set = False
        self.context = self.model_id = self.desc = None
        self.input_dataset = self.output_dataset = None
        self.inputs, self.outputs = [], []
        self.host_output = None
        self.name = f'ascend-npu{device_id}'
        try:
            acl_check(acl.init(), 'init')
            self.initialized = True
            acl_check(acl.rt.set_device(device_id), 'set_device')
            self.device_set = True
            self.context, ret = acl.rt.create_context(device_id)
            acl_check(ret, 'create_context')
            run_mode, ret = acl.rt.get_run_mode()
            acl_check(ret, 'get_run_mode')
            # On the AIpro the program runs on the device side, where CPU and NPU share memory
            # and CANN requires DEVICE_TO_DEVICE copies.
            device_side = run_mode == ACL_DEVICE
            self.copy_in = ACL_MEMCPY_DEVICE_TO_DEVICE if device_side else ACL_MEMCPY_HOST_TO_DEVICE
            self.copy_out = ACL_MEMCPY_DEVICE_TO_DEVICE if device_side else ACL_MEMCPY_DEVICE_TO_HOST
            self.model_id, ret = acl.mdl.load_from_file(str(path))
            acl_check(ret, f'load model {path}')
            self.desc = acl.mdl.create_desc()
            acl_check(acl.mdl.get_desc(self.desc, self.model_id), 'get_desc')
            if acl.mdl.get_num_inputs(self.desc) != 1:
                raise ValueError('OM model must have exactly one input')
            dims, ret = acl.mdl.get_input_dims(self.desc, 0)
            acl_check(ret, 'get_input_dims')
            shape = tuple(dims['dims'])
            self.seq_len = check_input_shape(shape, 'OM model')
            self.input_dataset = acl.mdl.create_dataset()
            self._add_buffers(self.input_dataset, self.inputs, 1, acl.mdl.get_input_size_by_index)
            self.output_dataset = acl.mdl.create_dataset()
            self._add_buffers(self.output_dataset, self.outputs, acl.mdl.get_num_outputs(self.desc),
                              acl.mdl.get_output_size_by_index)
            self.input_bytes = int(np.prod(shape)) * 4
            if self.inputs[0][1] != self.input_bytes:
                raise ValueError('OM input is not float32 NCHW; rebuild it with convert_om.sh')
            self.output_bytes = self.outputs[0][1]
            if self.output_bytes % (HEIGHT * WIDTH * 4):
                raise ValueError('OM output is not float32 heatmaps; rebuild it with convert_om.sh')
            self.host_output, ret = acl.rt.malloc_host(self.output_bytes)
            acl_check(ret, 'malloc_host')
        except Exception:
            self.close()
            raise

    def _add_buffers(self, dataset, buffers, count, size_of):
        acl = self.acl
        for index in range(count):
            size = size_of(self.desc, index)
            pointer, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            acl_check(ret, 'malloc')
            data = acl.create_data_buffer(pointer, size)
            buffers.append((pointer, size, data))
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)
            acl_check(ret, 'add_dataset_buffer')

    def __call__(self, tensor):
        acl = self.acl
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if tensor.nbytes != self.input_bytes:
            raise ValueError(f'Input has {tensor.nbytes} bytes, OM model expects {self.input_bytes}')
        # Inference may run on a different thread from the one that created the context.
        acl_check(acl.rt.set_context(self.context), 'set_context')
        pointer, size = self.inputs[0][:2]
        acl_check(acl.rt.memcpy(pointer, size, tensor.ctypes.data, tensor.nbytes, self.copy_in), 'memcpy input')
        acl_check(acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset), 'execute')
        pointer = self.outputs[0][0]
        acl_check(acl.rt.memcpy(self.host_output, self.output_bytes, pointer, self.output_bytes, self.copy_out),
                  'memcpy output')
        values = (ctypes.c_float * (self.output_bytes // 4)).from_address(self.host_output)
        return last_heatmap(np.ctypeslib.as_array(values)).copy()

    def close(self):
        acl = self.acl
        for dataset, buffers in ((self.input_dataset, self.inputs), (self.output_dataset, self.outputs)):
            for pointer, _, data in buffers:
                acl.destroy_data_buffer(data)
                acl.rt.free(pointer)
            buffers.clear()
            if dataset is not None:
                acl.mdl.destroy_dataset(dataset)
        self.input_dataset = self.output_dataset = None
        if self.host_output is not None:
            acl.rt.free_host(self.host_output)
            self.host_output = None
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None
        if self.desc is not None:
            acl.mdl.destroy_desc(self.desc)
            self.desc = None
        if self.context is not None:
            acl.rt.destroy_context(self.context)
            self.context = None
        if self.device_set:
            acl.rt.reset_device(self.device_id)
            self.device_set = False
        if self.initialized:
            acl.finalize()
            self.initialized = False


def create_engine(engine, model=None, device='auto', npu_id=0, threads=0):
    engine = resolve_engine(engine, model)
    path = Path(model) if model is not None else default_model(engine)
    if not path.is_file():
        hints = {'torch': 'run download_weights.py', 'onnx': 'run export_onnx.py',
                 'om': 'run export_onnx.py and convert_om.sh'}
        raise RuntimeError(f'Model file missing: {path}; {hints[engine]}')
    if engine == 'torch':
        return TorchEngine(path, device, threads)
    if engine == 'onnx':
        return OnnxEngine(path, threads)
    return AclEngine(path, npu_id)
