"""Export the official TrackNet checkpoint to a static ONNX model for Ascend ATC / ONNX Runtime.

The exported graph takes input [1, (seq_len + 1) * 3, 288, 512] float32 (background + frames, RGB,
0-1) and outputs only the latest frame heatmap [1, 1, 288, 512], which is all the live entry uses.
Run it on the development PC (PyTorch installed), then copy the ONNX file to the Orange Pi AIpro.
"""
import argparse
import inspect
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from inference import DEFAULT_CHECKPOINT, DEFAULT_ONNX, load_torch_model
from utils.general import HEIGHT, WIDTH


class LatestHeatmap(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network(x)[:, -1:]


def export(checkpoint, output, opset):
    network, seq_len = load_torch_model(checkpoint)
    model = LatestHeatmap(network).eval()
    dummy = torch.zeros(1, (seq_len + 1) * 3, HEIGHT, WIDTH)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = dict(input_names=['input'], output_names=['heatmap'], opset_version=opset, do_constant_folding=True)
    # Newer PyTorch defaults to the dynamo exporter; the TorchScript exporter yields the plain opset-11
    # graph that ATC on CANN handles best.
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        options['dynamo'] = False
    with torch.inference_mode():
        torch.onnx.export(model, dummy, str(output), **options)
    return model, seq_len


def verify(model, seq_len, output):
    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed; skipped numerical check.')
        return
    rng = np.random.default_rng(0)
    sample = rng.random((1, (seq_len + 1) * 3, HEIGHT, WIDTH), dtype=np.float32)
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {'input': sample})[0]
    with torch.inference_mode():
        torch_out = model(torch.from_numpy(sample)).numpy()
    error = float(np.abs(onnx_out - torch_out).max())
    print(f'ONNX Runtime vs PyTorch max abs diff: {error:.2e}')
    if error > 1e-3:
        raise SystemExit('ONNX output differs from PyTorch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, default=DEFAULT_ONNX)
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--no-verify', action='store_true')
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit('Checkpoint missing; run download_weights.py first')
    model, seq_len = export(args.checkpoint, args.output, args.opset)
    print(f'Exported {args.output} (input [1, {(seq_len + 1) * 3}, {HEIGHT}, {WIDTH}], output [1, 1, {HEIGHT}, {WIDTH}])')
    if not args.no_verify:
        verify(model, seq_len, args.output)
    print('Next on the Orange Pi AIpro: bash TrackNetV3/convert_om.sh')
    return 0


if __name__ == '__main__':
    sys.exit(main())
