#!/usr/bin/env python3
import json, struct, sys
from pathlib import Path

# Map variant keys to canonical Swift keys
ALIASES = []
# predictor.lstm
ALIASES += [
    ("predictor.lstm.weight_ih_l0", ["predictor.lstm.Wx_forward", "predictor.lstm.weight_ih_forward"]),
    ("predictor.lstm.weight_hh_l0", ["predictor.lstm.Wh_forward", "predictor.lstm.weight_hh_forward"]),
    ("predictor.lstm.bias_ih_l0", ["predictor.lstm.bias_ih_forward"]),
    ("predictor.lstm.bias_hh_l0", ["predictor.lstm.bias_hh_forward"]),
    ("predictor.lstm.weight_ih_l0_reverse", ["predictor.lstm.Wx_backward", "predictor.lstm.weight_ih_backward"]),
    ("predictor.lstm.weight_hh_l0_reverse", ["predictor.lstm.Wh_backward", "predictor.lstm.weight_hh_backward"]),
    ("predictor.lstm.bias_ih_l0_reverse", ["predictor.lstm.bias_ih_backward"]),
    ("predictor.lstm.bias_hh_l0_reverse", ["predictor.lstm.bias_hh_backward"]),
]
# predictor.text_encoder.lstms
for i in range(6):
    base = f"predictor.text_encoder.lstms.{i}"
    ALIASES += [
        (f"{base}.weight_ih_l0", [f"{base}.Wx_forward", f"{base}.weight_ih_forward"]),
        (f"{base}.weight_hh_l0", [f"{base}.Wh_forward", f"{base}.weight_hh_forward"]),
        (f"{base}.bias_ih_l0", [f"{base}.bias_ih_forward"]),
        (f"{base}.bias_hh_l0", [f"{base}.bias_hh_forward"]),
        (f"{base}.weight_ih_l0_reverse", [f"{base}.Wx_backward", f"{base}.weight_ih_backward"]),
        (f"{base}.weight_hh_l0_reverse", [f"{base}.Wh_backward", f"{base}.weight_hh_backward"]),
        (f"{base}.bias_ih_l0_reverse", [f"{base}.bias_ih_backward"]),
        (f"{base}.bias_hh_l0_reverse", [f"{base}.bias_hh_backward"]),
    ]
# shared lstm
ALIASES += [
    ("predictor.shared.weight_ih_l0", ["predictor.shared.Wx_forward", "predictor.shared.weight_ih_forward"]),
    ("predictor.shared.weight_hh_l0", ["predictor.shared.Wh_forward", "predictor.shared.weight_hh_forward"]),
    ("predictor.shared.bias_ih_l0", ["predictor.shared.bias_ih_forward"]),
    ("predictor.shared.bias_hh_l0", ["predictor.shared.bias_hh_forward"]),
    ("predictor.shared.weight_ih_l0_reverse", ["predictor.shared.Wx_backward", "predictor.shared.weight_ih_backward"]),
    ("predictor.shared.weight_hh_l0_reverse", ["predictor.shared.Wh_backward", "predictor.shared.weight_hh_backward"]),
    ("predictor.shared.bias_ih_l0_reverse", ["predictor.shared.bias_ih_backward"]),
    ("predictor.shared.bias_hh_l0_reverse", ["predictor.shared.bias_hh_backward"]),
]
# conv1x1 naming variants
ALIASES += [
    ("predictor.F0_proj.weight", ["predictor.F0_proj.linear_layer.weight"]),
    ("predictor.F0_proj.bias", ["predictor.F0_proj.linear_layer.bias"]),
    ("predictor.N_proj.weight", ["predictor.N_proj.linear_layer.weight"]),
    ("predictor.N_proj.bias", ["predictor.N_proj.linear_layer.bias"]),
]
# text_encoder layernorm
for i in range(3):
    ALIASES += [
        (f"text_encoder.cnn.{i}.1.gamma", [f"text_encoder.cnn.{i}.1.weight"]),
        (f"text_encoder.cnn.{i}.1.beta", [f"text_encoder.cnn.{i}.1.bias"]),
    ]
# text_encoder lstm
ALIASES += [
    ("text_encoder.lstm.weight_ih_l0", ["text_encoder.lstm.Wx_forward", "text_encoder.lstm.weight_ih_forward"]),
    ("text_encoder.lstm.weight_hh_l0", ["text_encoder.lstm.Wh_forward", "text_encoder.lstm.weight_hh_forward"]),
    ("text_encoder.lstm.bias_ih_l0", ["text_encoder.lstm.bias_ih_forward"]),
    ("text_encoder.lstm.bias_hh_l0", ["text_encoder.lstm.bias_hh_forward"]),
    ("text_encoder.lstm.weight_ih_l0_reverse", ["text_encoder.lstm.Wx_backward", "text_encoder.lstm.weight_ih_backward"]),
    ("text_encoder.lstm.weight_hh_l0_reverse", ["text_encoder.lstm.Wh_backward", "text_encoder.lstm.weight_hh_backward"]),
    ("text_encoder.lstm.bias_ih_l0_reverse", ["text_encoder.lstm.bias_ih_backward"]),
    ("text_encoder.lstm.bias_hh_l0_reverse", ["text_encoder.lstm.bias_hh_backward"]),
]

# Note: We rely on mlx's safetensors for I/O via Python if available; otherwise operate on header only is insufficient.
# For simplicity, we re-map keys in header only if you want to avoid full tensor rewrite. Better: use mlx or safetensors libs.

try:
    import safetensors.numpy as stnp
    import numpy as np
except Exception as e:
    print("Please pip install safetensors and numpy.")
    sys.exit(1)

if len(sys.argv) != 3:
    print("Usage: normalize_kokoro_weights.py <input.safetensors> <output.safetensors>")
    sys.exit(1)

src, dst = map(Path, sys.argv[1:])

# Load tensors
tensors = stnp.load_file(str(src))

# Build normalized dict
out = dict(tensors)

# Helper to alias
for target, cands in ALIASES:
    if target in out:
        continue
    for c in cands:
        if c in out:
            out[target] = out[c]
            break

# Save
stnp.save_file(out, str(dst), metadata={"format": "mlx"})
print(f"Wrote normalized weights to {dst}")
