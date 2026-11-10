"""Dry-run check: the mlx-lm baseline model reproduces the tiny fixture's oracle logits."""

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model

import ricursive_mlx

# The CPU computes float32 exactly enough for a tight bound; GPU matmuls do not.
mx.set_default_device(mx.cpu)

fixture = Path(sys.argv[1])
model, _ = load_model(
    fixture, get_model_classes=lambda config: (ricursive_mlx.Model, ricursive_mlx.ModelArgs)
)
oracle = json.loads((fixture / "oracle.json").read_text())
logits = model(mx.array([oracle["prompt"]]))[0]
difference = mx.abs(logits - mx.array(oracle["full_logits"])).max().item()
print(f"mlx-lm baseline model: max |dlogit| = {difference:.2e}")
sys.exit(0 if difference < 1e-4 else 1)
