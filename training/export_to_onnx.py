"""Session 8, step 2 of 2 for ONNX export: convert the production model
(already dumped to serving/_export_staging/model.joblib by
training/fetch_production_model.py -- run that first) to ONNX for Triton.

This script deliberately never imports mlflow -- see
fetch_production_model.py's docstring for why importing mlflow and
onnx/skl2onnx/onnxruntime in the same process segfaults on this machine.

Only the current production model is exported (confirmed with the user for
Session 8's scope: production-only, not all 4 model types).

Output:
  serving/triton_model_repo/fraud_rf/1/model.onnx  -- the ONNX graph
  serving/triton_model_repo/fraud_rf/config.pbtxt  -- Triton model config,
      generated from the *actual* ONNX graph's I/O names/shapes rather than
      hand-typed, so it can't drift out of sync with what skl2onnx produced.
  serving/model_metadata.json -- everything serving/app.py needs at request
      time: feature column order, sentinel fill values, category list,
      decision threshold, model version/run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data_prep import (  # noqa: E402
    CATEGORY_COLUMNS,
    FEATURE_COLUMNS,
    build_feature_matrix,
    compute_sentinel_fill_values,
    load_dataset,
    time_split,
)

STAGING_DIR = REPO_ROOT / "serving" / "_export_staging"
TRITON_REPO = REPO_ROOT / "serving" / "triton_model_repo"
TRITON_MODEL_NAME = "fraud_rf"
METADATA_PATH = REPO_ROOT / "serving" / "model_metadata.json"

# Tolerance for the sklearn-vs-onnxruntime sanity check below. Tree
# ensembles converted through skl2onnx run in float32 (ONNX's native
# numeric type), while sklearn scores in float64 -- a tiny, expected
# precision gap, not a conversion bug, as long as it stays this small.
MAX_ALLOWED_PROB_DIFF = 1e-4


def export() -> None:
    model_path = STAGING_DIR / "model.joblib"
    metadata_path = STAGING_DIR / "source_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise SystemExit(
            f"{model_path} / {metadata_path} not found -- run "
            "training/fetch_production_model.py first."
        )

    sk_model = joblib.load(model_path)
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"Loaded production model: v{source_metadata['model_version']} ({source_metadata['algorithm']})")

    n_features = len(FEATURE_COLUMNS)
    onnx_model = convert_sklearn(
        sk_model,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
        options={id(sk_model): {"zipmap": False}},
        target_opset=17,
    )
    onnx.checker.check_model(onnx_model)

    graph_inputs = onnx_model.graph.input
    graph_outputs = onnx_model.graph.output
    if len(graph_inputs) != 1:
        raise SystemExit(f"expected exactly 1 ONNX graph input, got {len(graph_inputs)}")
    # zipmap=False on a binary classifier produces 2 outputs: integer class
    # labels, and a float [N, 2] probability tensor. Serving only needs the
    # probabilities (fraud score = P(class=1)), identified by dtype/rank
    # rather than by a hardcoded name, since skl2onnx's exact output naming
    # has changed across versions.
    prob_output = None
    for out in graph_outputs:
        elem_type = out.type.tensor_type.elem_type
        if elem_type == onnx.TensorProto.FLOAT:
            prob_output = out
            break
    if prob_output is None:
        raise SystemExit(f"couldn't find a float probability output among {[o.name for o in graph_outputs]}")

    input_name = graph_inputs[0].name
    output_name = prob_output.name
    print(f"ONNX graph: input={input_name!r} [N,{n_features}], probability output={output_name!r}")

    # --- Sanity check: onnxruntime output must match sklearn's, on real data ---
    df = load_dataset()
    train_df, _val_df, _test_df = time_split(df)
    fill_values = compute_sentinel_fill_values(train_df)

    sample = build_feature_matrix(train_df.sample(n=500, random_state=42), fill_values)
    sample = sample[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    sk_probs = sk_model.predict_proba(sample.astype(np.float64))[:, 1]

    sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    onnx_out = sess.run([output_name], {input_name: sample})[0]
    onnx_probs = onnx_out[:, 1]

    max_diff = float(np.max(np.abs(sk_probs - onnx_probs)))
    print(f"sklearn-vs-onnx max abs probability diff over 500 sampled rows: {max_diff:.2e}")
    if max_diff > MAX_ALLOWED_PROB_DIFF:
        raise SystemExit(
            f"ONNX conversion diverges from sklearn by {max_diff:.2e}, exceeding "
            f"the {MAX_ALLOWED_PROB_DIFF:.0e} tolerance -- do not serve this export."
        )

    # --- Write Triton model repository layout ---
    version_dir = TRITON_REPO / TRITON_MODEL_NAME / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = version_dir / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    config_pbtxt = f"""\
name: "{TRITON_MODEL_NAME}"
platform: "onnxruntime_onnx"
max_batch_size: 64
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ {n_features} ]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }}
]
instance_group [
  {{
    kind: KIND_CPU
  }}
]
"""
    config_path = TRITON_REPO / TRITON_MODEL_NAME / "config.pbtxt"
    config_path.write_text(config_pbtxt, encoding="utf-8")
    print(f"Wrote {onnx_path.relative_to(REPO_ROOT)} and {config_path.relative_to(REPO_ROOT)}")

    # --- Write serving metadata ---
    metadata = {
        "triton_model_name": TRITON_MODEL_NAME,
        "triton_input_name": input_name,
        "triton_output_name": output_name,
        "n_features": n_features,
        "feature_columns": FEATURE_COLUMNS,
        "category_columns": CATEGORY_COLUMNS,
        "sentinel_fill_values": fill_values,
        "sklearn_vs_onnx_max_abs_diff": max_diff,
        **source_metadata,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {METADATA_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    export()
