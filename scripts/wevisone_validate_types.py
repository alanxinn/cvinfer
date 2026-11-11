"""Regression: FP16 inputs must retain FP32 RMS normalization in typed engines."""
import ctypes as C
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h
from trt10_runtime import BuilderConfig, bind, load_runtime


def main():
    root = Path("cache/wevisone/validation/types").resolve()
    root.mkdir(parents=True, exist_ok=True)
    build = Path("build/trt10-release").resolve()
    handles = load_runtime(build)
    dll = C.CDLL(str(build / "tensorrt_builder/alanxinn_tensorrt_builder.dll"))
    run = bind(dll, "trt_builder_build_onnx", C.c_int, C.c_char_p, C.c_char_p, C.POINTER(BuilderConfig), C.c_void_p, C.c_size_t)
    error = bind(dll, "trt_builder_last_error", C.c_char_p)
    nodes = [h.make_node("Cast", ["input"], ["x"], to=T.FLOAT), h.make_node("Mul", ["x", "x"], ["square"]),
             h.make_node("ReduceMean", ["square", "axes"], ["mean"], keepdims=1),
             h.make_node("Add", ["mean", "eps"], ["variance"]), h.make_node("Sqrt", ["variance"], ["denominator"]),
             h.make_node("Div", ["x", "denominator"], ["norm"]), h.make_node("Cast", ["norm"], ["half"], to=T.FLOAT16),
             h.make_node("Cast", ["half"], ["output"], to=T.FLOAT)]
    graph = h.make_graph(nodes, "fp32_norm_in_fp16", [h.make_tensor_value_info("input", T.FLOAT16, [1, 2])],
                         [h.make_tensor_value_info("output", T.FLOAT, [1, 2])],
                         initializer=[h.make_tensor("axes", T.INT64, [1], [1]), h.make_tensor("eps", T.FLOAT, [], [1e-6])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, root / "normalization.onnx")
    values = np.array([[1000, 600]], dtype=np.float16)
    values.tofile(root / "input.bin")
    expected = (values.astype(np.float32) / np.sqrt(np.mean(values.astype(np.float32) ** 2, axis=-1, keepdims=True) + 1e-6)).astype(np.float16).astype(np.float32)
    results = []
    for name, precision, strict in [("legacy_fp16", 1, 0), ("strongly_typed", 0, 2)]:
        config = BuilderConfig()
        bind(dll, "trt_builder_config_init", None, C.POINTER(BuilderConfig))(C.byref(config))
        config.precision, config.strict_types, config.enable_tf32, config.workspace_mib = precision, strict, 0, 256
        status = run(str(root / "normalization.onnx").encode(), str(root / (name + ".engine")).encode(), C.byref(config), None, 0)
        if status:
            raise RuntimeError(error().decode())
        destination = root / name
        subprocess.run([str(build / "bin/trt_raw_outputs.exe"), str(root / (name + ".engine")), str(root / "input.bin"), str(destination)], check=True, capture_output=True)
        parts = (destination / "outputs.tsv").read_text().strip().split("\t")
        actual = np.fromfile(destination / parts[1], dtype=np.float32).reshape(1, 2)
        difference = float(np.max(np.abs(actual - expected)))
        results.append({"mode": name, "expected": expected.tolist(), "actual": actual.tolist(), "max_abs": difference})
        if strict == 2:
            np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=0)
    (root / "report.json").write_text(json.dumps({"status": "PASS", "checks": results}, indent=2), encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
