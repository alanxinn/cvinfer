"""Regression tests for TensorRT 10 dynamic shapes, INT64 and the builder C ABI."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess

from validate_trt10 import BuilderConfig, bind, load_runtime


class Profile(C.Structure):
    _fields_ = [(name, C.c_char_p) for name in ["minimum", "optimal", "maximum"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--output", type=Path, default=Path("cache/trt10-contract"))
    a = p.parse_args()
    import numpy as np
    import onnx
    from onnx import helper as h, TensorProto as T
    build, root = a.build.resolve(), a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    handles = load_runtime(build)
    dll = C.CDLL(str(build / "tensorrt_builder/alanxinn_tensorrt_builder.dll"))
    config = BuilderConfig()
    bind(dll, "trt_builder_config_init", None, C.POINTER(BuilderConfig))(C.byref(config))
    config.enable_tf32 = 0
    run = bind(dll, "trt_builder_build_onnx", C.c_int, C.c_char_p, C.c_char_p,
               C.POINTER(BuilderConfig), C.POINTER(Profile), C.c_size_t)
    error = bind(dll, "trt_builder_last_error", C.c_char_p)
    for name, dtype, shape in [("dynamic", T.FLOAT, ["batch", 3]), ("int64", T.INT64, [1, 4])]:
        graph = h.make_graph([h.make_node("Identity", ["input"], ["output"])], name,
                             [h.make_tensor_value_info("input", dtype, shape)],
                             [h.make_tensor_value_info("output", dtype, shape)])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        onnx.save(model, root / (name + ".onnx"))
    profile = Profile(b"input:1x3", b"input:2x3", b"input:4x3")
    dynamic_onnx = str(root / "dynamic.onnx").encode()
    dynamic_engine = str(root / "dynamic.engine").encode()
    assert run(dynamic_onnx, dynamic_engine, C.byref(config), None, 0) == 1
    assert b"require at least one optimization profile" in error()
    invalid = Profile(b"input:4x3", b"input:2x3", b"input:1x3")
    assert run(dynamic_onnx, dynamic_engine, C.byref(config), C.byref(invalid), 1) == 1
    assert b"min <= opt <= max" in error()
    assert run(dynamic_onnx, dynamic_engine, C.byref(config), C.byref(profile), 1) == 0, error().decode()
    assert run(str(root / "int64.onnx").encode(), str(root / "int64.engine").encode(),
               C.byref(config), None, 0) == 0, error().decode()
    # Do not retain a loaded builder when rebuilding the test executable.
    done = subprocess.run([str(build / "bin/trt_session_contract.exe"), str(root / "dynamic.engine")],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    values = np.array([[0, 2 ** 35, -(2 ** 35), 42]], dtype=np.int64)
    values.tofile(root / "input-int64.bin")
    out = root / "int64-raw"
    done = subprocess.run([str(build / "bin/trt_raw_outputs.exe"), str(root / "int64.engine"),
                           str(root / "input-int64.bin"), str(out)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    parts = (out / "outputs.tsv").read_text().strip().split("\t")
    assert parts[2] == "INT64"
    actual = np.fromfile(out / parts[1], dtype=np.int64)
    np.testing.assert_array_equal(actual, values.ravel())
    report = {"status": "PASS", "checks": ["missing dynamic profile rejection", "invalid profile rejection",
              "dynamic dimensions 1/4/2", "host/device buffer reuse", "host copy toggle", "bad input size rejection",
              "output shape mutation rejection", "INT64 values exceeding INT32"]}
    (root / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
