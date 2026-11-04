"""Shared TensorRT10 C ABI helpers for engine building and Python inference."""
import ctypes as C
import os


class BuilderConfig(C.Structure):
    _fields_ = [("precision", C.c_int), ("workspace_mib", C.c_int64),
                ("device_id", C.c_int), ("enable_tf32", C.c_int),
                ("enable_sparse_weights", C.c_int), ("strict_types", C.c_int)]


class Profile(C.Structure):
    _fields_ = [(name, C.c_char_p) for name in ["minimum", "optimal", "maximum"]]


def bind(dll, name, restype, *args):
    function = getattr(dll, name)
    function.restype, function.argtypes = restype, list(args)
    return function


def load_runtime(build):
    # Callers retain handles for as long as native libraries are in use.
    paths = [build / "bin", build / "bintrt", build / "builder_runtime", build / "tensorrt_builder"]
    handles = [os.add_dll_directory(str(path)) for path in paths if path.exists()]
    os.environ["PATH"] = ";".join(str(path) for path in paths) + ";" + os.environ.get("PATH", "")
    return handles
