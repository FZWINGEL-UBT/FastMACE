"""Read a torch.save'd MACE checkpoint without torch: stub every class, rebuild tensors as NumPy arrays.

Usage (python -I): load(path) -> root object; Stub objects keep their pickled state in __dict__.
"""
import collections
import pickle
import zipfile

import numpy as np

_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class Stub:
    _qualname = "?"

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._args = args
        obj._kwargs = kwargs
        return obj

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            if isinstance(state[0], dict):
                self.__dict__.update(state[0])
            self.__dict__.update(state[1])
        else:
            self._state = state

    def __repr__(self):
        return f"<{self._qualname}>"

    # e3nn Irreps and similar are tuples/lists in their reduce args
    def __iter__(self):
        return iter(getattr(self, "_args", ()))


_classes = {}


def _stub(module, name):
    key = (module, name)
    if key not in _classes:
        _classes[key] = type(name, (Stub,), {"_qualname": f"{module}.{name}"})
    return _classes[key]


def _rebuild_tensor(storage, offset, size, stride, *rest):
    data, dtype = storage
    itemsize = np.dtype(dtype).itemsize
    if len(size) == 0:
        return np.array(data[offset])
    return np.lib.stride_tricks.as_strided(data[offset:], shape=tuple(size),
                                           strides=tuple(s * itemsize for s in stride)).copy()


def _rebuild_parameter(data, requires_grad=False, hooks=None, *rest):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, zf, prefix):
        super().__init__(file)
        self.zf, self.prefix, self.cache = zf, prefix, {}

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name.startswith("_rebuild_parameter"):
            return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module.startswith("numpy") or module in ("builtins", "copyreg", "_codecs"):
            return super().find_class(module, name)
        if module == "torch" and name in _DTYPES:
            return _stub(module, name)
        return _stub(module, name)

    def persistent_load(self, pid):
        typename, storage_type, key, location, numel = pid[:5]
        name = getattr(storage_type, "__name__", "")
        dtype = _DTYPES.get(name)
        if dtype is None:  # new-style: storage_type is torch.<dtype> stub; ask the dtype name
            dtype = {"float32": np.float32, "float": np.float32, "float64": np.float64,
                     "double": np.float64, "int64": np.int64, "long": np.int64,
                     "bool": np.bool_, "int32": np.int32}.get(name, np.float32)
        if key not in self.cache:
            raw = self.zf.read(f"{self.prefix}/data/{key}")
            self.cache[key] = (np.frombuffer(raw, dtype=dtype), dtype)
        return self.cache[key]


def load(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("/data.pkl")]
    with zf.open(pkl) as f:
        return _Unpickler(f, zf, prefix).load()


def modules(obj):
    return getattr(obj, "_modules", {}) or {}


def params(obj):
    out = {}
    for k, v in (getattr(obj, "_parameters", {}) or {}).items():
        out[k] = v
    for k, v in (getattr(obj, "_buffers", {}) or {}).items():
        out[k] = v
    return out


def tree(obj, name="model", depth=0, max_depth=3, out=None):
    out = [] if out is None else out
    ps = {k: getattr(v, "shape", v) for k, v in params(obj).items()}
    out.append("  " * depth + f"{name}: {getattr(obj, '_qualname', type(obj).__name__)} {ps if ps else ''}")
    if depth < max_depth:
        for k, v in modules(obj).items():
            tree(v, k, depth + 1, max_depth, out)
    return out
