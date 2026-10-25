#!/bin/bash
# Build OpenEquivariance 0.7.0's host extension (libtorch_tp_jit) for torch 2.14.1+cu130 and wrap it in a
# `fastmace-oeq-ext` wheel, one per CPython 3.10/3.11/3.12. Runs INSIDE quay.io/pypa/manylinux_2_28_x86_64
# (build.sh starts it). Layout inside the container:
#   /src   this directory (read-only): package skeleton, this script
#   /work  scratch (pip cache, sysroots, build trees), a docker volume
#   /out   the finished wheels
# Nothing is uploaded.
set -euo pipefail

TORCH_VERSION=${TORCH_VERSION:-2.14.1+cu130}
OEQ_VERSION=${OEQ_VERSION:-0.7.0}
PYBIND11_VERSION=${PYBIND11_VERSION:-3.1.0}
EXT_VERSION=${EXT_VERSION:-0.7.0.post1}  # <OEq version>.post<N>: rebuilds of the same OEq
PYTHONS=${PYTHONS:-"cp310-cp310 cp311-cp311 cp312-cp312"}
TORCH_INDEX=https://download.pytorch.org/whl/cu130
# CUDA 13 header/library wheels pinned to the ones torch 2.14.1+cu130 requires (mace_cueq_oeq).
NVIDIA_PINS="nvidia-cuda-runtime==13.0.96 nvidia-cuda-nvrtc==13.0.88 nvidia-cuda-cccl==13.0.85 \
nvidia-cuda-crt==13.0.88 nvidia-cublas==13.1.1.3 nvidia-cusparse==12.6.3.3 nvidia-cusolver==12.0.4.66 \
nvidia-cufft==12.0.0.61 nvidia-curand==10.4.0.35"  # ATen/cuda/CUDAContextLight.h includes their headers

export PIP_CACHE_DIR=/work/pip-cache PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p /work /out
cd /work

# CUDA headers and link libraries, shared by every Python (they are pure C).
if [ ! -f /work/cuda/nvidia/cu13/include/cusparse.h ] || [ ! -f /work/cuda/nvidia/cu13/include/crt/host_defines.h ]; then
  /opt/python/cp312-cp312/bin/python -m pip install --upgrade --no-deps --target /work/cuda $NVIDIA_PINS
fi
CU13=/work/cuda/nvidia/cu13
ls "$CU13/include/cuda.h" "$CU13/include/nvrtc.h" "$CU13/lib/libcudart.so.13" >/dev/null

# OEq sources, from the PyPI sdist/wheel (pure Python wheel carries extension/*.cpp).
if [ ! -d /work/oeq/openequivariance/extension ]; then
  /opt/python/cp312-cp312/bin/python -m pip install --no-deps --target /work/oeq "openequivariance==$OEQ_VERSION"
fi
OEQ=/work/oeq/openequivariance
OEQ_LICENSE=$(ls /work/oeq/openequivariance-$OEQ_VERSION.dist-info/licenses/LICENSE)

# libcuda.so: link against a stub that only names the soname; the driver provides it at run time.
mkdir -p /work/stubs
if [ ! -f /work/stubs/libcuda.so ]; then
  echo 'void cuGetErrorString(void){}' > /work/stubs/cuda_stub.c
  gcc -shared -fPIC -Wl,-soname,libcuda.so.1 -o /work/stubs/libcuda.so /work/stubs/cuda_stub.c
fi
for name in cudart nvrtc cublas; do ln -sf "$CU13/lib/lib$name.so.13" "/work/stubs/lib$name.so"; done

for tag in $PYTHONS; do
  PY=/opt/python/$tag/bin/python
  SYS=/work/sys-$tag
  if [ ! -f "$SYS/torch/version.py" ]; then
    $PY -m pip install --no-deps --target "$SYS" --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
    $PY -m pip install --no-deps --target "$SYS" "pybind11==$PYBIND11_VERSION"
  fi
  PYINC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
  BUILD=/work/build-$tag
  rm -rf "$BUILD"; mkdir -p "$BUILD/pkg"
  # The flags of torch.utils.cpp_extension.load for OEq 0.7.0's load_jit_extension, minus libpython
  # (only OEq's AOTI path needs it; a Python extension must not link it) and with relocatable RUNPATHs.
  CFLAGS="-DTORCH_EXTENSION_NAME=libtorch_tp_jit -DTORCH_API_INCLUDE_EXTENSION_H -I$OEQ/extension/backend \
-I$SYS/pybind11/include -I$SYS/torch/include -I$SYS/torch/include/torch/csrc/api/include -I$CU13/include \
-I$CU13/include/cccl -isystem $PYINC -fPIC -std=c++20 -O3 -DCUDA_BACKEND -D_GLIBCXX_USE_CXX11_ABI=1"
  for src in libtorch_tp_jit.cpp json11/json11.cpp; do
    g++ $CFLAGS -c "$OEQ/extension/$src" -o "$BUILD/$(basename "${src%.cpp}").o"
  done
  g++ "$BUILD/libtorch_tp_jit.o" "$BUILD/json11.o" -shared -Wl,--no-as-needed \
    -L/work/stubs -lcuda -lcudart -lnvrtc -lcublas -L"$SYS/torch/lib" -lc10 -ltorch_cpu -ltorch -ltorch_python \
    '-Wl,-rpath,$ORIGIN/../torch/lib:$ORIGIN/../nvidia/cu13/lib' -o "$BUILD/libtorch_tp_jit.so"
  strip --strip-unneeded "$BUILD/libtorch_tp_jit.so"

  PKG=$BUILD/pkg
  cp -r /src/package/. "$PKG/"
  cp "$BUILD/libtorch_tp_jit.so" "$PKG/fastmace_oeq_ext/"
  cp "$OEQ_LICENSE" "$PKG/fastmace_oeq_ext/LICENSE.openequivariance"
  GLIBC=$(objdump -T "$BUILD/libtorch_tp_jit.so" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
  $PY - "$PKG" "$tag" "$GLIBC" <<EOF
import json, sys, hashlib, platform
pkg, tag, glibc = sys.argv[1:]
so = f"{pkg}/fastmace_oeq_ext/libtorch_tp_jit.so"
manifest = {"extension_version": "$EXT_VERSION", "torch": "$TORCH_VERSION", "openequivariance": "$OEQ_VERSION",
            "cuda": "13.0", "python": tag.split("-")[0], "pybind11": "$PYBIND11_VERSION",
            "glibc_max_symbol": glibc, "platform": "manylinux_2_28_x86_64",
            "compiler": platform.python_compiler(), "library": "libtorch_tp_jit.so",
            "sha256": hashlib.sha256(open(so, "rb").read()).hexdigest()}
json.dump(manifest, open(f"{pkg}/fastmace_oeq_ext/manifest.json", "w"), indent=1, sort_keys=True)
EOF
  g++ --version | head -1 > "$PKG/fastmace_oeq_ext/BUILD_COMPILER.txt"
  sed -i "s/@EXT_VERSION@/$EXT_VERSION/; s/@TORCH_VERSION@/$TORCH_VERSION/" "$PKG/pyproject.toml"
  # Build a py3-none wheel, then retag it for this CPython and manylinux_2_28 (the .so is CPython-ABI specific).
  $PY -m pip wheel --no-deps --no-build-isolation -w "$BUILD/dist" "$PKG" 2>/dev/null \
    || { $PY -m pip install -q "setuptools>=77" wheel && $PY -m pip wheel --no-deps --no-build-isolation -w "$BUILD/dist" "$PKG"; }
  $PY -m pip install -q "wheel>=0.43" >/dev/null
  $PY -m wheel tags --remove --python-tag "${tag%%-*}" --abi-tag "${tag##*-}" \
    --platform-tag manylinux_2_28_x86_64 "$BUILD"/dist/*.whl
  cp "$BUILD"/dist/*manylinux_2_28_x86_64.whl /out/
done
ls -l /out
