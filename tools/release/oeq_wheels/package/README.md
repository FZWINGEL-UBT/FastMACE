# fastmace-oeq-ext

OpenEquivariance 0.7.0's host extension (`libtorch_tp_jit.so`), prebuilt for torch 2.14.1+cu130 on
Linux x86_64 (manylinux_2_28, glibc 2.28 or newer), CPython 3.10 to 3.12. Installed by
`pip install "fastmace[oeq]"`, it lets FastMACE use the hybrid cuEquivariance/OpenEquivariance backend
without g++, ninja or nvcc. FastMACE checks `manifest.json` and falls back to building the extension
itself when torch or OpenEquivariance differ from the versions it was built for.

The library is built unmodified from OpenEquivariance's sources (BSD 3-Clause, The Regents of the
University of California, LBNL; see `LICENSE.openequivariance`). The packaging is MIT (FastMACE).
