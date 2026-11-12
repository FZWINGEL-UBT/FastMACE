# deviation 3: the pinned toolchain's sysroot (glibc 2.39) builds binaries that need GLIBC_2.38+; these hosts have
# glibc 2.35 (Ubuntu 22.04 workstations) / 2.34 (Alex, AlmaLinux 9). Use conda-forge's glibc 2.34 sysroot instead.
f=${1:?toolchain_conda_explicit.txt}
sed -i 's|^https://conda.anaconda.org/conda-forge/noarch/sysroot_linux-64-2.39-hc4b9eeb_6.conda#.*|https://conda.anaconda.org/conda-forge/noarch/sysroot_linux-64-2.34-h087de78_3.conda#b04e3a97151611d6a35c8c068b2a03a2|; s|^https://conda.anaconda.org/conda-forge/noarch/kernel-headers_linux-64-6.12.0-he073ed8_6.conda#.*|https://conda.anaconda.org/conda-forge/noarch/kernel-headers_linux-64-5.14.0-he073ed8_3.conda#1e453742ed8f6b4250c66be1dd6ba4fe|' "$f"
grep -E "sysroot|kernel-headers" "$f"
