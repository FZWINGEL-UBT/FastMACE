# Vendored data

`dftd3_params.npz` is `torch_dftd/nn/params/dftd3_params.npz` of torch-dftd
0.5.1 (https://github.com/pfnet/torch-dftd, Preferred Networks, Inc.),
unchanged: 690,158 bytes, SHA-256
`52b7b35a957df55b813035ffde3a6b5a41dec585bdb1e281fb8ba67a729f3d92`
(`fastmace.longrange.d3.REFERENCE_TABLES`). It holds Grimme's D3 reference
data: the C6 references with their coordination numbers, R0, covalent
radii and r2/r4, by atomic number. torch-dftd is distributed under the MIT
licence in `LICENSE` next to this file; its copyright notice and licence
terms apply to the file.

It ships with FastMACE so that the native D3 path needs no `[d3]` extra
(SPEC §4.1, §14). `tests/cpu/test_d3_kernel.py` checks the hash and, when
torch-dftd is installed, that the file equals torch-dftd's own.
