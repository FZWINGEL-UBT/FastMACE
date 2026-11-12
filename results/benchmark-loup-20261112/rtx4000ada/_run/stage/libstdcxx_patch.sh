# deviation 4: Symmetrix-XL/LAMMPS objects built by the toolchain's gcc 13 need GLIBCXX_3.4.31; the hosts' system
# libstdc++ is older (Ubuntu 22.04: gcc 12; AlmaLinux 9: gcc 11). sxl_env (Symmetrix-XL and LAMMPS only) puts the
# toolchain's own lib/ (conda-forge libstdc++ 16.2) first on LD_LIBRARY_PATH. FastMACE / MACE+cuEq runs are untouched.
f=${1:?scripts/lib.sh}
grep -q 'LD_LIBRARY_PATH=\$TC/lib' "$f" || sed -i 's|^  export PATH=\$TC/bin:\$TOOLS:\$PATH$|&\n  export LD_LIBRARY_PATH=$TC/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}|' "$f"
grep -n 'LD_LIBRARY_PATH' "$f"
