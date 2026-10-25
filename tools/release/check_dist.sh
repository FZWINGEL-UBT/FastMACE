#!/bin/bash
# 6-T13 item 2: contents of the built wheel and sdist.
cd ~/tmp/6-T13
py=~/environments/fastmace_cpu/bin/python
$py - <<'EOF' | tee contents.txt
import tarfile, zipfile, glob
whl = glob.glob("dist/*.whl")[0]; sd = glob.glob("dist/*.tar.gz")[0]
names = zipfile.ZipFile(whl).namelist()
snames = [n.split("/", 1)[1] for n in tarfile.open(sd).getnames() if "/" in n]
need = ["fastmace/kernels/oeq_streaming/LICENSE", "fastmace/kernels/oeq_streaming/NOTICE.md",
        "fastmace/longrange/data/dftd3_params.npz", "fastmace/longrange/data/LICENSE",
        "fastmace/longrange/data/NOTICE.md"]
ok = True
for n in need:
    w, s = n in names, n in snames
    ok &= w and s
    print(f"{'ok ' if w and s else 'MISSING'} {n} wheel={w} sdist={s}")
for pat in ("fastmace/kernels/oeq_streaming/", "fastmace/protocols/data/"):
    w = sorted(x for x in names if x.startswith(pat) and not x.endswith(".py"))
    s = sorted(x for x in snames if x.startswith(pat) and not x.endswith(".py"))
    print(pat, "wheel:", [x.rsplit("/", 1)[1] for x in w])
    ok &= bool(w) and w == s and (pat != "fastmace/kernels/oeq_streaming/" or any(x.endswith(".cuh") for x in w))
lic = [x for x in names if ".dist-info/licenses/" in x]
print("dist-info licences:", lic)
ok &= any(x.endswith("licenses/LICENSE") for x in lic) and any(x.endswith("LGPL-2.1-or-later.txt") for x in lic)
print("sdist NOTICE.md:", "NOTICE.md" in snames, "LICENSES:", [x for x in snames if x.startswith("LICENSES/")])
ok &= "NOTICE.md" in snames
meta = zipfile.ZipFile(whl).read([x for x in names if x.endswith("METADATA")][0]).decode()
for line in meta.splitlines():
    if line.split(":")[0] in ("Name", "Version", "License-Expression", "License-File", "Requires-Python") or "oeq" in line:
        print(" ", line)
ok &= "License-Expression: MIT AND LGPL-2.1-or-later" in meta and "Version: 1.0.0" in meta
print("ITEM2", "PASS" if ok else "FAIL")
EOF
