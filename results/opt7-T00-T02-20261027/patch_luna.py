"""Add environment overrides to the scratch venv's LunaEquivariance (timing sweep only; never the source tree).

LUNA_SEG: edges per source segment (default 32); LUNA_RW: resident warps (launch bounds); LUNA_VEC: vector width
(1 or 2); LUNA_THREADS: thread limit per block; LUNA_DEBUG=1 prints the chosen parameters.
"""
from pathlib import Path

root = Path.home() / "environments/lunatp_screen/lib/python3.10/site-packages/LunaEquivariance/lunatp/uvu_conv/kernels"
utils = root / "utils.py"
s = utils.read_text()
if "LUNA_SEG" not in s:
    s = s.replace("_SOURCE_SEGMENT_EDGES = 32", "import os as _os\n_SOURCE_SEGMENT_EDGES = int(_os.environ.get('LUNA_SEG', 32))")
    utils.write_text(s)
backward = root / "backward.py"
s = backward.read_text()
if "LUNA_RW" not in s:
    s = s.replace("""        padded_channels = ((channels // vector + 31) // 32) * 32
        limit = min(128 if wide_block else 32, padded_channels)""",
                  """        import os as _os
        if _os.environ.get("LUNA_VEC"):
            vector = int(_os.environ["LUNA_VEC"])
        if _os.environ.get("LUNA_RW"):
            resident_warps = int(_os.environ["LUNA_RW"])
        padded_channels = ((channels // vector + 31) // 32) * 32
        limit = min(128 if wide_block else 32, padded_channels)
        if _os.environ.get("LUNA_THREADS"):
            limit = min(int(_os.environ["LUNA_THREADS"]), padded_channels)""")
    s = s.replace("""        kernel = f"lunatp_uvuconv_bwd_source_fp64<{group_id}, {tiles}, {vector}, {min_blocks}, {threads}, INDEX>\"""",
                  """        kernel = f"lunatp_uvuconv_bwd_source_fp64<{group_id}, {tiles}, {vector}, {min_blocks}, {threads}, INDEX>"
        if _os.environ.get("LUNA_DEBUG"):
            import sys as _sys
            print(f"LUNA bwd group {group_id}: fuse={fuse} wide_block={wide_block} vector={vector} "
                  f"resident_warps={resident_warps} threads={threads} tiles={tiles} min_blocks={min_blocks} "
                  f"live_words={live_words} segmented={segmented}", file=_sys.stderr)""")
    backward.write_text(s)
print("LUNA_RW" in backward.read_text(), "LUNA_SEG" in utils.read_text())
