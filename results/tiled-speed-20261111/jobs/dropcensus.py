"""Host census around a forced drop of the kept lists (idea 1 of plans/ideas-20261111-tiled.md, 11 November 2026).

Runs tools/bench/tiled_host_profile.py (its arguments follow) with HOSTPROF_CENSUS=1, and in addition:
- the first call's ``_auto_store`` drops the kept lists whatever the host has (the A100 397.5M record's path:
  lists filled by the search pass, then dropped when the state is allocated);
- just before the drop: the resident set and glibc's ``mallinfo2`` (no trim);
- just after it: a census (live host arrays, resident set before and after ``malloc_trim(0)``, ``mallinfo2`` before
  and after the trim). The resident set against the trimmed one is what the allocator kept of the dropped lists.
"""

import ctypes
import os
import sys
from pathlib import Path

os.environ.setdefault("HOSTPROF_CENSUS", "1")
tools = Path.cwd() / "tools" / "bench"
sys.path.insert(0, str(tools))
import tiled_host_profile as hp  # noqa: E402


class Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in
                ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks",
                 "keepcost")]


libc = ctypes.CDLL("libc.so.6")
libc.mallinfo2.restype = Mallinfo2


def mallinfo():
    m = libc.mallinfo2()
    return {name: int(getattr(m, name)) for name, _ in Mallinfo2._fields_}


original_census = hp.census


def census(tag):
    info = mallinfo()  # before the census's trim
    original_census(tag)
    hp.censuses[tag][-1]["mallinfo_before_trim"] = info
    hp.censuses[tag][-1]["mallinfo_after_trim"] = mallinfo()
    print(f"census {tag}: rss {hp.censuses[tag][-1]['rss'] / 2**30:.2f} GiB, trimmed "
          f"{hp.censuses[tag][-1]['rss_trimmed'] / 2**30:.2f}, arrays {hp.censuses[tag][-1]['arrays'] / 2**30:.2f}, "
          f"malloc free {info['fordblks'] / 2**30:.2f} in use {info['uordblks'] / 2**30:.2f} "
          f"mmapped {info['hblkhd'] / 2**30:.2f}", flush=True)


hp.census = census
from fastmace.model import tiled  # noqa: E402

store = tiled.TiledForceField._auto_store
forced = [False]


def auto_store(self, state, n, width):
    if not forced[0] and self._kept is not None and not getattr(self, "_reusing", False):
        forced[0] = True
        lists = tiled._kept_bytes(self._kept)
        print(f"forced drop: kept lists {lists / 2**30:.2f} GiB", flush=True)
        info = mallinfo()
        print(f"before drop: rss {hp.rss() / 2**30:.2f} GiB, malloc free {info['fordblks'] / 2**30:.2f} in use "
              f"{info['uordblks'] / 2**30:.2f} mmapped {info['hblkhd'] / 2**30:.2f}", flush=True)
        hp.censuses.setdefault("before-drop", []).append(dict(rss=hp.rss(), mallinfo=info, lists=lists))
        self._drop_kept("forced (drop census)")
        census("after-drop")
        return self.dtype, f"the call's dtype after a forced drop of the kept lists ({lists / 2**30:.1f} GiB)"
    return store(self, state, n, width)


tiled.TiledForceField._auto_store = auto_store
hp.main(sys.argv[1:])
