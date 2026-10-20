"""2-T19: profile Registry.resolve on the pristine MH-1 model (CPU is enough: resolution reads sources and modules).

    python profile_resolve.py CHECKPOINT [--label mh1-r2scan] [--repeat 2]
"""

import argparse
import cProfile
import io
import pstats
import time

import fastmace  # noqa: F401  (before torch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--label", default="mh1-r2scan")
    p.add_argument("--repeat", type=int, default=2)
    args = p.parse_args()
    from fastmace.model import loading
    from fastmace.rewrites import registry as reg

    loaded = loading.load_pristine(args.label, checkpoint=args.checkpoint, sha256=loading.sha256_of(args.checkpoint))
    model = loaded.model
    for attempt in range(args.repeat):
        started = time.perf_counter()
        registry = reg.default_registry()
        built = time.perf_counter() - started
        profile = cProfile.Profile()
        started = time.perf_counter()
        profile.enable()
        registry.resolve(model)
        profile.disable()
        print(f"attempt {attempt}: default_registry {built:.3f} s, resolve {time.perf_counter() - started:.3f} s")
        out = io.StringIO()
        pstats.Stats(profile, stream=out).sort_stats("cumulative").print_stats(25)
        print(out.getvalue())


if __name__ == "__main__":
    main()
