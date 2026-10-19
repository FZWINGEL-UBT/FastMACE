import pickle, sys
import numpy as np
a, b = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])


def flat(x):
    if isinstance(x, np.ndarray):
        return [x]
    if isinstance(x, list):
        return [y for v in x for y in flat(v)]
    return []


for key in sorted(set(a) | set(b)):
    fa = [y for call in a.get(key, []) for y in flat(call)]
    fb = [y for call in b.get(key, []) for y in flat(call)]
    worst = max((np.abs(x - y).max() for x, y in zip(fa, fb) if x.shape == y.shape and x.size), default=None)
    print(key, "calls", len(a.get(key, [])), len(b.get(key, [])), "arrays", len(fa), len(fb), "max diff", worst)
