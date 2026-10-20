"""2-T14 order plugin (outside the repository): --t14-seed N shuffles modules, then tests within each."""
import random


def pytest_addoption(parser):
    parser.addoption("--t14-seed", type=int, default=None, help="shuffle test order with this seed")


def pytest_report_header(config):
    seed = config.getoption("t14_seed")
    return [f"t14 order: {'default' if seed is None else f'seed {seed}'}"]


def pytest_collection_modifyitems(session, config, items):
    seed = config.getoption("t14_seed")
    if seed is None:
        return
    rng = random.Random(seed)
    modules = {}
    for item in items:
        modules.setdefault(item.nodeid.split("::")[0], []).append(item)
    names = sorted(modules)
    rng.shuffle(names)
    shuffled = []
    for name in names:
        group = modules[name]
        rng.shuffle(group)
        shuffled.extend(group)
    items[:] = shuffled
