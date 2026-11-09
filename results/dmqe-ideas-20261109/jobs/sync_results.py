"""Install this lab's new files; merge its two index additions without replacing NAS edits."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

allowed = Path('/mnt/homes/zwingel').resolve()
source, target = (Path(value).resolve() for value in sys.argv[1:3])
if not source.is_relative_to(allowed) or not target.is_relative_to(allowed):
    raise SystemExit('Both paths must remain under /mnt/homes/zwingel')

ledger = Path('docs/optimisation-ledger.md')
index = Path('docs/validation/speed/README.md')
marker = '## dm_qe transfer labs (9 November 2026)'
row_prefix = '| [dmqe-transfer-labs-20261109]'
updates = []
for path in source.rglob('*'):
    if not path.is_file():
        continue
    relative = path.relative_to(source)
    destination = target / relative
    if not destination.resolve().is_relative_to(allowed):
        raise SystemExit(f'Unsafe destination: {destination}')
    if relative == ledger and destination.exists():
        addition = marker + path.read_text().split(marker, 1)[1]
        previous = destination.read_text()
        if marker in previous:
            if previous.split(marker, 1)[1].strip() != addition.split(marker, 1)[1].strip():
                raise SystemExit('Conflicting NAS lab ledger section; preserving it')
            continue
        updates.append((destination, previous.rstrip('\n') + '\n\n' + addition))
    elif relative == index and destination.exists():
        addition = next(line for line in path.read_text().splitlines() if line.startswith(row_prefix))
        previous = destination.read_text()
        if row_prefix in previous:
            if addition not in previous:
                raise SystemExit('Conflicting NAS lab index row; preserving it')
            continue
        anchor = '| --- | --- | --- | --- |\n'
        if anchor not in previous:
            raise SystemExit('NAS speed index changed structure; preserving it')
        updates.append((destination, previous.replace(anchor, anchor + addition + '\n', 1)))
    else:
        if destination.exists() and destination.read_bytes() != path.read_bytes():
            raise SystemExit(f'Conflicting NAS file: {relative}; preserving it')
        updates.append((destination, path))

# Preflight every destination before writing anything.
for destination, content in updates:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, Path):
        shutil.copyfile(content, destination)
    else:
        destination.write_text(content)
print(f'Synced {len(updates)} lab files/index additions; canonical source marker untouched')
