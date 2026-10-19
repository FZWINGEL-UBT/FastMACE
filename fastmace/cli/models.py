"""``fastmace models``: the versioned label table (SPEC §7.5, §7.6).

One row per label: head, licence, file name and SHA-256 of the released
checkpoint, then the licence texts and the aliases that take ``head=``.
Reads the table only; looks at no file and downloads nothing.
"""

from fastmace.model import labels

HELP = "list the model labels: heads, file names, SHA-256 and licences"


def add_arguments(parser):
    pass


def file_name(checkpoint):
    """The released file name of a checkpoint (the last part of its official URL)."""
    return checkpoint.url.rsplit("/", 1)[-1]


def _head(entry):
    if entry.head is not None:
        return entry.head
    return entry.checkpoint.heads[0] if entry.checkpoint.heads else "-"


def rows():
    """(label, head, licence, file name, SHA-256) for every label, in table order."""
    return [(name, _head(entry), entry.licence, file_name(entry.checkpoint), entry.checkpoint.sha256)
            for name, entry in labels.LABELS.items()]


def run(args):
    table = rows()
    header = ("label", "head", "licence", "file", "sha256")
    widths = [max(len(row[i]) for row in [header, *table]) for i in range(len(header))]
    for row in [header, *table]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    print()
    print("Licences:")
    for key, text in labels.LICENCE_TEXT.items():
        print(f"  {key}: {text}")
    print()
    print("Aliases (pass head=):")
    for alias, checkpoint in labels.ALIASES.items():
        heads = ", ".join(labels.CHECKPOINTS[checkpoint].heads) or "-"
        print(f"  {alias} -> checkpoint {checkpoint} (heads: {heads})")
    print()
    print("fm.load(label) looks for the file in $FASTMACE_MODELS, then <cache root>/models; "
          "download=True fetches it from the official URL and checks the SHA-256 before unpickling.")
    return 0
