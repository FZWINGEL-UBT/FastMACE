"""1b-T19: ``fastmace models``, the versioned label table (SPEC §7.5, §7.6)."""

from fastmace import cli
from fastmace.model import labels


def _rows(out):
    return {line.split()[0]: line for line in out.splitlines() if line.split() and line.split()[0] in labels.LABELS}


def test_models_lists_all_19_labels(capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    rows = _rows(out)
    assert len(labels.LABELS) == 19
    assert set(rows) == set(labels.LABELS)


def test_each_row_has_head_file_hash_and_licence(capsys):
    cli.main(["models"])
    rows = _rows(capsys.readouterr().out)
    for name, entry in labels.LABELS.items():
        row = rows[name]
        file_name = entry.checkpoint.url.rsplit("/", 1)[-1]
        assert file_name in row, name
        assert entry.checkpoint.sha256 in row, name
        assert entry.licence in row.split(), name
        head = entry.head or (entry.checkpoint.heads[0] if entry.checkpoint.heads else "-")
        assert head in row.split(), name


def test_models_shows_licence_texts_and_aliases(capsys):
    cli.main(["models"])
    out = capsys.readouterr().out
    for text in labels.LICENCE_TEXT.values():
        assert text in out
    for alias, checkpoint in labels.ALIASES.items():
        assert alias in out and checkpoint in out


def test_models_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FASTMACE_CACHE", str(tmp_path / "cache"))
    cli.main(["models"])
    assert list(tmp_path.iterdir()) == []
