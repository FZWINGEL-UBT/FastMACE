"""Merge and check the per-part records of the rejected-optimisation catalogue (11 November 2026).

Row-level fields (idea, switch, status, date, effect, links) are parsed here from the ledger's index rows and are
authoritative. The agents' records (parts/part-NN.jsonl) add what needs reading: kind, stage, GPUs, baseline, reason,
reopening condition, flags. Their record_exists / code_found answers are recomputed here and compared.

    python plans/notes/rejected-ledger-20261111/merge.py            (from the repository root; CPU only)

Writes catalogue.jsonl, catalogue.md, summary.md and check.txt beside this file.
"""
import io
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEDGER = ROOT / "docs" / "optimisation-ledger.md"
KEYS = ["ledger_line", "id", "area", "idea", "status", "date", "kind", "stage", "switch_or_tool", "numbers",
        "tested_on", "not_tested_on", "model_structure", "baseline", "reason", "reopen", "entry_lines",
        "record_paths", "record_exists", "code_found", "flags", "notes"]
KINDS = {"REJECTED", "STOPPED", "NOT_BUILT", "DROPPED", "CLOSED", "PARTIAL", "OTHER"}
STAGES = {"bound", "lab", "ab", "gate", "unknown"}
FLAGS = {"single_gpu", "no_build", "stand_in", "tie", "mixed", "accuracy", "resources", "unexplained",
         "retest_noted", "superseded"}
GPU_NAMES = (("H200", r"H200"), ("A100", r"A100"), ("RTX 4000 Ada", r"\bAda\b|sm_89|murphy|leto|leeloo"),
             ("RTX 5060 Ti", r"5060|sm_120|\blocal\b"), ("RTX A4000", r"A4000|coraline|sm_86"), ("L4", r"\bL4\b"))


def cells(line):
    """The eight cells of an index row (pipes inside backticks stay in their cell)."""
    parts, cur, tick = [], "", False
    for ch in line:
        if ch == "`":
            tick = not tick
        if ch == "|" and not tick:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    parts = parts[1:-1] if parts and parts[0] == "" else parts
    if len(parts) > 8:  # a stray pipe: keep the first six cells, the last one, and join the rest as the effect
        parts = parts[:6] + [" | ".join(parts[6:-1]), parts[-1]]
    return parts + [""] * (8 - len(parts))


def link_paths(text):
    return [m for m in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text) if not m.startswith("http")]


def identifiers(switch):
    """Names from the Switch/flag cell worth searching the code for."""
    out = []
    for token in re.findall(r"`([^`]+)`", switch):
        for piece in re.split(r"[,;\s]+", token):
            piece = piece.strip("()[]\"'")
            name = piece.split(".")[-1] if not piece.endswith(".py") else piece.split("/")[-1]
            name = re.sub(r"=.*", "", name)
            if len(name) >= 5 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
                out.append(name)
    return out


def grep_found(name):
    r = subprocess.run(["git", "-C", str(ROOT), "grep", "-l", "-F", name, "--", "fastmace", "tools"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def ratios(text):
    """Bare numbers that look like speed ratios (0.5 to 1.99, not followed by a unit or a per cent sign)."""
    out = []
    for m in re.finditer(r"(?<![\d.,\-–e])([01]\.\d{2,4})(?![\d.])(\s*(?:%|µs|us|ms|s\b|GB|GiB|MB|kB|eV|Å|A\b|x\b|×|e-))?", text):
        v = float(m.group(1))
        if m.group(2) is None and 0.5 <= v < 2.0:
            out.append(v)
    return out


def main():
    ledger = io.open(LEDGER, encoding="utf-8").read().split("\n")
    work = []
    for f in sorted((HERE / "worklists").glob("part-*.tsv")):
        part = int(f.stem.split("-")[1])
        for row in io.open(f, encoding="utf-8").read().split("\n")[1:]:
            if row.strip():
                line, ident, link, area, _ = row.split("\t")
                work.append((part, int(line), ident, link, area))
    records, problems = {}, []
    for f in sorted((HERE / "parts").glob("part-*.jsonl"), key=lambda p: (p.stem.endswith("-fix"), p.name)):
        part = int(f.stem.split("-")[1])
        for n, raw in enumerate(io.open(f, encoding="utf-8").read().split("\n"), 1):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                problems.append(f"part {part:02d} line {n}: invalid JSON ({exc})")
                continue
            key = (rec.get("ledger_line"), rec.get("id"))
            if key in records and not f.stem.endswith("-fix"):  # part-NN-fix.jsonl replaces rows of part-NN
                problems.append(f"part {part:02d}: duplicate {key}")
            rec["_part"] = part
            records[key] = rec
    wanted = {(w[1], w[2]) for w in work}
    for key in sorted(set(records) - wanted, key=str):
        problems.append(f"extra record {key} (part {records[key]['_part']:02d})")
    merged, code_cache = [], {}
    for part, line, ident, link, area in work:
        c = cells(ledger[line - 1])
        date = re.search(r"\d{4}-\d{2}-\d{2}", c[5])
        paths = link_paths(c[7]) or link_paths(c[0])
        exists = None
        for p in paths[:1]:
            exists = (ROOT / "docs" / p).resolve().exists()
        names = identifiers(c[2])
        found = None
        if names:
            for nm in names[:3]:
                if nm not in code_cache:
                    code_cache[nm] = grep_found(nm)
            found = any(code_cache[nm] for nm in names[:3])
        text = " ".join(c[4:7])
        row = dict(ledger_line=line, id=ident, area=area, part=part, idea=c[1], switch_or_tool=c[2], engine=c[3],
                   status=c[4], decision_date=c[5], date=date.group(0) if date else None, effect=c[6], evidence=c[7],
                   record_paths=paths[:3], record_exists=exists, code_identifiers=names[:3], code_found=found,
                   gpus_in_row=[g for g, pat in GPU_NAMES if re.search(pat, text)], ratios_in_effect=ratios(c[6]))
        rec = records.get((line, ident))
        if rec is None:
            problems.append(f"part {part:02d}: missing {ident} (ledger line {line})")
            row["agent"] = None
        else:
            miss = [k for k in KEYS if k not in rec]
            if miss:
                problems.append(f"part {part:02d} {ident}: keys missing {miss}")
            if rec.get("kind") not in KINDS:
                problems.append(f"part {part:02d} {ident}: kind {rec.get('kind')!r}")
            if rec.get("stage") not in STAGES:
                problems.append(f"part {part:02d} {ident}: stage {rec.get('stage')!r}")
            bad = [x for x in (rec.get("flags") or []) if x not in FLAGS]
            if bad:
                problems.append(f"part {part:02d} {ident}: unknown flags {bad}")
            if rec.get("status") and not c[4].replace("`", "").startswith(str(rec["status"]).replace("`", "")[:25]):
                problems.append(f"part {part:02d} {ident}: status differs from the ledger ({str(rec['status'])[:40]!r})")
            if rec.get("record_exists") is not None and exists is not None and rec["record_exists"] != exists:
                problems.append(f"part {part:02d} {ident}: record_exists {rec['record_exists']} but the check says {exists}")
            if rec.get("code_found") is not None and found is not None and rec["code_found"] != found:
                problems.append(f"part {part:02d} {ident}: code_found {rec['code_found']} but the check says {found}")
            row["agent"] = {k: rec.get(k) for k in ("kind", "stage", "numbers", "tested_on", "not_tested_on",
                                                    "model_structure", "baseline", "reason", "reopen", "entry_lines",
                                                    "flags", "notes")}
        merged.append(row)
    with io.open(HERE / "catalogue.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    done = [r for r in merged if r["agent"]]
    with io.open(HERE / "check.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"worklist rows {len(work)}, records {len(done)}, problems {len(problems)}\n")
        f.write("\n".join(problems) + ("\n" if problems else ""))

    def short(text, n):
        text = re.sub(r"\s+", " ", str(text or "")).replace("|", "/")
        return text if len(text) <= n else text[: n - 1] + "…"

    md = ["# Rejected, stopped and not-built optimisations in the ledger (catalogue, 11 November 2026)", "",
          f"{len(merged)} index rows of `docs/optimisation-ledger.md` whose status is rejected, stopped, dropped, cut, "
          "closed or not built (or adopted on some GPUs and rejected on others). Idea, status and date are parsed "
          "from the ledger; stage, GPUs, reason and flags were read from the entries by Haiku subagents and are "
          "**not verified** beyond the checks in `check.txt`. Flags are defined in `INSTRUCTIONS.md`.", ""]
    by_area = defaultdict(list)
    for r in merged:
        by_area[r["area"]].append(r)
    for area, rows in by_area.items():
        md += [f"## {area} ({len(rows)})", "", "| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |",
               "| ---: | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            a = r["agent"] or {}
            md.append("| {} | `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                r["ledger_line"], r["id"], short(r["idea"], 110), short(r["status"], 60), r["date"] or "–",
                a.get("stage") or "–", short(", ".join(a.get("tested_on") or r["gpus_in_row"]), 50),
                short(a.get("reason"), 170), short(", ".join(a.get("flags") or []), 60)))
        md.append("")
    io.open(HERE / "catalogue.md", "w", encoding="utf-8", newline="\n").write("\n".join(md))

    kinds = Counter((r["agent"] or {}).get("kind") for r in done)
    stages = Counter((r["agent"] or {}).get("stage") for r in done)
    flags = Counter(x for r in done for x in (r["agent"].get("flags") or []))
    months = Counter((r["date"] or "none")[:7] for r in merged)
    sm = ["# Summary of the rejected-optimisation catalogue (11 November 2026)", "",
          f"Rows {len(merged)}, with an agent record {len(done)}, check problems {len(problems)} (see `check.txt`).", "",
          "Source: every index row of `docs/optimisation-ledger.md` (main, 11 November 2026) whose status is "
          "rejected, stopped, dropped, cut, closed or not built, or adopted on some GPUs and rejected on others. "
          "Eleven Haiku subagents read each row, its fuller entry and its linked report (brief: `INSTRUCTIONS.md`) "
          "and recorded what the records say; they did not judge the rejections. `merge.py` parses idea, status, "
          "date and links from the ledger itself and compares the agents' file and code checks with its own.", "",
          "Reliability: kind, stage, GPUs, reason and flags are the agents' reading. One random row per part was "
          "compared with the ledger by hand: status and reason matched in all 11, the stage label was debatable in "
          "2. Treat the flags as a filter for what to read, not as findings. Inconsistencies the agents reported in "
          "the ledger are in `agent-notes.md`. The full table is `catalogue.md`; the data are `catalogue.jsonl`.", "",
          "| Kind | Rows |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in kinds.most_common()]
    sm += ["", "| Stage where it stopped | Rows |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in stages.most_common()]
    sm += ["", "| Flag | Rows |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in flags.most_common()]
    sm += ["", "| Month of the decision | Rows |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in sorted(months.items())]
    areas = Counter(r["area"] for r in merged)
    sm += ["", "| Area | Rows |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in areas.most_common()]
    named = Counter(g for r in done for g in set(r["gpus_in_row"]))
    sm += ["", "GPUs named in the index row's status or effect (a row can name several; "
           f"{sum(1 for r in done if not r['gpus_in_row'])} rows name none):", "", "| GPU | Rows |", "| --- | ---: |"]
    sm += [f"| {k} | {v} |" for k, v in named.most_common()]

    def ids(rows):
        count = Counter(r["id"] for r in rows)
        return ", ".join(f"`{k}`" + (f" (×{v})" if v > 1 else "") for k, v in count.items()) or "none"

    timed = [r for r in done if r["agent"].get("stage") in ("lab", "ab", "gate")]
    sm += ["", "## Rows by flag, among those that were timed or gated (stage lab, ab or gate)", "",
           f"{len(timed)} of the {len(done)} rows. The flags come from the agents' reading and are not verified row by row.", ""]
    for flag in ("unexplained", "accuracy", "resources", "tie", "mixed", "retest_noted", "single_gpu", "stand_in"):
        rows = [r for r in timed if flag in (r["agent"].get("flags") or [])]
        sm += [f"### `{flag}` ({len(rows)})", "", ids(rows), ""]
    old = [r for r in timed if r["date"] and r["date"] < "2026-11-05" and r["code_found"]
           and r["agent"].get("kind") in ("REJECTED", "CLOSED", "STOPPED", "PARTIAL")]
    sm += [f"## Timed before 5 November 2026 and the switch or tool still exists in the code ({len(old)})", "",
           "These were measured against older defaults (the BUTTERFLY re-screen of 9 November flipped one such "
           "rejection).", "", ids(old), ""]
    io.open(HERE / "summary.md", "w", encoding="utf-8", newline="\n").write("\n".join(sm) + "\n")
    print(f"rows {len(merged)}, records {len(done)}, problems {len(problems)}")
    for p in problems[:40]:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
