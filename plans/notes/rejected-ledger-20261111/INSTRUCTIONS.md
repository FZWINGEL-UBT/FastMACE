# Catalogue of rejected optimisations in the ledger: instructions for one part

You gather facts about rejected optimisation ideas from a project ledger. A reviewer will later judge whether each
rejection was sound. **You only collect what the records say. Do not judge, guess or compute new numbers.**

## Files

- Ledger (read only): `D:\Margraf\FastMACE\docs\optimisation-ledger.md` (5,528 lines, some lines are very long).
- Your worklist: `D:\Margraf\FastMACE\plans\notes\rejected-ledger-20261111\worklists\part-NN.tsv` (NN is your part
  number). One row per ledger entry you must cover: `ledger_line`, `id`, `link`, `area`, `status_start`.
- Your output: `D:\Margraf\FastMACE\plans\notes\rejected-ledger-20261111\parts\part-NN.jsonl`. This is the only file
  you may create or change.

## Hard rules

- Everything except your output file is read only. Do not edit the ledger or any other file.
- Stay inside `D:\Margraf\FastMACE`. Do not open other checkouts or worktrees.
- No git commands that change anything (no add, commit, checkout, stash, pull). No tests, no Python model runs, no
  GPU, no WSL, no network.
- Cover every row of your worklist, in order. Do not add entries that are not on it.

## Steps

1. Read your worklist. Read ledger lines 1058–1085 once (the "Closed directions" table: direction, evidence,
   "Reopen only if").
2. For each worklist row:
   1. Read the index row at `ledger_line` (Read tool, `offset` = that line, `limit` 1). Its cells are
      `ID | Idea | Switch/flag | Engine | Status | Decision/date | Effect | Evidence`. If the line looks cut off, print
      it with Bash: `sed -n '<line>p' docs/optimisation-ledger.md`.
   2. Find the fuller record:
      - If `link` starts with `#`, search the ledger for `<a id="<anchor>">` (Grep with line numbers) and read from
        there to the next `<a id=` line (at most 60 lines). Its bullets are Switch, Engine, Date/round, What, Status,
        Measured, Why, Sources.
      - If `link` is a path such as `../results/x/README.md#section`, the path is relative to `docs/`. Check that the
        file exists. If it does, find the section (Grep for words of the idea or the anchor) and read at most 80
        lines of it.
      - If neither is found, use the index row alone and say so in `notes`.
   3. Check two things on disk:
      - `record_exists`: does the first linked report file (from the Evidence cell or the ID link) exist?
      - `code_found`: take the main identifier from the Switch/flag cell (for example `SPLINE_VECTOR` or
        `tools/fp16x3_gemm_bench.py`) and search `fastmace/` and `tools/` for it (Grep, files with matches). `true`,
        `false`, or `null` when the cell is `none` or empty.
   4. Write one JSON object (see below).
3. Append finished objects to your output file in batches of about 8 with Bash, exactly like this (the quoted
   delimiter keeps `$` and backticks literal):

   ```bash
   cat >> "/d/Margraf/FastMACE/plans/notes/rejected-ledger-20261111/parts/part-NN.jsonl" <<'JSONL_EOF'
   {"ledger_line": 123, "id": "example-id", ...}
   {"ledger_line": 124, "id": "other-id", ...}
   JSONL_EOF
   ```

4. At the end check the file: `wc -l` must equal the number of worklist rows, and
   `python -c "import json,sys; [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]; print('ok')" <file>`
   must print `ok`. Fix any line that fails.

## The JSON object

One object per line, valid JSON, double quotes, no line breaks inside a string. Use `null` when the records do not
say. Copy numbers and names as written.

| Key | Content |
| --- | --- |
| `ledger_line` | integer, from the worklist |
| `id` | from the worklist |
| `area` | from the worklist |
| `idea` | the Idea cell, at most 200 characters |
| `status` | the Status cell, at most 200 characters |
| `date` | the Decision/date cell |
| `kind` | one of `REJECTED`, `STOPPED`, `NOT_BUILT`, `DROPPED`, `CLOSED`, `PARTIAL` (adopted on some GPUs or models and rejected or closed on others), `OTHER` |
| `stage` | where it stopped: `bound` (decided from an estimate, a count or a profile on file; nothing was built), `lab` (a kernel-level lab, microbenchmark or probe was timed), `ab` (a built switch was timed in the real MD step or force call, side a against side b), `gate` (it failed an accuracy or force/stress check), `unknown` |
| `switch_or_tool` | the Switch/flag cell, at most 200 characters |
| `numbers` | the key result numbers, copied from the Effect cell or the Measured bullet, at most 250 characters |
| `tested_on` | list of GPUs with a measurement, as written (for example `"RTX 5060 Ti"`, `"RTX 4000 Ada"`, `"A100"`, `"H200"`, `"RTX A4000"`); `["CPU"]` for CPU-only counts; `[]` if none is named |
| `not_tested_on` | list of GPUs the text says were not measured; `[]` if none |
| `model_structure` | model and structures used, as written, at most 100 characters, or `null` |
| `baseline` | what it was compared against, as written (for example `"cuBLAS IEEE FP32"`, `"the default plan"`), or `null` |
| `reason` | the stated reason it was rejected, at most 35 words, in the record's own terms |
| `reopen` | a stated condition for trying again (from the entry, or the "Reopen only if" cell of a Closed-directions row that names this ID), or `null` |
| `entry_lines` | `[first, last]` ledger lines of the fuller record, or `null` |
| `record_paths` | list of linked report paths as written (at most 3) |
| `record_exists` | `true`, `false` or `null` |
| `code_found` | `true`, `false` or `null` |
| `flags` | list, only from the vocabulary below; `[]` if none applies |
| `notes` | anything odd or missing, at most 40 words, or `null` |

### Flags (use only when the text clearly supports it)

- `single_gpu`: measurements from exactly one GPU model are given.
- `no_build`: nothing was built or timed on a GPU; the decision rests on an estimate, a count or a bound.
- `stand_in`: the timing or accuracy came from a substitute for the real change: a probe, falsifier, ceiling,
  emulation, microbenchmark, a library call in place of the kernel, or a lab kernel that was never put in the engine.
- `tie`: every quoted speed ratio lies between 0.99 and 1.01.
- `mixed`: faster (ratio below 0.99) in at least one quoted case and slower (above 1.01) in another.
- `accuracy`: the stated reason is accuracy, rounding, a failed gate or energy drift, not speed.
- `resources`: the stated reason involves registers, spills, shared memory, blocks (CTAs) per SM, cache or memory
  traffic.
- `unexplained`: the text says the cause was not profiled, not traced or not explained, or only gives a "likely"
  cause.
- `retest_noted`: the text says it was measured again later, should be re-screened, or was "not measured" on some GPU.
- `superseded`: closed because another change replaced it or made it moot.

## Your final message

At most 150 words: how many entries you wrote, the IDs whose fuller record or report file you could not find, and
any row where the status and the numbers seem to contradict each other. No other commentary.
