# Resource framework: faster results from the same resources

Written 7 November 2026 from the session history of 7 October – 7 November 2026. Times are CET unless marked UTC.
This is a proposal. Nothing in it is adopted until the owner or the conductor says so.

## Summary

- **Most paid GPU time is not measurement.** On Modal, only 17 % (H200) to 26 % (A100) of a job's wall time is
  timed MD steps. The rest is compiling extensions and setup (35–39 %) and container start, image and loading
  (38–44 %).
- **The Modal budget runs out early and blocks itself.** About $100 of November's $120 (4 profiles × $30) was gone by
  6–7 November. The budget guard reserves each job's worst case, and those reservations add up to 5.3× what the
  jobs really cost. Jobs are refused while real money is still left.
- **Sessions wait in the foreground.** 1,807 tool calls blocked for more than a minute each, about 120 hours in
  total. That is about 38 % of all active session time.
- **The same failures repeat.** WSL ran out of memory five times (16, 21, 22 and 24 October, then 6 November). Shell
  quoting errors appear in 86 transcripts. Each lesson went into a memory note, but not into a tool, so the
  failures came back.
- **Late ideas are small and costly.** From 5 to 7 November, 19 per-GPU switches were adopted for about $150 of Modal
  time, roughly $8 each. Most gave 0.5–1.5 % of the step. Many more ideas were measured and then dropped.
- **Free compute sits unused.** The Alex A100s (free, but queued) and Colab (200 compute units a month, no cap from
  the owner) did little of the A100 work. Modal A100 jobs cost $107, nearly half of all Modal spend.

The framework below gives each resource one job, packs A/Bs into fewer containers, kills ideas before they reach a
paid GPU, and replaces per-task sessions with a few standing lanes. The expected result is about 2–3× more
adoption decisions per dollar, and a fixed two-hour cycle from idea to default.

## 1. How the work ran

| Period | What happened | Shape of the work |
| --- | --- | --- |
| 7–16 Oct | Research engine in ResearchModule; speed rounds 0–4 | One or two long sessions; local GPU and leeloo |
| 17–25 Oct | FastMACE build, phases 0–5 and opt rounds 1–4 | One conductor (`7d08cb4d`) and about 120 short worker sessions in worktrees |
| 26 Oct – 1 Nov | Rounds 5–10, capacity, tiled backend | A few long workers (`opt10-gate-epilogue`, `tiled-backend`); workstations back after the outage |
| 2–4 Nov | Conductor hit the account's weekly limit; another account's conductor (`0ef98cc0`) took over | Work ran inline in one session; tiled branches waited 28–118 h to merge |
| 4–6 Nov | A100 and H200 OMAT-0 rounds; h200-ideas session (14 revisions) | Modal on four profiles; spend peaked at about $74 on 6 Nov |
| 6–7 Nov | New conductor (`2aeb1c92`); ideas-queue batches; OFF/POLAR on Alex | Batches of 17–19 Modal jobs; Alex for OFF/POLAR |

Totals over the month:

| Measure | Value | Source |
| --- | --- | --- |
| Top-level sessions | 171 (204 `start_session` calls, 53 subagents) | transcripts in `~/.claude/projects/D--Margraf*`, `G--Margraf-ResearchModule*` |
| Active session time | 318 h (gaps over 5 min not counted); sessions stayed open for 2,622 h in total | same |
| Bash calls | 36,857; about 5,600 of them were polls (`tail`, `squeue`, `pgrep`, `nvidia-smi`, `ls -t`) | same |
| Foreground calls over 60 s | 1,807 calls, about 120 h | same |
| Explicit `sleep` in commands | 872 calls, about 21 h | same |
| Modal jobs | 389 in the ledger (from 22 Oct), $228 at GPU list price | `~/.fastmace-modal/modal-ledger.csv` |
| Modal by GPU | A100 $107 (213 jobs), H200 $77 (114), L4 $29, others $15 | same |
| Merges into main | 373; median 12 min from branch tip to merge, p90 1.2 h | `git log --first-parent main` |

## 2. Findings: where the time and money went

Ranked by size.

### F1. Container overhead dominates Modal spend

For the 145 Modal jobs that wrote an A/B result file:

| GPU | Wall | Setup and compile (`setup_s`) | Timed MD steps | Everything else |
| --- | --- | --- | --- | --- |
| A100 | 22.1 h | 7.8 h (35 %) | 5.8 h (26 %) | 8.5 h (38 %) |
| H200 | 6.6 h | 2.6 h (39 %) | 1.1 h (17 %) | 2.9 h (44 %) |

- Example: `bin16-a100-a` ran 497 s. The first side's setup took 138 s, the second side's 16 s, and the timed steps
  about 130 s. The first side pays for a cold extension build. "Precompiled extension shared object not found"
  appears in 70 transcripts.
- Most jobs carry one switch: one A/B in both orders, plus its own compile, model load and container start.
- The median job is short (H200 357 s, A100 517 s), so fixed costs weigh heavily.

### F2. The budget guard blocks work while money is left

- The guard reserves timeout + 300 s at list price for every job in flight. Over all jobs the reservations sum to
  $1,197 against $228 of real cost (5.3×). The median job used 15 % of its timeout.
- 16 of 155 launch logs show budget refusals (32 refusals), plus 29 hits of the app-create rate limit.
- After the WSL crashes on 6 November, stale reservations held $18.70 (third and fourth profiles) until removed by
  hand. One gate waited about an hour on an almost empty profile before it was moved to `fourth`.
- Modal was busy for about 48 h in total, and for 35 h of those only one job was running. Four or more jobs ran
  together for only 3.3 h, although four profiles were available.
- November's credit went fast: estimated $49 on 5 Nov, $74 on 6 Nov and $30 on 7 Nov. On 6–7 Nov the profiles had
  about $2 (second), $3 (fzwingel), $6.50 (third) and $9 (fourth) left. Nothing is held back for the rest of the month.

### F3. Failed and repeated runs

| Cause | Count | Cost | Example |
| --- | --- | --- | --- |
| A100 PCIe landing treated as wrong GPU (exit 3) | 43 jobs, plus retries | $4.30 and the retry waits | Retried until 6 Nov, when the owner said PCIe is fine |
| Timeouts (exit 124) | 6 jobs | $6.80 | – |
| Job errors (exit 1, 127) | 20 jobs | $5.30 | – |
| WSL out of memory | 5 incidents | 0.5–3 h each, all running jobs lost | 6 Nov: 19 Modal clients at about 9 GB each; batch 5 was relaunched four times between 14:46 and 15:30 |
| WSL powering off | 1 incident | one batch lost | WSL stops the VM 17 s after the last `wsl` call, and that killed detached jobs |
| Shell quoting | `unexpected EOF while looking for matching` in 86 transcripts, about 260 results | about 1 min each, plus the risk | 4 Nov: an empty `$d` copied the WSL root into a results folder |
| Launcher edited while in use | several | lost jobs | 6 Nov 13:46 UTC: "I dropped the commit argument when I rewrote the `run` helper" |

In all, 71 Modal jobs (18 %) ended with an error, costing $20 (9 % of spend). The relaunch time and attention cost
more than the money.

### F4. Waiting in the foreground

- 1,807 tool calls blocked for more than 60 s, about 120 h in total. Most were polls, `sleep`, and suites or jobs run
  in the foreground.
- The owner had to stop this twice: blocking `ctx_execute` polling on 4 Nov, and four parallel suites that made the
  PC unusable on 22 Oct.
- A blocked session cannot review, plan or read results.

### F5. Alex: free but slow to start

- Group GPU cap: jobs wait in `AssocGrpGRES` at about 7 concurrent A100s for the group `v130cb`.
- On 6 Nov all 16 screens were pending by 14:30. Slurm estimated a 22:30 start for later jobs, and one profile
  started at 20:40.
- Many short ssh calls from Windows (no ControlMaster) got this PC throttled from about 17:40. Two 2-minute pollers
  were running.
- Two sessions did the same OFF/POLAR-on-Alex task: `local_9c6a796a` from the old conductor and `local_2ae93bf5` from
  the new one, about an hour apart.

### F6. Sessions and the conductor

- In the build phase (17–24 Oct), about 120 worker sessions were active for 0.7 h on average but stayed open for
  4–68 h in "waiting" or "prestart". Each new session reads AGENTS.md, SPEC.md and the plan again.
- 30 % of the first conductor's 7,702 tool calls were coordination: messages, plan updates, `list_events` and session
  starts.
- There were four conductors in five days (2, 4 and 6 Nov), caused by the account's weekly limit and a broken shell.
  On 6 Nov two conductors overlapped from 13:11 to 16:12.
- Merging is not the bottleneck: the median is 12 min. The exception is 4 Nov, when branches waited 28–118 h after
  the account stop.
- On 22 Oct, one local GPU served four workers. One worker's jobs starved four short test runs for 37 min.

### F7. Idea hit rate and size

- Screen table, 20–27 Oct: about 17 of 47 screens were adopted (36 %). Gains were often 5–15 %.
- 5–7 Nov: the h200-ideas plan grew to 2,622 lines and about 100 idea headings over 14 revisions. 19 switches were
  adopted (5 on 5 Nov, 14 on 6 Nov), mostly 0.5–1.5 % of the step each.
- On 6 Nov the owner stopped further idea rounds, because all remaining ideas together promised 1.5–3.5 %.
- Many late ideas were bounded or labbed first. That was right, but some still reached a paid A/B with a bound close
  to the noise floor (about 0.3–0.5 %).
- From idea to default took about a day on 6 Nov: queue launched in the afternoon, adopted the same evening. Waiting
  for A/Bs, retries and relaunches set that time, not merging.

### F8. Workstations and local GPU

- leto runs the full CPU suite (5,181 tests) in 13 min. The local PC needs over 4 h.
- Workstations are capped at 4 GPUs and need an idle check before every job. They were used heavily: murphy about
  2,800 command mentions, leto 2,300, leeloo 600.
- Colab was used in only 3 sessions (30 commands), despite the free allowance.

## 3. The framework

### 3.1 One job per resource

| Resource | Use it for | Do not use it for |
| --- | --- | --- |
| **Modal H200** (4 profiles) | Final H200 A/Bs (both orders) and H200 gates, packed | Labs, profiling, retries, A100 work |
| **Modal A100** | A100 checks only when a decision is blocked and Alex's estimated start is over 6 h away | Routine A100 screens |
| **Alex A100 40/80 GB** (free, queued) | All routine A100 work: screens, finals, gates, profiles, OFF/POLAR, long runs. Overnight batches | Anything needed within the hour |
| **Colab A100/L4** (free) | Labs, falsifiers, kernel bounds, A100 pre-screens | Final decisions (the A100 model varies) |
| **Workstations** (RTX 4000 Ada, at most 4, idle check) | CPU suites on leto (one per merge batch); Ada A/Bs; drifts and long accuracy runs | Unchecked jobs; files outside `/mnt/homes/zwingel` |
| **Local RTX 5060 Ti** | Building and smoke-testing kernels; quick falsifiers; 5060 Ti A/Bs | CPU suites; more than one GPU job at a time |
| **WSL** | Launcher host for Modal only | Running many clients; holding long jobs without a system unit |

Rule of thumb: **a paid GPU only confirms.** Bounds, labs and falsifiers run on free hardware first.

### 3.2 Lanes instead of task sessions

Run a few standing sessions ("lanes"), each in its own worktree and branch, for days:

| Lane | Owns | Main hardware |
| --- | --- | --- |
| H200 lane | H200 OMAT-0 switches | Local GPU for builds, Modal H200 for finals |
| A100 lane | A100 OMAT-0 switches | Colab and local for labs, Alex for finals |
| Models lane | MACE-OFF and MACE-POLAR | Alex |
| Lab lane | Idea intake, ledger check, bounds and falsifiers for all lanes | Local GPU, Colab, Ada |

When to use what:

- **New session:** only for work that owns a branch, runs longer than about 2 h, and needs no input from the conductor
  along the way.
- **Subagent:** anything read-only and short: ledger lookups, transcript searches, reviews, profile reading. The
  result comes back in the same turn.
- **Inline:** anything a few tool calls settle.

Target: at most 4 lanes plus the conductor, and under 10 new sessions a week (204 in the month so far).

### 3.3 Idea triage and kill rules

Each idea moves through fixed stages. It stops at the first one it fails.

| Stage | Where | Time box | Kill if |
| --- | --- | --- | --- |
| S0 Ledger check | Subagent | 10 min | Already closed in the optimisation ledger, or out of scope (multi-GPU, per-crystal tables, dynamics changes) |
| S1 Bound | Breakdown or profile already on file | 30 min | Bound under 1.0 % of the step on the target GPU (about 2–3× the noise floor). Pure work removal is exempt: one not-slower A/B on any GPU |
| S2 Lab or falsifier | Free GPU (local, Colab, Ada) | Half a lane-day | Kernel-level gain is less than twice what S1 needs |
| S3 Packed A/B | H200 on Modal, A100 on Alex | Next batch | Not faster in both build orders |
| S4 Gate | Same GPU, packed | Next batch | Fails `tools/force_stress_gate.py` (rounding switches only) |
| S5 Adopt | Branch, then leto suite at the next merge batch | Same day | – |

More rules:

- No idea goes back into the queue without new evidence.
- A round ends when its five largest open bounds add up to less than 3 %. Report that to the owner rather than
  starting the next revision.
- Every stop is written into the ledger with its bound. That keeps the next session from repeating it.

### 3.4 Standard job templates and caches

Move the scratchpad scripts into the repository as versioned, unchanging templates under `tools/jobs/`:

- **`ab_pack`**: one container, one baseline, N switches (aim for 4–8), both build orders. One model load and one
  compile for all of them.
- **`gate_pack`**: several gates in one container.
- **`profile`**: one breakdown or profile run.
- **Launcher**: reads a batch manifest (CSV: idea id, switch, GPU, profile, timeout). It copies the templates into
  the batch folder before launching, so edits never touch running jobs. It writes the profile, batch id and idea id
  to the ledger.

Caches:

- A Modal image with FastMACE's CUDA extensions prebuilt for sm_80 and sm_90, rebuilt only when the kernel sources
  change.
- A Modal Volume for the Triton, Inductor and torch-extension caches, keyed by commit and GPU.
- The slim source tarball (13 MB, code only) as the default for every Modal job.
- Alex: a fixed venv and cache under `$WORK`, reused by every job.

Shell rules, built into the templates:

- No inline multi-line `bash -c`, and no `$` inside `wsl …` from Git Bash. Every remote command is a script file.
- Computed paths are checked before any `cp -r` or `rm`.

### 3.5 Batching and queueing rules

- **Fixed cycle.** Launch a batch every 2 h in the daytime, or sooner once 4 or more switches are ready. Overnight,
  one large Alex batch and one Modal H200 batch.
- **Timeouts from data.** Set each template's timeout to twice its p90 wall time: about 20 min for an H200 A/B and
  25 min for an A100 A/B. The reserved worst case then falls from 5.3× to about 2× the real cost.
- **Stale reservations expire.** Each in-flight entry gets a time to live (timeout + 10 min), so a crash can no
  longer hold budget.
- **Client cap.** At most 6 Modal clients in WSL. Launch 25 s apart. Run batches as a `systemd-run` system unit with
  the Windows keep-alive.
- **One retry, then triage.** The launcher retries a failed job once. A second failure goes to a person to look at,
  not to another automatic loop.
- **Alex.** One job array per batch. One poller, at most every 15 min, with one ssh connection per poll that does all
  the work.
- **Waiting.** Every wait over 60 s uses `run_in_background` or Monitor and is never a foreground `sleep`.

### 3.6 Modal budget allocation

Each month has $120 (4 × $30). Proposed split:

| Profile | Role | Monthly envelope |
| --- | --- | --- |
| `fourth` | H200 finals | $30 |
| `third` | H200 finals and gates | $30 |
| `fzwingel` | H200 gates and profiles | $30 |
| `second` | A100 checks that cannot wait for Alex; overflow | $30 |

- **Weekly pacing.** Each profile may spend at most $8 per week. Unused money carries over. The last week of the
  month can spend what is left.
- **Expected ratio.** With A100 work on Alex, about 85–90 % of Modal money should go to the H200, up from 34 %.
- **Weekly check.** One `billing summary` per week, recorded in the ledger. Pause a profile when it passes its
  pace by more than $5.

### 3.7 Parallel-testing policy

| Test | Where | Parallel? |
| --- | --- | --- |
| Targeted test files for a change | Lane's worktree; CPU only on leto | Yes, one per lane |
| Full CPU suite (merge tier) | leto, after the idle check, in the background | One per merge batch, never per branch |
| Kernel build and smoke tests | Local RTX 5060 Ti | One GPU job at a time |
| Labs, falsifiers, bounds | Colab, local, Ada | Yes, across machines |
| Target-GPU A/B and gate | Modal H200, Alex A100 | Packed; whole batches in parallel across profiles and the Alex queue |
| Ada A/Bs, drifts | murphy, leto, leeloo (at most 4 GPUs, idle check) | Yes |

### 3.8 The conductor's role and handoffs

The conductor schedules and decides adoption. It does not run experiments itself.

- **One queue file.** For example `plans/queue.md`, with one row per idea: stage, lane, bound, batch, result. Lanes
  update their own rows. The conductor reads the file instead of messaging every lane.
- **Cycle.** Every 2 h: read lane notes, approve the batch manifest, launch it, adopt winners, update
  `defaults-by-gpu.md`, and start the leto suite for the merge batch.
- **One conductor at a time.** The queue file starts with a lock line naming the conductor's session id. A new
  conductor stops the old one before it starts work. This prevents the overlaps and duplicate sessions of 6 Nov.
- **Handoff file always current.** It lists running jobs, budget left per profile, open decisions and lane states.
  Hand off before the account's weekly limit is reached, not after.
- **Lessons go into tools.** A failure that costs over 30 min gets a fix in a template or a guard. A memory note alone
  is not enough.

## 4. Expected speed-up and how to measure it

All figures are estimates from this month's data. Measure them weekly from the ledger, the A/B JSON files and the
transcripts.

| Change | Expected effect | Metric | Baseline (this month) | Target |
| --- | --- | --- | --- | --- |
| Packed A/Bs (4–8 switches per job) | Fewer container starts and compiles per decision: about 2× more decisions per dollar | Modal $ per switch decision | about $1–2 per switch (both GPUs, before gating) | under $0.60 |
| Prebuilt extensions and a cache Volume | First-side setup from about 140 s to about 20 s; about 25 % off a typical A/B job | Timed-step share of wall time (`setup_s` and steps in the JSON) | 17 % (H200), 26 % (A100) | 50 % or more |
| Timeouts at 2× p90, reservation expiry | No refusals while money is left; more concurrent jobs per profile | Refusals per batch; reserved ÷ real cost | 32 refusals; 5.3× | 0; about 2× |
| A100 to Alex and Colab | About $100 a month of Modal freed for the H200 | A100 share of Modal $; median Alex queue wait | 47 %; hours | under 15 %; known and planned for |
| Kill rules S0–S2 | 40–60 % fewer paid A/Bs for the same number of adoptions | Hit rate of paid A/Bs; $ per adopted switch | about $8 per adoption (5–7 Nov) | 40 % or more; under $4 |
| Background waits | Sessions free during waits; about 120 h of blocked time freed | Hours of foreground calls over 60 s per week | about 30 h a week | under 3 h |
| Script templates, client cap, system units | No repeat of the WSL and quoting losses | WSL crashes; quoting errors; relaunched batches | 5 crashes; about 260 errors | 0; under 10 |
| Lanes, one conductor, handoff file | Less coordination; no duplicate sessions | Coordination share of conductor tool calls; sessions started | 30 %; 204 | under 15 %; under 10 a week |
| Fixed 2-hour batch cycle | Predictable idea-to-default time | Time from S3 launch to adoption (median) | about a day, with relaunches | under 4 h for the H200; next morning for the A100 via Alex |

Together, F1, F2 and F7 suggest 2–3× more adoption decisions per Modal dollar. The cycle time should become steady,
because waits are planned rather than polled.

## 5. Quick wins and larger changes

### Quick wins (a day or less each)

1. Set template timeouts to 2× p90 (H200 A/B 20 min, A100 25 min) and give in-flight reservations a time to live.
2. Make the slim tarball and a client cap of 6 the defaults in `ab_job.sh`, and launch as a system unit with the
   keep-alive. Most of this exists already in the conductor's scratchpad.
3. Pack every A/B job with 4–8 switches. `ab_job.sh` already takes a list of A/B entries.
4. Apply kill rules S0–S2 to the current queue before the next paid batch.
5. Do every wait over 60 s in the background. Run one Alex poller at most every 15 min.
6. Add `profile`, `batch` and `idea` columns to the Modal ledger.
7. Send A100 labs and pre-screens to Colab. Send A100 finals to an overnight Alex batch.
8. Put a lock line naming the active conductor in the queue file.

### Larger changes (a few days each)

1. A Modal image with prebuilt extensions, plus a cache Volume.
2. Move the templates and the manifest-driven launcher into `tools/jobs/`, unchanged during a run, with ledger
   integration.
3. An Alex job-array template with a nightly cycle, including gates and profiles.
4. Restructure the work into the four lanes, with `plans/queue.md` as the single queue.
5. A weekly metrics script that reads the ledger, the A/B JSON files and the transcripts, and prints the table in
   section 4.

## Appendix: method and limits

- Transcripts: 222 JSONL files under `C:\Users\fzwin\.claude\projects\` (`D--Margraf*` and
  `G--Margraf-ResearchModule*`), parsed for timestamps, tool calls and tool results. "Active time" counts gaps of
  5 min or less. Pattern counts (quoting errors, budget text) include some repeats of the same text and are rough.
- Modal: `~/.fastmace-modal/modal-ledger.csv` in WSL (389 rows), plus `modal.json` and A/B JSON files in 309 run
  folders under `~/fm-runs`. Costs are GPU list-price estimates. Metered cost is higher, because the cap covers CPU,
  memory and container start as well. The ledger has no profile column, so per-profile figures come from the
  conductor's billing checks.
- Git: `git log --first-parent main` for merge times; `docs/optimisation-ledger.md` and
  `docs/current/defaults-by-gpu.md` for adoptions and hit rates.
- Alex and workstation figures come from session transcripts and notes. There is no central job log for them, so
  their numbers are less complete.
- MkDocs: `docs/process/` is not in `mkdocs.yml`'s nav or `exclude_docs`. Add `process/` to `exclude_docs` before
  committing this file, or `tests/cpu/test_docs_build.py` (strict build) will fail.
