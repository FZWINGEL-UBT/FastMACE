# 2-T10 Durable output and restart files

Plan task 2-T10 (`plans/phase-2.md`). SPEC §10.6 (atomic restart), §12.2 "Restart".

## What changed

Before this task, chunk files were flushed but not fsynced (1b-T06). The restart file was written to a
temporary file and renamed, with no fsync. After a power loss or an NFS client crash, a restart
could therefore count rows and frames that were never on disk, or be empty after the rename.

`fastmace/io/durable.py` holds the durable write. It is always on, with no switch (conductor
answer Q4). At every restart point the order is:

1. `ChunkedOutput.restart_point()` closes the current chunk pair, fsyncs `thermo.<n>.csv` and
   `trajectory.<n>.traj`, then fsyncs their directory. `close()` does the same for the last chunk.
2. `write_restart` writes `restart.npz.tmp.npz` and flushes and fsyncs it. It then calls
   `os.replace` onto `restart.npz` and fsyncs the directory. The restart's bytes are unchanged.
3. `simulation.json` (`simulation._write_json`) and `metadata.json`
   (`provenance.write_metadata`) are written the same way, through `atomic_write`.

Rows and frames written between restart points are flushed only, as before. The directory fsync is
POSIX only. It is skipped when the file system refuses it with `EINVAL`, `ENOTSUP` or `EOPNOTSUPP`.
Any other error raises, and so does a failed file fsync (conductor answer Q2).

Tests: `tests/cpu/test_durable.py` replaces `os.fsync` and `os.replace` with recorders. Each fsync
is named by the file its descriptor refers to. The tests check the order above for the synchronous
and the asynchronous writer, including a stop request. They also check that nothing is fsynced
between restart points, that the errno rules hold, and that the two JSON records are written
durably.

## Cost

`tools/durable_cost.py` runs alternating blocks of fsync on and fsync off in one process. Fsync off
replaces `os.fsync` with a no-op, which is the 1b behaviour. Each block has 5 restart points, and
there are 6 blocks of each, so 30 restart points per mode. Each restart point times
`restart_point()` plus a restart-sized `write_atomically`, right after a full chunk has been
written. The chunk is 500 frames and rows of 512 atoms (18.6 MB), which is 10,000 steps at the
default cadences. The restart is 138 kB. All of the chunk is still dirty when it is fsynced, so
this is an upper bound.

| machine | file system | directory | fsync on, median (min-max) | fsync off, median | extra |
| --- | --- | --- | --- | --- | --- |
| murphy (CPU only, idle) | NFS | `/mnt/homes/zwingel/fastmace-2-T10/scratch` | 7.14 ms (6.59-8.30) | 7.14 ms | 0.005 ms |
| PC, WSL2 | ext4 | `~/fastmace-2-T10-scratch` | 25.4 ms (21.3-40.0) | 0.26 ms | 25.2 ms |

On NFS the fsync costs nothing measurable. The Linux NFS client already writes and commits a file's
data when the file is closed (close-to-open consistency), so the 7 ms is paid with or without fsync.
On local ext4 (WSL2's virtual disk), fsync makes the cost real: about 25 ms per restart point,
spent writing the 18.6 MB chunk to disk. At one restart point per 10,000 steps, that is 2.5 µs per
step, under 0.1 % of even a 5 ms MD step. On NFS it is 0.5 ns per step.

Environment: murphy, `/mnt/homes/zwingel/environments/mace_cueq_latest` (Python 3.10.12), no GPU
used. FastMACE commit 88c1abb (branch `fastmace/2-T10`). The WSL run (PC, WSL2 Ubuntu, CPU only) used
`~/environments/fastmace_cpu`.
Raw times: `results/2-T10-durable-20261020/`.
