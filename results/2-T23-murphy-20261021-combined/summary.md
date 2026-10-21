Traceback (most recent call last):
  File "/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t23-949aaf/tools/acceptance/summarize2.py", line 216, in <module>
    main()
  File "/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t23-949aaf/tools/acceptance/summarize2.py", line 208, in main
    summary = summarize_run(args.run_root, args.machine, args.allow_option_changes, args.one_sided)
  File "/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t23-949aaf/tools/acceptance/summarize2.py", line 130, in summarize_run
    summary["forces"] = forces_rows(run_root, ab, controls)
  File "/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t23-949aaf/tools/acceptance/summarize2.py", line 85, in forces_rows
    a = np.load(run_root / name / "forces" / f"{structure}-{side}.npy")
  File "/home/fzwin/environments/fastmace_cpu/lib/python3.10/site-packages/numpy/lib/_npyio_impl.py", line 451, in load
    fid = stack.enter_context(open(os.fspath(file), "rb"))
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t23-949aaf/results/2-T23-murphy-20261021-combined/regression2-ab-2-first/forces/NaCl-2.npy'
