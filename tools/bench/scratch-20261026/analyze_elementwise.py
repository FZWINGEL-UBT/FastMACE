import csv

ops = []
with open("/mnt/d/Margraf/FastMACE/results/2-T20a-step-breakdown-20261020/NbCl5/ops.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ops.append((float(row["us_per_call"]), row["op"], row["category"], row["launches_per_call"], row.get("kernel", "")))

ops.sort(key=lambda x: x[0], reverse=True)
print(f"Total ops: {len(ops)}. Top 20 by time per force call:")
for us, op, cat, invocations, dims in ops[:20]:
    print(f"  {us:8.1f} µs | {float(invocations):4.1f} inv | {op:<32} | {cat}")
