import json

for line in open("int8emu.jsonl"):
    d = json.loads(line)
    if "shape" not in d:
        print(d)
        continue
    one = d.get("int8_one_product_us")
    print(d["shape"], "sgemm %.0f us err %.1e | int8x1 %s | emu3 %.0f+%.0f us err %.1e | emu4 %.0f+%.0f us err %.1e" % (
        d["sgemm_us"], d["sgemm_err"], "-" if one is None else "%.0f" % one, d["emu3_us"], d["slice_a3_us"],
        d["emu3_err"], d["emu4_us"], d["slice_a4_us"], d["emu4_err"]))
