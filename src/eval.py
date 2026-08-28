import glob
import json
import numpy as np

RUNS_DIR = "runs"
records = []


for path in sorted(glob.glob(f"{RUNS_DIR}/*.json")):
    with open(path) as f:
        records.append(json.load(f))

print(f"{'Item':<14}{'Ground Truth':>14}{'Predicted':>12}{'Error':>12}")
for record in records:
    print(f"{record['name']:<14}{record['ground_truth_ml']:>14.1f}{record['volume_ml']:>12.1f}{record['error_pct']:>11.1f}%")

errors = np.array([abs(record["error_pct"]) for record in records])
print(f"\nMean absolute percentage error over {len(errors)} scans: {errors.mean():.1f}%")