import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import extract
import depth_export
import segment
import fuse

RUNS_DIR = "runs"
FRAMES_DIR = "/kaggle/working/sam_frames"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrs", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--food-xy", nargs=2, type=int)
    parser.add_argument("--plate-xy", nargs=2, type=int)
    parser.add_argument("--ground-truth", type=float)
    # Whether to save keyframe 0 with ticks to read off point prompt coordinates
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)
    scan = extract.extract_scan(args.vrs)

    if args.preview:
        preview_path = f"{RUNS_DIR}/{args.name}_preview.png"
        segment.save_prompt_grid(scan["rectified_left_images"][0], preview_path)
        print("Wrote image grid preview to", preview_path)
        return
    
    fx = scan["intrinsics"][0]
    keyframe_depths = depth_export.compute_keyframe_depths(scan["rectified_left_images"], scan["rectified_right_images"], scan["baseline_m"], fx)

    food_masks, plate_masks = segment.segment_scan(scan["rectified_left_images"], tuple(args.food_xy), tuple(args.plate_xy), FRAMES_DIR)

    report = fuse.estimate_volume(keyframe_depths, food_masks, plate_masks, scan["keyframe_poses"], scan["intrinsics"], scan["image_shape"])

    if report["status"] != "ok":
        print("Scan failed:", report["status"])
        return

    # Export mesh as a PLY file
    mesh_path = f"{RUNS_DIR}/{args.name}.ply"
    report["mesh"].export(mesh_path)

    record = {"name": args.name, "vrs": args.vrs, "volume_ml": report["volume_ml"], "ground_truth_ml": args.ground_truth}
    print(f"Volume: {report['volume_ml']} mL")
    if args.ground_truth:
        error_ml = report["volume_ml"] - args.ground_truth
        record["error_pct"] = 100 * error_ml / args.ground_truth
        print(f"Ground Truth {args.ground_truth} mL, Error {error_ml:+.1f} mL ({record["error_pct"]:+.1f}%)")

    with open(f"{RUNS_DIR}/{args.name}.json", "w") as f:
        json.dump(record, f, indent=2)
    print("Wrote reconstructed mesh to", mesh_path, "and output values to", f"{RUNS_DIR}/{args.name}.json")


if __name__ == "__main__":
    main()