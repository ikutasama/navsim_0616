"""
Merge continuous clips results from multiple run directories into one complete mini set.

Usage:
  python merge_continuous_runs.py \
    --run_dirs dir1 dir2 dir3 \
    --output_dir /path/to/merged_output
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def merge_runs(run_dirs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    all_clips = []

    for rd in run_dirs:
        rd = Path(rd)
        # Support both top-level CSV and shard subdirs
        frame_csv = rd / "continuous_frame_results.csv"
        clip_csv = rd / "continuous_clip_summary.csv"

        # If not at top level, check shards
        if not frame_csv.exists():
            shard_dirs = sorted((rd / "shards").glob("gpu*")) if (rd / "shards").exists() else []
            for sd in shard_dirs:
                sf = sd / "continuous_frame_results.csv"
                sc = sd / "continuous_clip_summary.csv"
                if sf.exists():
                    df = pd.read_csv(sf)
                    all_frames.append(df)
                    print(f"  frames from {sf}: {len(df)} rows")
                if sc.exists():
                    dc = pd.read_csv(sc)
                    all_clips.append(dc)
                    print(f"  clips from {sc}: {len(dc)} rows")
                # Copy videos
                sv = sd / "videos"
                if sv.exists():
                    for v in sv.glob("*.mp4"):
                        dst = videos_dir / v.name
                        if not dst.exists():
                            shutil.copy2(v, dst)
        else:
            if frame_csv.exists():
                df = pd.read_csv(frame_csv)
                all_frames.append(df)
                print(f"  frames from {frame_csv}: {len(df)} rows")
            if clip_csv.exists():
                dc = pd.read_csv(clip_csv)
                all_clips.append(dc)
                print(f"  clips from {clip_csv}: {len(dc)} rows")
            # Copy videos
            sv = rd / "videos"
            if sv.exists():
                for v in sv.glob("*.mp4"):
                    dst = videos_dir / v.name
                    if not dst.exists():
                        shutil.copy2(v, dst)

    if not all_frames:
        print("ERROR: No frame CSVs found!")
        return
    if not all_clips:
        print("ERROR: No clip CSVs found!")
        return

    # Concat and deduplicate (in case of overlapping runs)
    frame_df = pd.concat(all_frames, ignore_index=True)
    clip_df = pd.concat(all_clips, ignore_index=True)

    # Deduplicate by (clip_index, frame_index) keeping last
    if "clip_index" in frame_df.columns and "frame_index" in frame_df.columns:
        before = len(frame_df)
        frame_df = frame_df.drop_duplicates(subset=["clip_index", "frame_index"], keep="last")
        frame_df = frame_df.sort_values(["clip_index", "frame_index"], kind="stable").reset_index(drop=True)
        print(f"  frames dedup: {before} -> {len(frame_df)}")
    if "clip_index" in clip_df.columns:
        before = len(clip_df)
        clip_df = clip_df.drop_duplicates(subset=["clip_index"], keep="last")
        clip_df = clip_df.sort_values("clip_index", kind="stable").reset_index(drop=True)
        print(f"  clips dedup: {before} -> {len(clip_df)}")

    # Save merged CSVs
    frame_df.to_csv(output_dir / "continuous_frame_results.csv", index=False)
    clip_df.to_csv(output_dir / "continuous_clip_summary.csv", index=False)

    # Print summary stats
    print(f"\n=== Merged Summary ===")
    print(f"  Total clips: {len(clip_df)}")
    print(f"  Total frames: {len(frame_df)}")
    print(f"  Clip range: {clip_df['clip_index'].min()} - {clip_df['clip_index'].max()}")
    if "mean_fde" in clip_df.columns:
        valid = clip_df["mean_fde"].dropna()
        print(f"  Mean FDE: {valid.mean():.2f}")
    if "mean_ade" in clip_df.columns:
        valid = clip_df["mean_ade"].dropna()
        print(f"  Mean ADE: {valid.mean():.2f}")
    if "complexity_level" in clip_df.columns:
        print(f"  Complexity distribution:")
        print(clip_df["complexity_level"].value_counts().sort_index().to_string())
    if "video_path" in clip_df.columns:
        n_videos = clip_df["video_path"].astype(str).str.len().gt(0).sum()
        print(f"  Videos: {n_videos}")
    print(f"\n  Output: {output_dir}")
    print(f"  Videos: {videos_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True, help="Multiple run output directories")
    parser.add_argument("--output_dir", required=True, help="Merged output directory")
    args = parser.parse_args()
    merge_runs(args.run_dirs, args.output_dir)
