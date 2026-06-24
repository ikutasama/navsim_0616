"""
Launch Alpamayo1.5 NavSim evaluation on multiple GPUs, wait for all shards, then merge results.

Example:
  python run_alpamayo_eval_multi_gpu.py --gpus 0 1 2 3
  python run_alpamayo_eval_multi_gpu.py --gpus 4 5 6 7 --max_eval_tokens 40

Notes:
- Each child process sees exactly one physical GPU via CUDA_VISIBLE_DEVICES=<gpu>.
- Therefore each child uses --device cuda:0 internally.
- Results are written into a fresh run directory to avoid merging stale CSV files.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def default_from_env(name: str, suffix: str = "", fallback: str = "") -> str:
    base = os.environ.get(name, "")
    if base:
        return base.rstrip("/") + suffix
    return fallback


def main():
    parser = argparse.ArgumentParser(description="Run Alpamayo NavSim eval on N GPUs and merge shard CSVs")
    parser.add_argument("--gpus", nargs="+", required=True, help="Physical GPU ids, e.g. --gpus 0 1 2 3")
    parser.add_argument("--navsim_log_path", default=default_from_env("OPENSCENE_DATA_ROOT", "/navsim_logs/mini"))
    parser.add_argument("--sensor_blobs_path", default=default_from_env("OPENSCENE_DATA_ROOT", "/sensor_blobs/mini"))
    parser.add_argument("--metric_cache_path", default=default_from_env("OPENSCENE_DATA_ROOT", "/metric_cache"))
    parser.add_argument("--model_path", default="/data/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B")
    parser.add_argument("--output_root", default=default_from_env("OPENSCENE_DATA_ROOT", "/exp/eval_results", "./eval_results"))
    parser.add_argument("--run_id", default=None, help="Run id directory name. Default: timestamped alpamayo_eval_YYYYmmdd_HHMMSS")
    parser.add_argument("--max_eval_tokens", type=int, default=0, help="0=all tokens; >0 only first N tokens before sharding")
    parser.add_argument("--save_cot_json", action="store_true")
    parser.add_argument("--merged_name", default="alpamayo_pdm_scores_merged", help="Merged CSV basename without .csv")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child processes")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    run_id = args.run_id or datetime.now().strftime("alpamayo_eval_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).expanduser().resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = [str(g) for g in args.gpus]
    total_shards = len(gpus)
    if total_shards < 1:
        raise ValueError("Need at least one GPU")

    print(f"[launcher] GPUs: {gpus}", flush=True)
    print(f"[launcher] total_shards: {total_shards}", flush=True)
    print(f"[launcher] output_dir: {output_dir}", flush=True)

    processes = []
    for shard_id, gpu in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            args.python,
            str(repo_dir / "run_alpamayo_eval.py"),
            "--navsim_log_path", args.navsim_log_path,
            "--sensor_blobs_path", args.sensor_blobs_path,
            "--metric_cache_path", args.metric_cache_path,
            "--model_path", args.model_path,
            "--output_dir", str(output_dir),
            "--max_eval_tokens", str(args.max_eval_tokens),
            "--shard_id", str(shard_id),
            "--total_shards", str(total_shards),
            "--device", "cuda:0",
        ]
        if args.save_cot_json:
            cmd.append("--save_cot_json")

        log_path = output_dir / f"shard{shard_id}_gpu{gpu}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        print(f"[launcher] start shard={shard_id}/{total_shards} physical_gpu={gpu} log={log_path}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(repo_dir), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((shard_id, gpu, proc, log_f, log_path))

    failed = []
    for shard_id, gpu, proc, log_f, log_path in processes:
        ret = proc.wait()
        log_f.close()
        if ret != 0:
            failed.append((shard_id, gpu, ret, log_path))
            print(f"[launcher] FAILED shard={shard_id} gpu={gpu} exit={ret} log={log_path}", flush=True)
        else:
            print(f"[launcher] done shard={shard_id} gpu={gpu} log={log_path}", flush=True)

    if failed:
        print("[launcher] At least one shard failed; not merging. Failed shards:", flush=True)
        for shard_id, gpu, ret, log_path in failed:
            print(f"  shard={shard_id} gpu={gpu} exit={ret} log={log_path}", flush=True)
        sys.exit(1)

    merge_cmd = [
        args.python,
        str(repo_dir / "merge_eval_results.py"),
        "--input_dir", str(output_dir),
        "--output_name", args.merged_name,
    ]
    print(f"[launcher] merging: {' '.join(merge_cmd)}", flush=True)
    subprocess.run(merge_cmd, cwd=str(repo_dir), check=True)

    merged_csv = output_dir / f"{args.merged_name}.csv"
    latest_link = Path(args.output_root).expanduser().resolve() / f"{args.merged_name}_latest.csv"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(merged_csv)
        print(f"[launcher] latest symlink: {latest_link} -> {merged_csv}", flush=True)
    except Exception as e:
        print(f"[launcher] could not create latest symlink: {e}", flush=True)

    print(f"[launcher] merged_csv: {merged_csv}", flush=True)


if __name__ == "__main__":
    main()
