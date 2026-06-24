"""
One-command full Alpamayo1.5 NavSim pipeline:
1) run inference/PDM evaluation on all or N tokens using multiple GPUs
2) merge shard CSVs
3) analyze CoT-trajectory consistency
4) analyze CoT failure patterns
5) optionally visualize top weak cases as PNG and per-scene GIFs

Example full mini evaluation on GPUs 1-4:
  python run_full_alpamayo_navsim_pipeline.py --gpus 1 2 3 4

Smoke test on 20 cases with visualization:
  python run_full_alpamayo_navsim_pipeline.py --gpus 1 2 3 4 --max_eval_tokens 20 --visualize_top_k 10 --make_case_gifs

Notes:
- max_eval_tokens=0 means all evaluable tokens.
- The script writes everything under one timestamped run directory.
- It assumes run_alpamayo_eval_multi_gpu.py already handles per-GPU sharding and merging.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def infer_data_root(repo_dir: Path) -> Path:
    env_root = os.environ.get("OPENSCENE_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    for cand in [repo_dir / "navsim" / "navsim_dataset", repo_dir / "navsim_dataset", repo_dir / "dataset"]:
        if (cand / "navsim_logs").exists() or (cand / "metric_cache").exists() or (cand / "sensor_blobs").exists():
            return cand.resolve()
    return (repo_dir / "navsim" / "navsim_dataset").resolve()


def run_cmd(cmd: List[str], cwd: Path, log_path: Path, env: Optional[dict] = None, keep_log: bool = False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] RUN: {' '.join(cmd)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[pipeline] FAILED exit={proc.returncode}: {' '.join(cmd)}", flush=True)
        print(f"[pipeline] tail -120 {log_path}:", flush=True)
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-120:]), flush=True)
        except Exception as e:
            print(f"<cannot read log: {e}>", flush=True)
        raise SystemExit(proc.returncode)
    if keep_log:
        print(f"[pipeline] OK log kept: {log_path}", flush=True)
    else:
        try:
            log_path.unlink()
        except Exception:
            pass
        print(f"[pipeline] OK", flush=True)


def require_path(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def cleanup_eval_intermediates(merged_csv: Path):
    """Keep human-facing merged CSV only; remove shard logs and per-shard CSVs."""
    eval_dir = merged_csv.parent
    patterns = [
        "shard*_gpu*.log",
        "alpamayo_pdm_scores_*_shard*.csv",
        "alpamayo_cot_outputs_*_shard*.json",
    ]
    removed = 0
    for pat in patterns:
        for p in eval_dir.glob(pat):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    if removed:
        print(f"[pipeline] cleaned {removed} shard/log intermediate files from {eval_dir}", flush=True)


def main():
    repo_dir = Path(__file__).resolve().parent
    data_root = infer_data_root(repo_dir)

    parser = argparse.ArgumentParser(description="Full Alpamayo1.5 NavSim inference + analysis pipeline")
    parser.add_argument("--gpus", nargs="+", required=True, help="Physical GPU ids, e.g. --gpus 1 2 3 4")
    parser.add_argument("--run_id", default=None, help="Default: full_alpamayo_navsim_YYYYmmdd_HHMMSS")
    parser.add_argument("--data_root", default=str(data_root))
    parser.add_argument("--metric_cache_path", default=None)
    parser.add_argument("--model_path", default="/data/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B")
    parser.add_argument("--output_root", default=None, help="Default: <data_root>/exp/full_pipeline")
    parser.add_argument("--max_eval_tokens", type=int, default=0, help="0=all evaluable tokens; >0 smoke-test first N tokens")
    parser.add_argument("--save_cot_json", action="store_true")
    parser.add_argument("--visualize_top_k", type=int, default=0, help="0 disables visualization; >0 creates top-K weak PNGs")
    parser.add_argument("--make_case_gifs", action="store_true", help="When visualizing, also create one GIF per scene")
    parser.add_argument("--select_visualization", choices=["weakest", "low_pdm", "inconsistent", "all"], default="weakest")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip_eval", action="store_true", help="Reuse --scores_csv and run analysis only")
    parser.add_argument("--scores_csv", default=None, help="Required with --skip_eval; otherwise optional existing merged CSV")
    parser.add_argument("--keep_logs", action="store_true", help="Keep step logs after successful completion; failure logs are always kept/printed")
    parser.add_argument("--keep_shards", action="store_true", help="Keep per-shard CSV/log files after merged CSV is produced")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    metric_cache_path = Path(args.metric_cache_path).expanduser().resolve() if args.metric_cache_path else data_root / "metric_cache"
    model_path = Path(args.model_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else data_root / "exp" / "full_pipeline"
    run_id = args.run_id or datetime.now().strftime("full_alpamayo_navsim_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    logs_dir = run_dir / "logs"
    eval_output_root = run_dir / "eval_results"
    cot_analysis_dir = run_dir / "cot_analysis"
    failure_dir = run_dir / "cot_failure_patterns"
    viz_dir = run_dir / "visualizations" / args.select_visualization

    run_dir.mkdir(parents=True, exist_ok=True)

    require_path(data_root, "data_root")
    require_path(metric_cache_path, "metric_cache_path")
    if not args.skip_eval:
        require_path(model_path, "model_path")

    env = os.environ.copy()
    env.setdefault("OPENSCENE_DATA_ROOT", str(data_root))
    env.setdefault("NAVSIM_EXP_ROOT", str(data_root / "exp"))
    env.setdefault("NAVSIM_DEVKIT_ROOT", str(repo_dir / "navsim"))

    print("[pipeline] ===== resolved paths =====", flush=True)
    print(f"[pipeline] repo_dir: {repo_dir}", flush=True)
    print(f"[pipeline] data_root: {data_root}", flush=True)
    print(f"[pipeline] metric_cache_path: {metric_cache_path}", flush=True)
    print(f"[pipeline] model_path: {model_path}", flush=True)
    print(f"[pipeline] run_dir: {run_dir}", flush=True)
    print(f"[pipeline] gpus: {args.gpus}", flush=True)
    print(f"[pipeline] max_eval_tokens: {args.max_eval_tokens} (0 means all)", flush=True)

    # 1) Eval / inference.
    if args.skip_eval:
        if not args.scores_csv:
            raise ValueError("--skip_eval requires --scores_csv")
        merged_csv = Path(args.scores_csv).expanduser().resolve()
        require_path(merged_csv, "scores_csv")
        print(f"[pipeline] skip eval; using scores_csv: {merged_csv}", flush=True)
    elif args.scores_csv:
        merged_csv = Path(args.scores_csv).expanduser().resolve()
        require_path(merged_csv, "scores_csv")
        print(f"[pipeline] using existing scores_csv instead of running eval: {merged_csv}", flush=True)
    else:
        eval_cmd = [
            args.python, str(repo_dir / "run_alpamayo_eval_multi_gpu.py"),
            "--gpus", *[str(g) for g in args.gpus],
            "--output_root", str(eval_output_root),
            "--max_eval_tokens", str(args.max_eval_tokens),
            "--model_path", str(model_path),
        ]
        if args.save_cot_json:
            eval_cmd.append("--save_cot_json")
        run_cmd(eval_cmd, cwd=repo_dir, log_path=logs_dir / "01_eval.log", env=env, keep_log=args.keep_logs)

        # run_alpamayo_eval_multi_gpu.py writes one timestamped subdir under eval_output_root.
        candidates = sorted(eval_output_root.glob("*/alpamayo_pdm_scores_merged.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            # In case run_id override inside launcher was changed, also check direct root.
            candidates = sorted(eval_output_root.glob("alpamayo_pdm_scores_merged.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"Cannot find merged CSV under {eval_output_root}")
        merged_csv = candidates[0].resolve()
        print(f"[pipeline] merged_csv: {merged_csv}", flush=True)
        if not args.keep_shards:
            cleanup_eval_intermediates(merged_csv)

    # Copy/link the merged CSV into run_dir for discoverability.
    canonical_scores = run_dir / "alpamayo_pdm_scores_merged.csv"
    if merged_csv != canonical_scores:
        try:
            if canonical_scores.exists() or canonical_scores.is_symlink():
                canonical_scores.unlink()
            canonical_scores.symlink_to(merged_csv)
        except Exception:
            shutil.copy2(merged_csv, canonical_scores)

    # 2) CoT consistency.
    consistency_cmd = [
        args.python, str(repo_dir / "analyze_cot_consistency.py"),
        "--scores_csv", str(merged_csv),
        "--metric_cache_path", str(metric_cache_path),
        "--output_dir", str(cot_analysis_dir),
    ]
    run_cmd(consistency_cmd, cwd=repo_dir, log_path=logs_dir / "02_cot_consistency.log", env=env, keep_log=args.keep_logs)
    consistency_csv = cot_analysis_dir / "cot_consistency_analysis.csv"
    require_path(consistency_csv, "cot_consistency_analysis.csv")

    # 3) CoT failure patterns.
    failure_cmd = [
        args.python, str(repo_dir / "analyze_cot_failure_patterns.py"),
        "--analysis_csv", str(consistency_csv),
        "--output_dir", str(failure_dir),
    ]
    run_cmd(failure_cmd, cwd=repo_dir, log_path=logs_dir / "03_cot_failure_patterns.log", env=env, keep_log=args.keep_logs)
    failure_csv = failure_dir / "cot_failure_pattern_enriched.csv"
    require_path(failure_csv, "cot_failure_pattern_enriched.csv")

    # 4) Optional visualization.
    if args.visualize_top_k > 0:
        viz_cmd = [
            args.python, str(repo_dir / "visualize_cot_trajectory_cases.py"),
            "--scores_csv", str(merged_csv),
            "--analysis_csv", str(failure_csv),
            "--metric_cache_path", str(metric_cache_path),
            "--navsim_log_path", str(data_root / "navsim_logs" / "mini"),
            "--sensor_blobs_path", str(data_root / "sensor_blobs" / "mini"),
            "--output_dir", str(viz_dir),
            "--select", args.select_visualization,
            "--top_k", str(args.visualize_top_k),
        ]
        if args.make_case_gifs:
            viz_cmd.append("--make_case_gifs")
        run_cmd(viz_cmd, cwd=repo_dir, log_path=logs_dir / "04_visualization.log", env=env, keep_log=args.keep_logs)

    # Summary.
    print("\n[pipeline] ===== DONE =====", flush=True)
    print(f"[pipeline] run_dir: {run_dir}", flush=True)
    print(f"[pipeline] merged_csv: {merged_csv}", flush=True)
    print(f"[pipeline] consistency_csv: {consistency_csv}", flush=True)
    print(f"[pipeline] failure_csv: {failure_csv}", flush=True)
    print(f"[pipeline] top_cases_csv: {failure_dir / 'top_weak_cot_traj_cases.csv'}", flush=True)
    print(f"[pipeline] top_cases_txt: {failure_dir / 'top_weak_cot_traj_cases.txt'}", flush=True)
    if args.visualize_top_k > 0:
        print(f"[pipeline] visualization_dir: {viz_dir}", flush=True)
    if args.keep_logs:
        print(f"[pipeline] logs_dir: {logs_dir}", flush=True)
    else:
        try:
            if logs_dir.exists() and not any(logs_dir.iterdir()):
                logs_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()
