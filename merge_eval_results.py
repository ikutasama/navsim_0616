"""
合并多GPU评估结果的CSV文件

Usage:
  python merge_eval_results.py --input_dir $OPENSCENE_DATA_ROOT/exp/eval_results
"""

import argparse
import os
import glob
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="合并多GPU评估结果")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="包含alpamayo_pdm_scores_*.csv的目录")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "alpamayo_pdm_scores_*.csv")))
    logger.info(f"  找到 {len(csv_files)} 个CSV文件")

    if len(csv_files) == 0:
        logger.error("  没有找到CSV文件!")
        return

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        logger.info(f"    {os.path.basename(f)}: {len(df)} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    # 去重（按token）
    if "token" in merged.columns:
        merged = merged.drop_duplicates(subset=["token"], keep="first")

    # 保存
    out_path = os.path.join(args.input_dir, "alpamayo_pdm_scores_merged.csv")
    merged.to_csv(out_path, index=False)

    # 统计
    valid_df = merged[merged["valid"] == True]
    logger.info(f"  合并结果: {len(merged)} rows, 有效={len(valid_df)}")

    if len(valid_df) > 0 and "pdm_score" in valid_df.columns:
        avg_pdm = valid_df["pdm_score"].mean()
        logger.info(f"  平均PDM Score: {avg_pdm:.4f}")

    logger.info(f"  保存到: {out_path}")


if __name__ == "__main__":
    main()
