"""
合并多GPU分片的评估CSV结果

Usage:
  python merge_eval_results.py --input_dir /path/to/eval_results

输出一个合并后的CSV，以及汇总PDM score统计。
"""

import argparse
import os
import glob
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="合并多GPU评估结果CSV")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="包含alpamayo_pdm_scores_*.csv的目录")
    parser.add_argument("--output_name", type=str, default="alpamayo_pdm_scores_merged",
                        help="合并后CSV文件名（不含.csv后缀）")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "alpamayo_pdm_scores_*.csv")))
    # 排除已合并的文件
    csv_files = [f for f in csv_files if "merged" not in f]

    if len(csv_files) == 0:
        logger.error(f"在 {args.input_dir} 中没找到CSV文件!")
        return

    logger.info(f"找到 {len(csv_files)} 个CSV文件")

    dfs = []
    for f in csv_files:
        logger.info(f"  读取: {os.path.basename(f)}")
        try:
            df = pd.read_csv(f)
            logger.info(f"    {len(df)} 行")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"    读取失败: {e}")

    if len(dfs) == 0:
        logger.error("没有可合并的数据!")
        return

    merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"合并后: {len(merged)} 行")

    # 统计
    valid = merged[merged["valid"] == True]
    logger.info(f"有效场景: {len(valid)}")

    if len(valid) > 0 and "pdm_score" in valid.columns:
        avg_pdm = valid["pdm_score"].mean()
        logger.info(f"平均PDM Score: {avg_pdm:.4f}")

        sub_metrics = [
            "no_at_fault_collisions", "drivable_area_compliance",
            "driving_direction_compliance", "traffic_light_compliance",
            "ego_progress", "time_to_collision_within_bound",
            "lane_keeping", "history_comfort",
        ]
        logger.info("各子指标:")
        for m in sub_metrics:
            if m in valid.columns:
                val = valid[m].mean()
                fail_rate = (valid[m] == 0).sum() / len(valid)
                logger.info(f"  {m}: mean={val:.4f}, fail_rate={fail_rate:.2%}")

    # 保存
    out_path = os.path.join(args.input_dir, f"{args.output_name}.csv")
    merged.to_csv(out_path, index=False)
    logger.info(f"合并结果保存到: {out_path}")


if __name__ == "__main__":
    main()
