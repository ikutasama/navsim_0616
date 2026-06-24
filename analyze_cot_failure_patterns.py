"""
Analyze weak Alpamayo1.5 NavSim cases from CoT / trajectory / PDM results.

This script is intentionally cheap and rule-based. It answers questions like:
- Which scenes have low PDM and low CoT-trajectory consistency?
- In low-PDM scenes, are CoTs longer or shorter?
- Do low-PDM CoTs omit relevant objects?
- Do low-PDM CoTs omit other-agent intention reasoning?

Input should usually be cot_consistency_analysis.csv produced by analyze_cot_consistency.py.

Example:
  python analyze_cot_failure_patterns.py \
    --analysis_csv $OPENSCENE_DATA_ROOT/exp/cot_analysis/cot_consistency_analysis.csv \
    --output_dir $OPENSCENE_DATA_ROOT/exp/cot_failure_patterns
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_text(*parts) -> str:
    vals = []
    for p in parts:
        if p is None:
            continue
        try:
            if pd.isna(p):
                continue
        except Exception:
            pass
        vals.append(str(p))
    return " ".join(vals).strip()


def has(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def add_cot_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        cot = safe_text(row.get("cot", ""))
        meta = safe_text(row.get("meta_action", ""))
        answer = safe_text(row.get("answer", ""))
        text = safe_text(cot, meta, answer)
        text_l = text.lower()
        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?", text)

        object_kw = r"\b(vehicle|car|truck|bus|van|pedestrian|walker|cyclist|bicycle|bike|motorcycle|traffic|object|obstacle|agent|lead vehicle|front vehicle)\b"
        vuln_kw = r"\b(pedestrian|walker|cyclist|bicycle|bike|motorcycle|crosswalk)\b"
        vehicle_kw = r"\b(vehicle|car|truck|bus|van|lead vehicle|front vehicle|traffic)\b"
        spatial_kw = r"\b(left|right|ahead|front|behind|rear|oncoming|adjacent|side|lane|intersection|crosswalk|merge|turn|curve)\b"
        other_intent_kw = r"\b(yield|yielding|merge|merging|cut in|cut-in|turning|turn|crossing|braking|brake|decelerating|accelerating|stopping|parked|waiting|approaching|oncoming|entering|exiting|overtaking|changing lane|lane change)\b"
        ego_decision_kw = r"\b(stop|slow|decelerate|brake|yield|keep|maintain|continue|turn|nudge|avoid|overtake|go around|change lane|proceed)\b"
        risk_kw = r"\b(risk|danger|hazard|collision|conflict|close|near|caution|careful|safe|unsafe|block|blocked|occlusion|crowded)\b"
        rule_kw = r"\b(red light|green light|yellow light|traffic light|signal|stop sign|lane|right of way|crosswalk|speed limit)\b"
        causal_kw = r"\b(because|therefore|so|thus|since|as a result|due to|in order to|to avoid)\b"
        uncertainty_kw = r"\b(may|might|likely|possibly|potential|uncertain|appears|seems)\b"

        feat = row.to_dict()
        feat.update({
            "cot_text": text,
            "cot_empty": len(text_l.strip()) == 0,
            "cot_len_chars_raw": len(text),
            "cot_len_words_raw": len(words),
            "cot_sentence_count": max(1, len(re.findall(r"[.!?;]+", cot))) if cot else 0,
            "mentions_object_any": has(object_kw, text_l),
            "mentions_vehicle_obj": has(vehicle_kw, text_l),
            "mentions_vulnerable_obj": has(vuln_kw, text_l),
            "mentions_spatial_relation": has(spatial_kw, text_l),
            "mentions_other_agent_intent": has(other_intent_kw, text_l),
            "mentions_ego_decision": has(ego_decision_kw, text_l),
            "mentions_risk_or_conflict": has(risk_kw, text_l),
            "mentions_traffic_rule": has(rule_kw, text_l),
            "has_causal_connector": has(causal_kw, text_l),
            "has_uncertainty_marker": has(uncertainty_kw, text_l),
            "object_keyword_count": count(object_kw, text_l),
            "other_intent_keyword_count": count(other_intent_kw, text_l),
            "risk_keyword_count": count(risk_kw, text_l),
        })
        feat["object_aware_score"] = float(
            feat["mentions_object_any"]
            + feat["mentions_spatial_relation"]
            + feat["mentions_risk_or_conflict"]
            + feat["mentions_other_agent_intent"]
        ) / 4.0
        feat["cot_specificity_score"] = float(
            feat["mentions_object_any"]
            + feat["mentions_spatial_relation"]
            + feat["mentions_other_agent_intent"]
            + feat["mentions_risk_or_conflict"]
            + feat["mentions_traffic_rule"]
            + feat["has_causal_connector"]
        ) / 6.0
        rows.append(feat)
    return pd.DataFrame(rows)


def summarize_group(df: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    sub = df[mask].copy()
    out = {"group": name, "n": len(sub)}
    if len(sub) == 0:
        return out
    numeric_cols = [
        "pdm_score", "cot_traj_consistency", "cot_len_words_raw", "cot_len_chars_raw",
        "cot_sentence_count", "object_aware_score", "cot_specificity_score",
        "n_objects", "n_vehicles", "n_pedestrians", "complexity_score",
    ]
    bool_cols = [
        "cot_empty", "mentions_object_any", "mentions_vehicle_obj", "mentions_vulnerable_obj",
        "mentions_spatial_relation", "mentions_other_agent_intent", "mentions_ego_decision",
        "mentions_risk_or_conflict", "mentions_traffic_rule", "has_causal_connector",
        "has_uncertainty_marker",
    ]
    for c in numeric_cols:
        if c in sub.columns:
            vals = pd.to_numeric(sub[c], errors="coerce")
            out[f"{c}_mean"] = vals.mean()
            out[f"{c}_median"] = vals.median()
    for c in bool_cols:
        if c in sub.columns:
            out[f"{c}_rate"] = sub[c].fillna(False).astype(bool).mean()
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze CoT patterns in low-PDM / inconsistent Alpamayo NavSim cases")
    parser.add_argument("--analysis_csv", required=True, help="cot_consistency_analysis.csv from analyze_cot_consistency.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--low_pdm_quantile", type=float, default=0.25, help="Lowest quantile used as weak PDM group")
    parser.add_argument("--high_pdm_quantile", type=float, default=0.75, help="Highest quantile used as strong PDM comparison group")
    parser.add_argument("--inconsistent_threshold", type=float, default=0.5, help="cot_traj_consistency <= threshold is inconsistent")
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.analysis_csv)
    if "valid" in df.columns:
        df = df[df["valid"] == True].copy()
    if "pdm_score" not in df.columns:
        raise ValueError("analysis_csv must contain pdm_score")

    df = add_cot_pattern_features(df)
    df["pdm_score"] = pd.to_numeric(df["pdm_score"], errors="coerce")
    df["cot_traj_consistency"] = pd.to_numeric(df.get("cot_traj_consistency", np.nan), errors="coerce")

    valid_pdm = df["pdm_score"].dropna()
    low_thr = float(valid_pdm.quantile(args.low_pdm_quantile))
    high_thr = float(valid_pdm.quantile(args.high_pdm_quantile))
    df["is_low_pdm"] = df["pdm_score"] <= low_thr
    df["is_high_pdm"] = df["pdm_score"] >= high_thr
    df["is_inconsistent"] = df["cot_traj_consistency"].notna() & (df["cot_traj_consistency"] <= args.inconsistent_threshold)
    df["is_weak_cot_traj_case"] = df["is_low_pdm"] & df["is_inconsistent"]

    # Ranking: low PDM + low consistency + generic CoT omissions. Higher = more suspicious.
    pdm_bad = 1.0 - df["pdm_score"].clip(0, 1).fillna(0.0)
    cons_bad = 1.0 - df["cot_traj_consistency"].clip(0, 1).fillna(0.5)
    omission = (
        (~df["mentions_object_any"].fillna(False)).astype(float)
        + (~df["mentions_other_agent_intent"].fillna(False)).astype(float)
        + (~df["mentions_spatial_relation"].fillna(False)).astype(float)
        + (~df["mentions_risk_or_conflict"].fillna(False)).astype(float)
    ) / 4.0
    df["weak_case_score"] = 0.45 * pdm_bad + 0.35 * cons_bad + 0.20 * omission

    enriched_csv = Path(args.output_dir) / "cot_failure_pattern_enriched.csv"
    df.to_csv(enriched_csv, index=False)

    group_rows = [
        summarize_group(df, "all_valid", pd.Series(True, index=df.index)),
        summarize_group(df, f"low_pdm_bottom_{args.low_pdm_quantile:.2f}_pdm_le_{low_thr:.4f}", df["is_low_pdm"]),
        summarize_group(df, f"high_pdm_top_{1-args.high_pdm_quantile:.2f}_pdm_ge_{high_thr:.4f}", df["is_high_pdm"]),
        summarize_group(df, f"inconsistent_cons_le_{args.inconsistent_threshold:.2f}", df["is_inconsistent"]),
        summarize_group(df, "weak_low_pdm_and_inconsistent", df["is_weak_cot_traj_case"]),
    ]
    summary = pd.DataFrame(group_rows)
    summary_csv = Path(args.output_dir) / "cot_pattern_summary_by_group.csv"
    summary.to_csv(summary_csv, index=False)

    # Lift table: low-PDM rate minus high-PDM rate for interpretable boolean CoT traits.
    bool_cols = [
        "cot_empty", "mentions_object_any", "mentions_vehicle_obj", "mentions_vulnerable_obj",
        "mentions_spatial_relation", "mentions_other_agent_intent", "mentions_ego_decision",
        "mentions_risk_or_conflict", "mentions_traffic_rule", "has_causal_connector",
        "has_uncertainty_marker",
    ]
    lift_rows = []
    low = df[df["is_low_pdm"]]
    high = df[df["is_high_pdm"]]
    for c in bool_cols:
        low_rate = low[c].fillna(False).astype(bool).mean() if len(low) else np.nan
        high_rate = high[c].fillna(False).astype(bool).mean() if len(high) else np.nan
        lift_rows.append({
            "feature": c,
            "low_pdm_rate": low_rate,
            "high_pdm_rate": high_rate,
            "low_minus_high": low_rate - high_rate,
            "interpretation": "positive means more common in low-PDM cases",
        })
    # Numeric differences.
    for c in ["cot_len_words_raw", "cot_len_chars_raw", "object_aware_score", "cot_specificity_score", "cot_traj_consistency"]:
        if c in df.columns:
            lift_rows.append({
                "feature": c,
                "low_pdm_rate": pd.to_numeric(low[c], errors="coerce").mean() if len(low) else np.nan,
                "high_pdm_rate": pd.to_numeric(high[c], errors="coerce").mean() if len(high) else np.nan,
                "low_minus_high": (pd.to_numeric(low[c], errors="coerce").mean() - pd.to_numeric(high[c], errors="coerce").mean()) if len(low) and len(high) else np.nan,
                "interpretation": "numeric mean: low-PDM minus high-PDM",
            })
    lift = pd.DataFrame(lift_rows).sort_values("low_minus_high", key=lambda s: s.abs(), ascending=False)
    lift_csv = Path(args.output_dir) / "cot_pattern_lift_low_vs_high_pdm.csv"
    lift.to_csv(lift_csv, index=False)

    rank_cols = [
        "token", "weak_case_score", "pdm_score", "cot_traj_consistency",
        "cot_len_words_raw", "cot_specificity_score", "object_aware_score",
        "mentions_object_any", "mentions_other_agent_intent", "mentions_risk_or_conflict",
        "mentions_spatial_relation", "n_objects", "n_vehicles", "n_pedestrians",
        "complexity_score", "meta_action", "cot", "answer",
    ]
    rank_cols = [c for c in rank_cols if c in df.columns]
    ranking = df.sort_values(["is_weak_cot_traj_case", "weak_case_score"], ascending=[False, False])[rank_cols]
    ranking_csv = Path(args.output_dir) / "top_weak_cot_traj_cases.csv"
    ranking.head(args.top_k).to_csv(ranking_csv, index=False)

    txt_path = Path(args.output_dir) / "top_weak_cot_traj_cases.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"low_pdm_threshold={low_thr:.6f}\n")
        f.write(f"high_pdm_threshold={high_thr:.6f}\n")
        f.write(f"inconsistent_threshold={args.inconsistent_threshold:.6f}\n\n")
        for i, (_, r) in enumerate(ranking.head(args.top_k).iterrows(), start=1):
            f.write("=" * 100 + "\n")
            f.write(f"#{i} token={r.get('token','')} weak_case_score={r.get('weak_case_score', np.nan):.4f} pdm={r.get('pdm_score', np.nan):.4f} consistency={r.get('cot_traj_consistency', np.nan)}\n")
            f.write(f"len_words={r.get('cot_len_words_raw', np.nan)} specificity={r.get('cot_specificity_score', np.nan)} object_aware={r.get('object_aware_score', np.nan)}\n")
            f.write(f"mentions_object={r.get('mentions_object_any', '')} mentions_other_intent={r.get('mentions_other_agent_intent', '')} mentions_risk={r.get('mentions_risk_or_conflict', '')}\n")
            f.write(f"meta_action: {safe_text(r.get('meta_action',''))}\n")
            f.write(f"cot: {safe_text(r.get('cot',''))}\n")
            ans = safe_text(r.get('answer',''))
            if ans:
                f.write(f"answer: {ans}\n")

    print(f"loaded rows: {len(df)}")
    print(f"low_pdm_threshold: {low_thr:.4f}; high_pdm_threshold: {high_thr:.4f}")
    print(f"weak low-PDM + inconsistent cases: {int(df['is_weak_cot_traj_case'].sum())}")
    print(f"saved: {enriched_csv}")
    print(f"saved: {summary_csv}")
    print(f"saved: {lift_csv}")
    print(f"saved: {ranking_csv}")
    print(f"saved: {txt_path}")


if __name__ == "__main__":
    main()
