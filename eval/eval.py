import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


# ============================================================
# Helpers
# ============================================================

@dataclass
class TrialRecord:
    source_file: str
    source_path: str
    mode: Optional[str]                # textual / multimodal
    scenario: Optional[str]            # besides_edges / plain_grounds
    variant: Optional[str]             # naive, fewshot_balanced, multifewshot...
    fewshot_base: Optional[str]        # for multimodal logs
    inferred_condition: Optional[str]  # G0/G1A/... derived from runner metadata
    modality: Optional[str]            # none / text / image_text
    label_coverage: Optional[str]      # none / biased_walk / biased_jump / narrow / balanced
    image_id: str
    repeat_idx: int
    repeat_total: Optional[int]
    grade_raw: Optional[str]
    correct: Optional[int]
    expected_action: Optional[str]
    pred_action: Optional[str]
    parse_success: int
    observation_text: Optional[str]
    reason_text: Optional[str]
    raw_json_text: Optional[str]
    error_type: Optional[str]


def safe_read_text(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Could not read file: {path}")


def try_parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def extract_top_level_json_block(block_text: str) -> Optional[str]:
    start = block_text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(block_text)):
        ch = block_text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return block_text[start:i + 1]
    return None


def canonicalize_action_dict(action_obj: Any) -> Optional[str]:
    if action_obj is None:
        return None
    try:
        if isinstance(action_obj, dict) and "sequence" in action_obj:
            seq = action_obj.get("sequence", [])
            if not seq:
                return "EMPTY_SEQUENCE"
            first = seq[0]
            act = str(first.get("action", "")).strip()
            keys = first.get("keys", []) or []
            keys_sorted = sorted(str(k).strip() for k in keys)
            return f"{act}:{'+'.join(keys_sorted)}"

        if isinstance(action_obj, dict) and "action" in action_obj:
            act = str(action_obj.get("action", "")).strip()
            keys = action_obj.get("keys", []) or []
            keys_sorted = sorted(str(k).strip() for k in keys)
            return f"{act}:{'+'.join(keys_sorted)}"
    except Exception:
        return None
    return None


def parse_expected_action_from_line(line: str) -> Optional[str]:
    m = re.search(r"expected:\s*(\{.*\})\s*$", line.strip())
    if not m:
        return None
    parsed = try_parse_json_block(m.group(1))
    return canonicalize_action_dict(parsed)


def parse_parsed_action_from_line(line: str) -> Optional[str]:
    m = re.search(r"parsed:\s*(\{.*\})\s*$", line.strip())
    if not m:
        return None
    parsed = try_parse_json_block(m.group(1))
    return canonicalize_action_dict(parsed)


def infer_from_path(path: Path, root: Path) -> Dict[str, Optional[str]]:
    rel = path.relative_to(root)
    parts = rel.parts

    scenario = parts[0] if len(parts) >= 3 else None
    variant_folder = parts[1] if len(parts) >= 3 else None

    name_lower = path.name.lower()
    if name_lower.startswith("textual_"):
        mode = "textual"
    elif name_lower.startswith("multimodal_"):
        mode = "multimodal"
    else:
        mode = None

    return {
        "scenario_from_path": scenario,
        "variant_from_path": variant_folder,
        "mode_from_name": mode,
    }


def infer_experiment_condition(
    mode: Optional[str],
    variant: Optional[str],
    fewshot_base: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if mode == "textual":
        if variant == "naive":
            return "G0", "none", "none"
        if variant == "fewshot_walk_bias":
            return "G1A", "text", "biased_walk"
        if variant == "fewshot_jump_bias":
            return "G1B", "text", "biased_jump"
        if variant == "fewshot_narrow":
            return "G2A", "text", "narrow"
        if variant == "fewshot_balanced":
            return "G2B", "text", "balanced"
        if variant == "no_cot":
            return "NO_COT", "none", "none"
        return None, "text", None

    if mode == "multimodal":
        if variant in {"fewshot", "multifewshot"}:
            if fewshot_base == "walk_bias":
                return "G3A", "image_text", "biased_walk"
            if fewshot_base == "jump_bias":
                return "G3B", "image_text", "biased_jump"
            if fewshot_base == "narrow":
                return "G4A", "image_text", "narrow"
            if fewshot_base == "balanced":
                return "G4B", "image_text", "balanced"
        if variant == "naive":
            return "MM_NAIVE", "image_text", "none"
        return None, "image_text", None

    return None, None, None


def classify_error(expected_action: Optional[str], pred_action: Optional[str], raw_json: Optional[Dict[str, Any]]) -> Optional[str]:
    if expected_action is None or pred_action is None:
        return "parse_or_missing"

    if expected_action == pred_action:
        return None

    if expected_action == "hold:d+space" and pred_action == "hold:d":
        if raw_json and isinstance(raw_json, dict):
            reason = str(raw_json.get("reason", "")).lower()
            if (
                "prepare for a jump" in reason
                or "approach the gap" in reason
                or "position" in reason
                or "not yet in a position to jump" in reason
                or "jump later" in reason
            ):
                return "delayed_jump_logic"
        return "walk_instead_of_jump"

    if expected_action == "hold:d" and pred_action == "hold:d+space":
        return "jump_instead_of_walk"

    return "other_action_mismatch"


# ============================================================
# Ground truth loading
# ============================================================

def find_cases_root_from_output_root(output_root: Path) -> Optional[Path]:
    """
    If root is:
      .../train_cases/train_output
    return:
      .../train_cases

    If root is:
      .../test_cases/test_output
    return:
      .../test_cases
    """
    name = output_root.name.lower()
    parent = output_root.parent
    if name in {"train_output", "test_output"} and parent.exists():
        return parent
    return None


def load_ground_truth_file(gt_path: Path) -> Dict[str, str]:
    """
    Return mapping:
      image filename -> canonical expected action
    """
    if not gt_path.is_file():
        return {}
    try:
        raw = json.loads(gt_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    out: Dict[str, str] = {}
    for image_name, action_obj in raw.items():
        canon = canonicalize_action_dict(action_obj)
        if canon is not None:
            out[str(image_name)] = canon
    return out


def build_ground_truth_lookup(cases_root: Optional[Path]) -> Dict[Tuple[str, str], str]:
    """
    key = (scenario, image_id)
    value = canonical expected action
    """
    lookup: Dict[Tuple[str, str], str] = {}
    if cases_root is None or not cases_root.exists():
        return lookup

    for scenario_dir in cases_root.iterdir():
        if not scenario_dir.is_dir():
            continue
        gt_path = scenario_dir / "ground_truth.txt"
        gt_map = load_ground_truth_file(gt_path)
        for image_id, canon_action in gt_map.items():
            lookup[(scenario_dir.name, image_id)] = canon_action
    return lookup


# ============================================================
# Parsing
# ============================================================

def parse_single_txt(path: Path, root: Path, gt_lookup: Dict[Tuple[str, str], str]) -> List[TrialRecord]:
    text = safe_read_text(path)
    path_meta = infer_from_path(path, root)

    scenario_match = re.search(r"scenario:\s*([^\n\r]+?)(?:\s+variant:|\s*$)", text, flags=re.IGNORECASE)
    variant_match = re.search(r"variant:\s*([^\n\r]+?)(?:\s+fewshot_base:|\s*$)", text, flags=re.IGNORECASE)
    fewshot_base_match = re.search(r"fewshot_base:\s*([^\n\r]+)", text, flags=re.IGNORECASE)

    scenario_header = scenario_match.group(1).strip() if scenario_match else None
    variant_header = variant_match.group(1).strip() if variant_match else None
    fewshot_base = fewshot_base_match.group(1).strip() if fewshot_base_match else None

    scenario = scenario_header or path_meta["scenario_from_path"]
    variant = variant_header or path_meta["variant_from_path"]
    mode = path_meta["mode_from_name"]

    inferred_condition, modality, label_coverage = infer_experiment_condition(
        mode=mode,
        variant=variant,
        fewshot_base=fewshot_base,
    )

    trial_pattern = re.compile(
        r"=+Screenshot\s+([^\s]+)\s+\(repeat\s+(\d+)/(\d+)\)=+\s*",
        flags=re.IGNORECASE
    )
    matches = list(trial_pattern.finditer(text))
    records: List[TrialRecord] = []

    if not matches:
        print(f"[WARN] No trial blocks found in {path}")
        return records

    for idx, m in enumerate(matches):
        image_id = m.group(1).strip()
        repeat_idx = int(m.group(2))
        repeat_total = int(m.group(3))

        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        raw_json_text = extract_top_level_json_block(block)
        parsed_json = try_parse_json_block(raw_json_text) if raw_json_text else None

        grade_match = re.search(r"GRADE:\s*(correct|incorrect)", block, flags=re.IGNORECASE)
        grade_raw = grade_match.group(1).lower() if grade_match else None
        correct = 1 if grade_raw == "correct" else 0 if grade_raw == "incorrect" else None

        expected_match = re.search(r"^expected:.*$", block, flags=re.MULTILINE)
        parsed_match = re.search(r"^parsed:.*$", block, flags=re.MULTILINE)

        # First try explicit expected: line from log
        expected_action = parse_expected_action_from_line(expected_match.group(0)) if expected_match else None

        # If missing, auto-fill from ground_truth.txt
        if expected_action is None and scenario is not None:
            expected_action = gt_lookup.get((scenario, image_id))

        # Predicted action
        pred_action = canonicalize_action_dict(parsed_json)
        if parsed_match:
            pred_action_line = parse_parsed_action_from_line(parsed_match.group(0))
            if pred_action_line is not None:
                pred_action = pred_action_line

        parse_success = 1 if isinstance(parsed_json, dict) else 0
        observation_text = parsed_json.get("observation") if isinstance(parsed_json, dict) else None
        reason_text = parsed_json.get("reason") if isinstance(parsed_json, dict) else None

        error_type = None
        if correct == 0:
            error_type = classify_error(expected_action, pred_action, parsed_json)

        records.append(
            TrialRecord(
                source_file=path.name,
                source_path=str(path),
                mode=mode,
                scenario=scenario,
                variant=variant,
                fewshot_base=fewshot_base,
                inferred_condition=inferred_condition,
                modality=modality,
                label_coverage=label_coverage,
                image_id=image_id,
                repeat_idx=repeat_idx,
                repeat_total=repeat_total,
                grade_raw=grade_raw,
                correct=correct,
                expected_action=expected_action,
                pred_action=pred_action,
                parse_success=parse_success,
                observation_text=observation_text,
                reason_text=reason_text,
                raw_json_text=raw_json_text,
                error_type=error_type,
            )
        )

    return records


def records_to_df(records: List[TrialRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in records])


# ============================================================
# Analysis
# ============================================================

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    low, high = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return low, high


def summarize_accuracy(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
          .agg(
              n=("correct", "size"),
              n_correct=("correct", "sum"),
              parse_rate=("parse_success", "mean"),
          )
          .reset_index()
    )
    grouped["accuracy"] = grouped["n_correct"] / grouped["n"]

    ci_lows = []
    ci_highs = []
    for _, row in grouped.iterrows():
        low, high = wilson_ci(int(row["n_correct"]), int(row["n"]))
        ci_lows.append(low)
        ci_highs.append(high)

    grouped["ci_low"] = ci_lows
    grouped["ci_high"] = ci_highs
    grouped["accuracy_pct"] = grouped["accuracy"] * 100
    grouped["ci_low_pct"] = grouped["ci_low"] * 100
    grouped["ci_high_pct"] = grouped["ci_high"] * 100
    grouped["parse_rate_pct"] = grouped["parse_rate"] * 100
    return grouped


def summarize_action_distribution(df: pd.DataFrame, by_cols: List[str]) -> pd.DataFrame:
    sub = df.dropna(subset=["expected_action", "pred_action"]).copy()
    if sub.empty:
        return pd.DataFrame()

    out = (
        sub.groupby(by_cols + ["expected_action", "pred_action"], dropna=False)
           .size()
           .reset_index(name="count")
    )
    return out


def two_group_fisher(df: pd.DataFrame, cond_a: str, cond_b: str, scenario: Optional[str] = None) -> Dict[str, Any]:
    sub = df.copy()
    if scenario is not None:
        sub = sub[sub["scenario"] == scenario]

    a = sub[sub["inferred_condition"] == cond_a]
    b = sub[sub["inferred_condition"] == cond_b]

    a_correct = int(a["correct"].sum())
    a_total = int(a["correct"].count())
    b_correct = int(b["correct"].sum())
    b_total = int(b["correct"].count())

    table = np.array([
        [a_correct, a_total - a_correct],
        [b_correct, b_total - b_correct]
    ])

    odds_ratio, p_value = fisher_exact(table)
    rd = (a_correct / a_total) - (b_correct / b_total) if a_total and b_total else np.nan

    return {
        "contrast": f"{cond_a} vs {cond_b}",
        "scenario": scenario if scenario is not None else "overall",
        "a_n": a_total,
        "a_correct": a_correct,
        "a_acc": a_correct / a_total if a_total else np.nan,
        "b_n": b_total,
        "b_correct": b_correct,
        "b_acc": b_correct / b_total if b_total else np.nan,
        "risk_diff": rd,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def planned_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("G0", "G2B"),
        ("G0", "G4B"),
        ("G2B", "G4B"),
        ("G1A", "G3A"),
        ("G1B", "G3B"),
        ("G2A", "G4A"),
    ]
    scenarios = [None, "plain_grounds", "besides_edges"]

    existing = set(df["inferred_condition"].dropna().unique())
    rows: List[Dict[str, Any]] = []

    for a, b in comparisons:
        if a not in existing or b not in existing:
            continue
        for scenario in scenarios:
            rows.append(two_group_fisher(df, a, b, scenario))

    out = pd.DataFrame(rows)
    if len(out) > 0:
        reject, p_adj, _, _ = multipletests(out["p_value"], method="holm")
        out["p_adj_holm"] = p_adj
        out["reject_holm_0.05"] = reject
    return out


def fit_gee(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    sub = sub.dropna(subset=["correct", "inferred_condition", "scenario", "image_id"])

    keep = {"G0", "G1A", "G1B", "G2A", "G2B", "G3A", "G3B", "G4A", "G4B"}
    sub = sub[sub["inferred_condition"].isin(keep)].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["cluster_id"] = sub["scenario"].astype(str) + "::" + sub["image_id"].astype(str)

    try:
        model = GEE.from_formula(
            "correct ~ C(inferred_condition) + C(scenario)",
            groups="cluster_id",
            cov_struct=Exchangeable(),
            family=Binomial(),
            data=sub,
        )
        result = model.fit()

        summary_table = result.summary2().tables[1].copy()
        summary_table = summary_table.reset_index().rename(columns={"index": "term"})
        if "Coef." in summary_table.columns:
            summary_table["odds_ratio"] = np.exp(summary_table["Coef."])
        if "Coef." in summary_table.columns and "Std.Err." in summary_table.columns:
            summary_table["or_ci_low"] = np.exp(summary_table["Coef."] - 1.96 * summary_table["Std.Err."])
            summary_table["or_ci_high"] = np.exp(summary_table["Coef."] + 1.96 * summary_table["Std.Err."])
        return summary_table
    except Exception as e:
        print(f"[WARN] GEE fit failed: {e}")
        return pd.DataFrame()


def error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["correct"] == 0].copy()
    if sub.empty:
        return pd.DataFrame()

    out = (
        sub.groupby(["inferred_condition", "scenario", "error_type"], dropna=False)
           .size()
           .reset_index(name="count")
    )
    totals = (
        sub.groupby(["inferred_condition", "scenario"], dropna=False)
           .size()
           .reset_index(name="total_errors")
    )
    out = out.merge(totals, on=["inferred_condition", "scenario"], how="left")
    out["error_pct_within_group"] = out["count"] / out["total_errors"] * 100
    return out


def save_latex(df: pd.DataFrame, path: Path, cols: List[str]) -> None:
    if df.empty:
        return
    path.write_text(df[cols].to_latex(index=False, escape=False), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VLM train_output logs directly.")
    parser.add_argument("--root", required=True, help="Path to train_cases/train_output or test_cases/test_output")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis outputs")
    parser.add_argument("--scenario", default=None, help="Optional scenario filter, e.g. besides_edges")
    parser.add_argument("--mode", choices=["textual", "multimodal"], default=None, help="Optional mode filter")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(root.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under: {root}")

    cases_root = find_cases_root_from_output_root(root)
    gt_lookup = build_ground_truth_lookup(cases_root)

    all_records: List[TrialRecord] = []
    for txt_path in txt_files:
        records = parse_single_txt(txt_path, root, gt_lookup)
        all_records.extend(records)

    if not all_records:
        raise RuntimeError("No trial records parsed.")

    df = records_to_df(all_records)

    if args.scenario:
        df = df[df["scenario"] == args.scenario].copy()
    if args.mode:
        df = df[df["mode"] == args.mode].copy()

    # save raw dataset
    df.to_csv(output_dir / "trial_level_results.csv", index=False, encoding="utf-8-sig")

    # summaries
    core_df = df.dropna(subset=["inferred_condition"]).copy()

    summary_condition = summarize_accuracy(core_df, ["inferred_condition"])
    summary_condition_scenario = summarize_accuracy(core_df, ["inferred_condition", "scenario"])
    summary_condition_image = summarize_accuracy(core_df, ["inferred_condition", "scenario", "image_id"])
    summary_variant = summarize_accuracy(df, ["mode", "variant", "fewshot_base", "scenario"])

    summary_condition.to_csv(output_dir / "summary_by_condition.csv", index=False, encoding="utf-8-sig")
    summary_condition_scenario.to_csv(output_dir / "summary_by_condition_scenario.csv", index=False, encoding="utf-8-sig")
    summary_condition_image.to_csv(output_dir / "summary_by_condition_scenario_image.csv", index=False, encoding="utf-8-sig")
    summary_variant.to_csv(output_dir / "summary_by_variant.csv", index=False, encoding="utf-8-sig")

    # action-level summaries (now expected_action should be auto-filled)
    action_dist = summarize_action_distribution(df, ["inferred_condition", "scenario"])
    action_dist.to_csv(output_dir / "action_distribution.csv", index=False, encoding="utf-8-sig")

    # comparisons
    comparisons = planned_comparisons(core_df)
    comparisons.to_csv(output_dir / "planned_comparisons.csv", index=False, encoding="utf-8-sig")

    # GEE
    gee_df = fit_gee(core_df)
    gee_df.to_csv(output_dir / "gee_model_results.csv", index=False, encoding="utf-8-sig")

    # error analysis
    errors_df = error_breakdown(core_df)
    errors_df.to_csv(output_dir / "error_breakdown.csv", index=False, encoding="utf-8-sig")

    # latex
    save_latex(
        summary_condition_scenario,
        output_dir / "summary_by_condition_scenario.tex",
        ["inferred_condition", "scenario", "n_correct", "n", "accuracy_pct", "ci_low_pct", "ci_high_pct", "parse_rate_pct"]
    )
    if not comparisons.empty:
        save_latex(
            comparisons,
            output_dir / "planned_comparisons.tex",
            ["contrast", "scenario", "a_acc", "b_acc", "risk_diff", "odds_ratio", "p_value", "p_adj_holm"]
        )

    # quick fill-rate diagnostics
    expected_fill_rate = df["expected_action"].notna().mean() * 100 if len(df) else 0
    pred_fill_rate = df["pred_action"].notna().mean() * 100 if len(df) else 0

    print("Done.")
    print(f"Parsed trials: {len(df)}")
    print(f"Expected-action fill rate: {expected_fill_rate:.1f}%")
    print(f"Pred-action fill rate: {pred_fill_rate:.1f}%")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
