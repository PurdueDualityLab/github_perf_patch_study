import os
import re
import tempfile
from typing import Dict, List, Optional

import pandas as pd
from github import Github
from dotenv import load_dotenv
import huggingface_hub
import lizard

# ============================================================
# Configuration
# ============================================================

METRICS = ["Total nloc", "Avg.NLOC", "AvgCCN", "Fun Cnt", "Avg.token"]

# File extensions we will analyze
SUPPORTED_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".fs",
    ".fsi",
    ".java",
    ".go",
    ".cs",
    ".cpp",
    ".c",
    ".rs",
    ".kt",
    ".rb",
    ".php",
    ".zig",
    ".lua",
    ".rkt",
}

load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


# ============================================================
# Helpers: dataset & PR selection
# ============================================================


def load_human_perf_prs() -> pd.DataFrame:
    """
    Load Human-PRs and filter to performance PRs using the human_pr_task_type table.
    This uses the AIDev parquet files via the HuggingFace 'hf://' URI scheme.
    """
    human_pr = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pull_request.parquet")
    human_task = pd.read_parquet(
        "hf://datasets/hao-li/AIDev/human_pr_task_type.parquet"
    )

    perf = (
        human_pr.merge(human_task[["id", "type"]], on="id", how="inner")
        .query("type == 'perf'")
        .copy()
    )

    missing_cols = [c for c in ["id", "html_url"] if c not in perf.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected columns in human_pull_request: {missing_cols}"
        )

    return perf[["id", "html_url"]]


def parse_github_url(html_url: str):
    """
    Parse a GitHub PR URL like:
      https://github.com/owner/repo/pull/123
    and return (owner, repo, number).
    """
    m = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", html_url)
    if not m:
        raise ValueError(f"Cannot parse GitHub PR url: {html_url}")
    owner, repo, number_str = m.groups()
    return owner, repo, int(number_str)


# ============================================================
# Helpers: lizard / CCN (Python API version)
# ============================================================


def compute_metrics_from_lizard_result(
    result: lizard.FileInformation,
) -> Dict[str, float]:
    """
    Build a metrics dictionary from a lizard FileInformation object.
      - Total nloc: total lines of code in the file
      - Avg.NLOC: mean NLOC across functions in the file
      - AvgCCN: mean cyclomatic complexity across functions
      - Fun Cnt: number of functions
      - Avg.token: mean token_count across functions
    """
    total_nloc = float(result.nloc or 0)
    funs = result.function_list or []
    fun_cnt = float(len(funs))

    if not funs:
        return {
            "Total nloc": total_nloc,
            "Avg.NLOC": float("nan"),
            "AvgCCN": float("nan"),
            "Fun Cnt": 0.0,
            "Avg.token": float("nan"),
        }

    sum_nloc = 0.0
    sum_ccn = 0.0
    sum_tokens = 0.0

    for f in funs:
        fn_nloc = getattr(f, "nloc", 0) or 0
        fn_ccn = getattr(f, "cyclomatic_complexity", 0) or 0
        fn_tokens = getattr(f, "token_count", 0) or 0

        sum_nloc += float(fn_nloc)
        sum_ccn += float(fn_ccn)
        sum_tokens += float(fn_tokens)

    avg_nloc = sum_nloc / fun_cnt
    avg_ccn = sum_ccn / fun_cnt
    avg_tokens = sum_tokens / fun_cnt

    return {
        "Total nloc": total_nloc,
        "Avg.NLOC": avg_nloc,
        "AvgCCN": avg_ccn,
        "Fun Cnt": fun_cnt,
        "Avg.token": avg_tokens,
    }


def compute_metrics_for_file(
    repo,
    path: str,
    ref: str,
    pr_id: int,
    label: str,
) -> Optional[Dict[str, float]]:
    """
    Fetch a file at a given commit (ref) from GitHub, run lizard (Python API),
    and return the parsed metrics dict. Returns None on failure.

    This function:
    - Safely decodes the source using UTF-8 with replacement.
    - Uses lizard.analyze_file.analyze_source_code to compute metrics.
    """
    try:
        content_file = repo.get_contents(path, ref=ref)
    except Exception as e:
        print(
            f"[WARN]   Failed to fetch {path} at {ref[:7]} "
            f"(PR {pr_id}, {label}): {e}"
        )
        return None

    raw_bytes = content_file.decoded_content

    try:
        code_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        print(
            f"[WARN]   Unicode decode error for {path} at {ref[:7]} "
            f"(PR {pr_id}, {label}): {e}. Using replacement characters."
        )
        code_text = raw_bytes.decode("utf-8", errors="replace")

    filename_hint = os.path.basename(path)
    try:
        analysis_result = lizard.analyze_file.analyze_source_code(
            filename_hint, code_text
        )
    except Exception as e:
        print(
            f"[WARN]   lizard (Python API) failed for {path} at {ref[:7]} "
            f"(PR {pr_id}, {label}): {e}"
        )
        return None

    metrics = compute_metrics_from_lizard_result(analysis_result)
    return metrics


def aggregate_metrics(file_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate a list of metric dicts by taking the mean for each metric in METRICS.
    """
    agg: Dict[str, float] = {}
    for m in METRICS:
        vals = [fm[m] for fm in file_metrics if fm is not None and m in fm]
        if not vals:
            agg[m] = float("nan")
        else:
            agg[m] = sum(vals) / len(vals)
    return agg


# ============================================================
# Main analysis
# ============================================================


def compute_ccn_deltas_for_prs(
    gh: Github,
    max_prs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Iterate over human perf PRs, compute CCN metrics before/after,
    and return a DataFrame with deltas per PR.
    """
    perf_prs = load_human_perf_prs()

    if max_prs is not None:
        perf_prs = perf_prs.head(max_prs)

    rows = []
    total_prs = len(perf_prs)
    processed_prs = 0
    skipped_prs = 0

    for _, row in perf_prs.iterrows():
        pr_id = row["id"]
        html_url = row["html_url"]

        try:
            owner, repo_name, number = parse_github_url(html_url)
            repo = gh.get_repo(f"{owner}/{repo_name}")
            pr = repo.get_pull(number)
        except Exception as e:
            print(f"[WARN] Skipping PR {pr_id} ({html_url}): {e}")
            continue

        base_sha = pr.base.sha
        head_sha = pr.head.sha

        print(
            f"\n[INFO] PR {pr_id}: {owner}/{repo_name}#{number} "
            f"| base={base_sha[:7]} head={head_sha[:7]}"
        )

        base_metrics_list: List[Dict[str, float]] = []
        head_metrics_list: List[Dict[str, float]] = []

        try:
            files = list(pr.get_files())
        except Exception as e:
            print(f"[WARN] Error fetching files for PR {pr_id}: {e}")
            continue

        if not files:
            print(f"[WARN] No files list returned for PR {pr_id}, skipping.")
            continue

        for f in files:
            path = f.filename
            ext = os.path.splitext(path)[1].lower()

            if ext not in SUPPORTED_EXTS:
                print(f"[WARN] Unsupported extension {ext} for {path}; skipping file.")
                continue

            try:
                base_m = compute_metrics_for_file(
                    repo,
                    path,
                    base_sha,
                    pr_id=pr_id,
                    label="base",
                )
            except Exception as e:
                print(
                    f"[WARN]   Skipping file {path} in PR {pr_id} "
                    f"when computing base metrics: {e}"
                )
                base_m = None

            try:
                head_m = compute_metrics_for_file(
                    repo,
                    path,
                    head_sha,
                    pr_id=pr_id,
                    label="head",
                )
            except Exception as e:
                print(
                    f"[WARN]   Skipping file {path} in PR {pr_id} "
                    f"when computing head metrics: {e}"
                )
                head_m = None

            if base_m is not None:
                base_metrics_list.append(base_m)
            if head_m is not None:
                head_metrics_list.append(head_m)

        if not base_metrics_list or not head_metrics_list:
            print(f"[WARN] No metrics for PR {pr_id}, skipping this PR.")
            skipped_prs += 1
            continue

        agg_base = aggregate_metrics(base_metrics_list)
        agg_head = aggregate_metrics(head_metrics_list)

        for metric in METRICS:
            orig = agg_base[metric]
            opt = agg_head[metric]

            if (not pd.isna(orig)) and (not pd.isna(opt)) and orig != 0:
                delta = opt - orig
                delta_pct = delta / orig * 100.0
            else:
                delta = float("nan")
                delta_pct = float("nan")

            rows.append(
                {
                    "pr_id": pr_id,
                    "html_url": html_url,
                    "metric": metric,
                    "original": orig,
                    "optimized": opt,
                    "delta": delta,
                    "delta_pct": delta_pct,
                }
            )
        processed_prs += 1

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] Result DataFrame is empty (no PRs produced metrics).")
    print(
        f"[INFO] PR summary: total={total_prs}, processed={processed_prs}, skipped={skipped_prs}"
    )
    return df


def main():
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "Please set GITHUB_TOKEN environment variable with a valid GitHub token."
        )
    gh = Github(GITHUB_TOKEN)

    result = compute_ccn_deltas_for_prs(gh=gh)

    out_path = "human_perf_pr_ccn_deltas.csv"
    result.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
