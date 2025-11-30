import pandas as pd

# Input CSVs
# gpt_csv = "./human_perf_prs_with_gpt_analysis_full_catalog.csv"
# qwen_csv = "./human_perf_prs_with_gemini_analysis_full_catalog.csv"

gpt_csv = "./ai_perf_prs_with_gpt_analysis_full_catalog.csv"
qwen_csv = "./ai_perf_prs_with_gemini_analysis_full_catalog.csv"

# Output CSV for mismatches
unmatched_csv = "./ai_perf_prs_pattern_mismatches_gpt_gemini_full_catalog.csv"

key = "id"

df_gpt = pd.read_csv(gpt_csv)
df_qwen = pd.read_csv(qwen_csv)

# Normalize missing/whitespace values so NaN vs '' don't register as mismatches
for col in ["optimization_pattern", "optimization_subpattern"]:
    df_gpt[col] = df_gpt[col].fillna("").str.strip()
    df_qwen[col] = df_qwen[col].fillna("").str.strip()

# Keep comparison columns for quick stats
merged_compare = df_gpt[[key, "optimization_pattern", "optimization_subpattern"]].merge(
    df_qwen[[key, "optimization_pattern", "optimization_subpattern"]],
    on=key,
    suffixes=("_gpt", "_qwen")
)

merged_compare["pattern_match"] = merged_compare["optimization_pattern_gpt"] == merged_compare["optimization_pattern_qwen"]
merged_compare["subpattern_match"] = merged_compare["optimization_subpattern_gpt"] == merged_compare["optimization_subpattern_qwen"]

summary = {
    "total": len(merged_compare),
    "pattern_match_count": int(merged_compare["pattern_match"].sum()),
    "subpattern_match_count": int(merged_compare["subpattern_match"].sum()),
}

print("Comparison Summary:")
print(f"Total PRs Analyzed: {summary['total']}")
print(f"Pattern Matches: {summary['pattern_match_count']} ({(summary['pattern_match_count'] / summary['total']) * 100:.2f}%)")
print(f"Subpattern Matches: {summary['subpattern_match_count']} ({(summary['subpattern_match_count'] / summary['total']) * 100:.2f}%)")

# Build full merge to export mismatches with all data from both CSVs
full_merged = df_gpt.merge(df_qwen, on=key, suffixes=("_gpt", "_qwen"), how="inner")
not_matched = full_merged[
    (full_merged["optimization_pattern_gpt"] != full_merged["optimization_pattern_qwen"])
    | (full_merged["optimization_subpattern_gpt"] != full_merged["optimization_subpattern_qwen"])
].copy()

print(f"Saving mismatched rows to {unmatched_csv} ({len(not_matched)} rows)...")
not_matched.to_csv(unmatched_csv, index=False)


# Comparison Summary:
# Total PRs Analyzed: 326
# Pattern Matches: 188 (57.67%)
# Subpattern Matches: 150 (46.01%)
# Saving mismatched rows to ./ai_perf_prs_pattern_mismatches_gpt_gemini.csv (176 rows)...

# Total PRs Analyzed: 326
# Pattern Matches: 205 (62.88%)
# Subpattern Matches: 176 (53.99%)
# Saving mismatched rows to ./ai_perf_prs_pattern_mismatches_gpt_gemini_full_catalog.csv (153 rows)...

# Total PRs Analyzed: 83
# Pattern Matches: 36 (43.37%)
# Subpattern Matches: 30 (36.14%)
# Saving mismatched rows to ./human_perf_prs_pattern_mismatches_gpt_gemini.csv (53 rows)...

# Total PRs Analyzed: 83
# Pattern Matches: 46 (55.42%)
# Subpattern Matches: 35 (42.17%)
# Saving mismatched rows to ./human_perf_prs_pattern_mismatches_gpt_gemini_full_catalog.csv (48 rows)...