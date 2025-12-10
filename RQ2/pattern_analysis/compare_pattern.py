import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Input CSVs
gpt_csv = "./llm_data/final_data/human_perf_prs_with_gpt_analysis_full_catalog.csv"
gemini_csv = "./llm_data/final_data/human_perf_prs_with_gemini_analysis_full_catalog.csv"

# gpt_csv = "./llm_data/final_data/ai_perf_prs_with_gpt_analysis_full_catalog.csv"
# gemini_csv = "./llm_data/final_data/ai_perf_prs_with_gemini_analysis_full_catalog.csv"

# Output CSV for mismatches
unmatched_csv = "./llm_data/final_data/human_perf_prs_pattern_mismatches_gpt_gemini_full_catalog.csv"
# unmatched_csv = "./llm_data/final_data/ai_perf_prs_pattern_mismatches_gpt_gemini_full_catalog.csv"

# # Input CSVs
# # gpt_csv = "./human_perf_prs_with_gpt_analysis_new_catalog.csv"
# # gemini_csv = "./human_perf_prs_with_gemini_analysis_new_catalog.csv"

# gpt_csv = "./ai_perf_prs_with_gpt_analysis_new_catalog.csv"
# gemini_csv = "./ai_perf_prs_with_gemini_analysis_new_catalog.csv"

# # Output CSV for mismatches
# # unmatched_csv = "./human_perf_prs_pattern_mismatches_gpt_gemini_new_catalog.csv"
# unmatched_csv = "./ai_perf_prs_pattern_mismatches_gpt_gemini_new_catalog.csv"

unmatched_ids_urls_csv = unmatched_csv.replace(".csv", "_ids_urls.csv")

key = "id"

df_gpt = pd.read_csv(gpt_csv)
df_gemini = pd.read_csv(gemini_csv)

# Normalize missing/whitespace values so NaN vs '' don't register as mismatches
for col in ["optimization_pattern", "optimization_subpattern"]:
    df_gpt[col] = df_gpt[col].fillna("").str.strip()
    df_gemini[col] = df_gemini[col].fillna("").str.strip()

# Keep comparison columns for quick stats
merged_compare = df_gpt[[key, "optimization_pattern", "optimization_subpattern"]].merge(
    df_gemini[[key, "optimization_pattern", "optimization_subpattern"]],
    on=key,
    suffixes=("_gpt", "_gemini")
)

merged_compare["pattern_match"] = merged_compare["optimization_pattern_gpt"] == merged_compare["optimization_pattern_gemini"]
merged_compare["subpattern_match"] = merged_compare["optimization_subpattern_gpt"] == merged_compare["optimization_subpattern_gemini"]

summary = {
    "total": len(merged_compare),
    "pattern_match_count": int(merged_compare["pattern_match"].sum()),
    "subpattern_match_count": int(merged_compare["subpattern_match"].sum()),
    "pattern_kappa": cohen_kappa_score(
        merged_compare["optimization_pattern_gpt"], merged_compare["optimization_pattern_gemini"]
    ),
    "subpattern_kappa": cohen_kappa_score(
        merged_compare["optimization_subpattern_gpt"], merged_compare["optimization_subpattern_gemini"]
    ),
}

print("Comparison Summary:")
print(f"Total PRs Analyzed: {summary['total']}")
print(f"Pattern Matches: {summary['pattern_match_count']} ({(summary['pattern_match_count'] / summary['total']) * 100:.2f}%)")
print(f"Subpattern Matches: {summary['subpattern_match_count']} ({(summary['subpattern_match_count'] / summary['total']) * 100:.2f}%)")
print(f"Pattern Cohen's Kappa: {summary['pattern_kappa']:.4f}")
print(f"Subpattern Cohen's Kappa: {summary['subpattern_kappa']:.4f}")

# Build full merge to export mismatches with all data from both CSVs
full_merged = df_gpt.merge(df_gemini, on=key, suffixes=("_gpt", "_gemini"), how="inner")
not_matched = full_merged[
    (full_merged["optimization_pattern_gpt"] != full_merged["optimization_pattern_gemini"])
    | (full_merged["optimization_subpattern_gpt"] != full_merged["optimization_subpattern_gemini"])
].copy()

print(f"Saving mismatched rows to {unmatched_csv} ({len(not_matched)} rows)...")
not_matched.to_csv(unmatched_csv, index=False)

html_url_col = None
if "html_url" in not_matched.columns:
    html_url_col = "html_url"
elif "html_url_gpt" in not_matched.columns:
    html_url_col = "html_url_gpt"
elif "html_url_gemini" in not_matched.columns:
    html_url_col = "html_url_gemini"
else:
    raise ValueError("No html_url column found in mismatched data")

not_matched_urls = not_matched[[key, html_url_col]].copy()
if html_url_col != "html_url":
    not_matched_urls.rename(columns={html_url_col: "html_url"}, inplace=True)

print(f"Saving mismatch ids/html_urls to {unmatched_ids_urls_csv} ({len(not_matched_urls)} rows)...")
not_matched_urls.to_csv(unmatched_ids_urls_csv, index=False)


# Comparison Summary:
# Total PRs Analyzed: 326
# Pattern Matches: 188 (57.67%)
# Subpattern Matches: 150 (46.01%)
# Saving mismatched rows to ./ai_perf_prs_pattern_mismatches_gpt_gemini.csv (176 rows)...

# Total PRs Analyzed: 83
# Pattern Matches: 36 (43.37%)
# Subpattern Matches: 30 (36.14%)
# Saving mismatched rows to ./human_perf_prs_pattern_mismatches_gpt_gemini.csv (53 rows)...

# Total PRs Analyzed: 326
# Pattern Matches: 205 (62.88%)
# Subpattern Matches: 176 (53.99%)
# Saving mismatched rows to ./ai_perf_prs_pattern_mismatches_gpt_gemini_full_catalog.csv (153 rows)...
# Pattern Cohen's Kappa: 0.5221
# Subpattern Cohen's Kappa: 0.4774

# Total PRs Analyzed: 83
# Pattern Matches: 46 (55.42%)
# Subpattern Matches: 35 (42.17%)
# Saving mismatched rows to ./human_perf_prs_pattern_mismatches_gpt_gemini_full_catalog.csv (48 rows)...
# Pattern Cohen's Kappa: 0.4246
# Subpattern Cohen's Kappa: 0.3512

# Total PRs Analyzed: 137
# Pattern Matches: 61 (44.53%)
# Subpattern Matches: 38 (27.74%)
# Saving mismatched rows to ./ai_perf_prs_pattern_mismatches_gpt_gemini_new_catalog.csv (102 rows)...
# Pattern Cohen's Kappa: 0.3362
# Subpattern Cohen's Kappa: 0.2305

# Total PRs Analyzed: 45
# Pattern Matches: 25 (55.56%)
# Subpattern Matches: 15 (33.33%)
# Saving mismatched rows to ./human_perf_prs_pattern_mismatches_gpt_gemini_new_catalog.csv (31 rows)...
# Pattern Cohen's Kappa: 0.4675
# Subpattern Cohen's Kappa: 0.2707