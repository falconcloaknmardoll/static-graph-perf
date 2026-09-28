"""O16: apply the 8d pre-registration's §9 decision table to the frozen 8d results.

phase8d_evaluate.py's `p5_pass` tests §3's threshold only. This reads the frozen
experiments/phase8d_results.json (unchanged) and records, beside it, which §9 rows match.

  python scripts/phase8d_p5_rule_check.py
"""
import json

r = json.load(open("experiments/phase8d_results.json"))
pc = r["primary_vs_channel"]
threshold_met = r["conv_oracle_mean"] <= 1.05                      # §3 P5, and §9 row 1's first half
sig_beats_channel = pc["delta"] < 0 and pc["p_t"] < 0.05            # §8: p<0.05, oracle vs channel, CONV
rows = {
    "row1_P5_confirmed (<=1.05 and significantly beats channel)": threshold_met and sig_beats_channel,
    "row2_partial (beats channel, well above 1.05)": sig_beats_channel and not threshold_met,
    "row3_oracle_approx_channel (no significant further gain)": not sig_beats_channel,
    "row4_confound_adjacent (OTHER gain >= CONV gain)": r["other_gain"] >= r["conv_gain"],
}
out = {"source": "experiments/phase8d_results.json (frozen, not modified)",
       "conv_oracle_mean": r["conv_oracle_mean"], "oracle_vs_channel": pc,
       "s3_threshold_met": threshold_met, "s8_significant_vs_channel": sig_beats_channel,
       "s9_rows_matched": rows,
       "note": ("p5_pass in phase8d_results.json = s3_threshold_met only. Under §9, P5 is confirmed only "
                "if row 1 matches. Row 3's wording ('oracle ~ channel') assumed a small effect; here the "
                "effect (-0.0725) is not small but the test is underpowered (phase8d_power_addendum.md). "
                "How the paper reads row 3 is an author decision logged under DEFENSIBILITY O16.")}
json.dump(out, open("experiments/phase8d_p5_rule_check.json", "w"), indent=1)
for k, v in rows.items():
    print(f"{'MATCH' if v else '  -  '}  {k}")
print(f"s3 threshold met: {threshold_met}; significant vs channel: {sig_beats_channel} (p={pc['p_t']:.4f})")
