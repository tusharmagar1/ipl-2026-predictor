"""
STEP 1 — Load & Explore the IPL Dataset
========================================
This script loads matches.csv and gives you a full picture of the data:
- Shape, columns, and data types
- IPL winners by season
- Most successful teams of all time
- Toss analysis

Run: python step1_explore.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_matches

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 1: Loading IPL Dataset")
print("=" * 60)

matches = load_matches()

print(f"\n✅ Loaded {len(matches)} matches")
print(f"   Seasons covered : {sorted(matches['season'].unique())}")
print(f"   Total teams      : {len(set(matches['team1'].unique()) | set(matches['team2'].unique()))}")
print(f"\nColumn names:\n  {list(matches.columns)}")
print(f"\nFirst 3 rows:")
print(matches.head(3).to_string())


# ─────────────────────────────────────────────
# 2. IPL CHAMPIONS BY SEASON
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  IPL Champions by Season (last match = Final)")
print("─" * 60)

season_winners = {}
for season in sorted(matches["season"].unique()):
    season_matches = matches[matches["season"] == season]
    final = season_matches.iloc[-1]  # Last match of the season = Final
    season_winners[season] = final["winner"]

winners_df = pd.DataFrame(list(season_winners.items()), columns=["Season", "Champion"])
print(winners_df.to_string(index=False))


# ─────────────────────────────────────────────
# 3. MOST TITLES
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Total IPL Titles Won")
print("─" * 60)

title_counts = winners_df["Champion"].value_counts()
for team, count in title_counts.items():
    bar = "🏆" * count
    print(f"  {team:<35} {bar} ({count})")


# ─────────────────────────────────────────────
# 4. TOSS ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Toss Analysis")
print("─" * 60)

toss_match_winner = matches[matches["toss_winner"] == matches["winner"]]
toss_effect_pct = len(toss_match_winner) / len(matches) * 100
print(f"  Toss winner also won the match: {toss_effect_pct:.1f}% of the time")

toss_decision_win = matches.groupby("toss_decision")["winner"].count()
print(f"\n  Matches by toss decision:")
print(f"    Bat first  : {matches[matches['toss_decision'] == 'bat'].shape[0]} matches")
print(f"    Field first: {matches[matches['toss_decision'] == 'field'].shape[0]} matches")


# ─────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("IPL Historical Analysis (2008–2025)", fontsize=15, fontweight="bold")

# Plot 1: Titles won
title_counts.plot(kind="bar", ax=axes[0], color="gold", edgecolor="black")
axes[0].set_title("IPL Titles Won by Team")
axes[0].set_xlabel("Team")
axes[0].set_ylabel("Titles")
axes[0].tick_params(axis="x", rotation=45)

# Plot 2: Wins per team (all-time)
all_wins = matches["winner"].value_counts()
all_wins.head(10).plot(kind="barh", ax=axes[1], color="steelblue", edgecolor="black")
axes[1].set_title("All-Time Match Wins (Top 10 Teams)")
axes[1].set_xlabel("Wins")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("step1_analysis.png", dpi=150, bbox_inches="tight")
print("\n✅ Chart saved: step1_analysis.png")
plt.show()

print("\n✅ Step 1 complete! Run: python step2_features.py\n")
