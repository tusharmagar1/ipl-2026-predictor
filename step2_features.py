"""
STEP 2 — Feature Engineering  (v2 — adds Head-to-Head feature)
================================================================
Turns raw match history into numbers an ML model can learn from.

Features per team-pair:
  - Win rate (overall)
  - Recent win rate (last 3 seasons)
  - IPL titles won
  - Toss win rate
  - Win rate after winning toss
  - Seasons played
  ★ NEW → Head-to-Head win rate (t1 vs t2 historically)

Run: python step2_features.py
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Helper: IPL champion for each season
# ─────────────────────────────────────────────
def get_season_winners(matches):
    """Returns {season: champion_team}"""
    winners = {}
    for season in sorted(matches["season"].unique()):
        sm = matches[matches["season"] == season]
        final = sm[sm.get("stage", pd.Series(dtype=str)).str.contains("final", na=False)]
        winners[season] = final.iloc[-1]["winner"] if len(final) > 0 else sm.iloc[-1]["winner"]
    return winners


# ─────────────────────────────────────────────
# ★ NEW: Head-to-Head helper
# ─────────────────────────────────────────────
def compute_h2h_win_rate(df, t1, t2, default=0.5):
    """
    Given a DataFrame `df` (already filtered to past data),
    returns the fraction of meetings t1 won against t2.

    Falls back to `default` (0.5 = coin flip) when they've
    never met — avoids introducing bias for new matchups.
    """
    h2h = df[
        ((df["team1"] == t1) & (df["team2"] == t2)) |
        ((df["team1"] == t2) & (df["team2"] == t1))
    ]
    if len(h2h) == 0:
        return default
    return len(h2h[h2h["winner"] == t1]) / len(h2h)


# ─────────────────────────────────────────────
# Team-level features (no H2H here — that's matchup-level)
# ─────────────────────────────────────────────
def compute_team_features(matches, cutoff_season=None):
    """
    Compute a feature vector for every team.

    cutoff_season: if set, only use matches BEFORE this season
                   (prevents data leakage into training).
    """
    df = matches[matches["season"] < cutoff_season].copy() if cutoff_season else matches.copy()
    if df.empty:
        return pd.DataFrame()

    season_winners = get_season_winners(df)
    teams = set(df["team1"].unique()) | set(df["team2"].unique())
    rows = []

    for team in teams:
        tm    = df[(df["team1"] == team) | (df["team2"] == team)]
        total = len(tm)
        if total == 0:
            continue

        wins     = len(tm[tm["winner"] == team])
        win_rate = wins / total

        last3      = sorted(df["season"].unique())[-3:]
        rtm        = df[df["season"].isin(last3)]
        rtm        = rtm[(rtm["team1"] == team) | (rtm["team2"] == team)]
        recent_wr  = len(rtm[rtm["winner"] == team]) / len(rtm) if len(rtm) > 0 else 0

        titles     = sum(1 for w in season_winners.values() if w == team)
        toss_wins  = len(tm[tm["toss_winner"] == team])
        won_toss   = tm[tm["toss_winner"] == team]
        t2w        = len(won_toss[won_toss["winner"] == team]) / len(won_toss) if len(won_toss) > 0 else 0

        rows.append({
            "team"            : team,
            "win_rate"        : round(win_rate, 4),
            "recent_win_rate" : round(recent_wr, 4),
            "titles"          : titles,
            "toss_win_rate"   : round(toss_wins / total, 4),
            "toss_to_win_rate": round(t2w, 4),
            "total_matches"   : total,
            "seasons_played"  : tm["season"].nunique(),
        })

    return pd.DataFrame(rows).set_index("team")


# ─────────────────────────────────────────────
# Match-level training dataset
# ─────────────────────────────────────────────
def build_training_data(matches, start_from_season=2012):
    """
    For every match from start_from_season onwards, create a row:
      - Team 1 features  (computed on PAST data only)
      - Team 2 features  (computed on PAST data only)
      - ★ H2H win rate   (computed on PAST data only)
      - Toss info
      - Label: 1 if Team 1 wins, 0 otherwise
    """
    rows = []

    for season in sorted(matches["season"].unique()):
        if season < start_from_season:
            continue

        past     = matches[matches["season"] < season].copy()
        features = compute_team_features(matches, cutoff_season=season)
        if features.empty or past.empty:
            continue

        for _, match in matches[matches["season"] == season].iterrows():
            t1 = match["team1"]
            t2 = match["team2"]

            if t1 not in features.index or t2 not in features.index:
                continue

            f1 = features.loc[t1]
            f2 = features.loc[t2]

            # ★ H2H feature — uses only data BEFORE this season
            h2h_rate = compute_h2h_win_rate(past, t1, t2)

            toss_is_t1 = 1 if match["toss_winner"] == t1 else 0

            rows.append({
                # Team 1 features
                "t1_win_rate"        : f1["win_rate"],
                "t1_recent_win_rate" : f1["recent_win_rate"],
                "t1_titles"          : f1["titles"],
                "t1_toss_win_rate"   : f1["toss_win_rate"],
                "t1_toss2win_rate"   : f1["toss_to_win_rate"],
                "t1_seasons"         : f1["seasons_played"],
                # Team 2 features
                "t2_win_rate"        : f2["win_rate"],
                "t2_recent_win_rate" : f2["recent_win_rate"],
                "t2_titles"          : f2["titles"],
                "t2_toss_win_rate"   : f2["toss_win_rate"],
                "t2_toss2win_rate"   : f2["toss_to_win_rate"],
                "t2_seasons"         : f2["seasons_played"],
                # Toss
                "toss_winner_is_t1"  : toss_is_t1,
                # ★ Head-to-Head
                "t1_h2h_win_rate"    : round(h2h_rate, 4),
                # Label
                "team1_wins"         : 1 if match["winner"] == t1 else 0,
                # Reference columns (not fed into model)
                "team1"              : t1,
                "team2"              : t2,
                "season"             : season,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 2: Feature Engineering  (v2 — with H2H)")
    print("=" * 60)

    from data_loader import load_matches
    matches = load_matches()

    print("\n📊 Team Features (all data):")
    all_features = compute_team_features(matches)
    print(all_features.sort_values("win_rate", ascending=False).round(3).to_string())

    print("\n🔨 Building match-level training dataset...")
    training_data = build_training_data(matches)
    training_data.to_csv("training_data.csv", index=False)

    print(f"\n✅ Training dataset: {len(training_data)} rows × {len(training_data.columns)} columns")
    print(f"   Saved to: training_data.csv")

    print(f"\n   Label balance:")
    print(f"     Team 1 wins: {training_data['team1_wins'].sum()} ({training_data['team1_wins'].mean()*100:.1f}%)")
    print(f"     Team 2 wins: {(1-training_data['team1_wins']).sum()} ({(1-training_data['team1_wins'].mean())*100:.1f}%)")

    print(f"\n   ★ H2H win rate sample (first 5 rows):")
    print(training_data[["team1", "team2", "t1_h2h_win_rate", "team1_wins"]].head())

    print("\n✅ Step 2 complete! Run: python step3_train.py\n")
