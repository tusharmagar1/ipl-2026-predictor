"""
data_loader.py — Auto-detects and loads the IPL dataset
=========================================================
Handles BOTH formats:
  1. Ball-by-ball data (your dataset) → converts to match-level
  2. Match-level data (matches.csv style) → loads directly
"""

import pandas as pd
import os

POSSIBLE_FILES = [
    "IPL.csv", "IPL.xlsx",
    "ipl.csv", "ipl.xlsx",
    "matches.csv", "matches.xlsx",
    "IPL_2008_2025.csv", "ipl_2008_2025.csv",
]


def load_matches():
    found_file = None
    for fname in POSSIBLE_FILES:
        if os.path.exists(fname):
            found_file = fname
            break

    if not found_file:
        print("\n❌ Dataset file not found!")
        print(f"   Supported names: {POSSIBLE_FILES}")
        exit(1)

    print(f"   📂 Loading file: {found_file}")

    if found_file.endswith(".xlsx"):
        df = pd.read_excel(found_file)
    else:
        df = pd.read_csv(found_file, low_memory=False)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "winner" in df.columns and "team1" in df.columns:
        print("   ✅ Match-level format detected")
        df = df.dropna(subset=["winner"])
        return df

    elif "match_id" in df.columns and "batting_team" in df.columns:
        print("   🔄 Ball-by-ball format detected → converting to match-level...")
        return convert_ball_by_ball_to_matches(df)

    else:
        print(f"\n❌ Unrecognized format. Columns: {list(df.columns)}")
        exit(1)


def convert_ball_by_ball_to_matches(df):
    print("   ⏳ Aggregating data (may take a moment)...")

    match_info = df.drop_duplicates(subset=["match_id"]).copy()

    innings1 = df[df["innings"] == 1].drop_duplicates(subset=["match_id"])[
        ["match_id", "batting_team", "bowling_team"]
    ]
    innings1 = innings1.rename(columns={"batting_team": "team1", "bowling_team": "team2"})
    match_info = match_info.merge(innings1, on="match_id", how="left")

    # Find winner column - look for one containing team names
    winner_col = None
    team_names = set(df["batting_team"].dropna().unique()) | set(df["bowling_team"].dropna().unique())

    for col in ["match_won_by", "win_outcome", "winner", "winning_team"]:
        if col in match_info.columns:
            vals = match_info[col].dropna().astype(str).unique()
            overlap = sum(1 for v in vals if v.strip() in team_names)
            if overlap > 2:
                winner_col = col
                break

    if winner_col is None:
        # Fallback: pick any col that has team names in it
        for col in match_info.columns:
            vals = match_info[col].dropna().astype(str).unique()
            overlap = sum(1 for v in vals if v.strip() in team_names)
            if overlap > 2:
                winner_col = col
                print(f"   ℹ️  Using '{col}' as winner column")
                break

    if winner_col is None:
        print("   ❌ Could not determine winner column automatically.")
        print(f"   Available columns: {list(match_info.columns)}")
        exit(1)

    print(f"   ✅ Winner column identified: '{winner_col}'")

    rows = []
    for _, row in match_info.iterrows():
        season = row.get("season") or row.get("year")
        try:
            season = int(str(season).split("/")[0].strip())
        except:
            continue

        team1 = row.get("team1")
        team2 = row.get("team2")
        if pd.isna(team1) or pd.isna(team2):
            continue

        winner = row.get(winner_col)
        if pd.isna(winner) or str(winner).strip().lower() in ["no result", "tie", "draw", "nan", ""]:
            continue

        rows.append({
            "match_id"      : row.get("match_id"),
            "season"        : season,
            "team1"         : str(team1).strip(),
            "team2"         : str(team2).strip(),
            "winner"        : str(winner).strip(),
            "toss_winner"   : str(row.get("toss_winner", "")).strip(),
            "toss_decision" : str(row.get("toss_decision", "")).strip(),
            "venue"         : row.get("venue", ""),
            "city"          : row.get("city", ""),
            "stage"         : str(row.get("stage", "")).lower(),
        })

    matches = pd.DataFrame(rows).sort_values(["season", "match_id"]).reset_index(drop=True)
    print(f"   ✅ Converted: {len(matches)} matches | Seasons: {sorted(matches['season'].unique())}")
    return matches