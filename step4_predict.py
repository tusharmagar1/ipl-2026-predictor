"""
STEP 4 — Predict the IPL 2026 Winner  (v2 — XGBoost + H2H)
=============================================================
Uses the upgraded XGBoost model + Head-to-Head feature to
simulate the full IPL 2026 season and playoffs.

Run: python step4_predict.py
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from step2_features import compute_team_features, compute_h2h_win_rate


IPL_2026_TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

FEATURE_COLS = [
    "t1_win_rate", "t1_recent_win_rate", "t1_titles",
    "t1_toss_win_rate", "t1_toss2win_rate", "t1_seasons",
    "t2_win_rate", "t2_recent_win_rate", "t2_titles",
    "t2_toss_win_rate", "t2_toss2win_rate", "t2_seasons",
    "toss_winner_is_t1",
    "t1_h2h_win_rate",    # ★ NEW
]


def predict_win_prob(model, f1, f2, h2h_rate=0.5, toss_is_t1=0.5):
    """
    Returns probability that team1 beats team2.

    f1, f2      : team feature Series/dict from compute_team_features()
    h2h_rate    : t1's historical win rate vs t2 (0.5 = unknown)
    toss_is_t1  : 0=T2 won toss, 1=T1 won toss, 0.5=neutral
    """
    row = [[
        f1["win_rate"], f1["recent_win_rate"], f1["titles"],
        f1["toss_win_rate"], f1["toss_to_win_rate"], f1["seasons_played"],
        f2["win_rate"], f2["recent_win_rate"], f2["titles"],
        f2["toss_win_rate"], f2["toss_to_win_rate"], f2["seasons_played"],
        toss_is_t1,
        h2h_rate,
    ]]
    return model.predict_proba(row)[0][1]


def simulate_league_stage(model, features, teams, all_matches):
    """
    Round-robin: every team vs every other team.
    Uses historical H2H rates from all_matches.
    Returns standings DataFrame + matchup detail DataFrame.
    """
    points   = {t: 0.0 for t in teams}
    matchups = []

    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            if i >= j:
                continue
            if t1 not in features.index or t2 not in features.index:
                print(f"   ⚠️  Missing features: {t1} or {t2} — skipping")
                continue

            f1, f2 = features.loc[t1], features.loc[t2]

            # H2H rates
            h2h_t1 = compute_h2h_win_rate(all_matches, t1, t2)
            h2h_t2 = compute_h2h_win_rate(all_matches, t2, t1)

            # Two legs
            p1 = predict_win_prob(model, f1, f2, h2h_rate=h2h_t1)   # t1 vs t2
            p2 = predict_win_prob(model, f2, f1, h2h_rate=h2h_t2)   # t2 vs t1

            # Expected points (2 per win)
            points[t1] += p1 * 2 + (1 - p2) * 2
            points[t2] += (1 - p1) * 2 + p2 * 2

            fav  = t1 if p1 >= 0.5 else t2
            prob = max(p1, 1-p1)
            matchups.append({
                "Match"          : f"{t1}  vs  {t2}",
                "Predicted Winner": fav,
                "Win Probability" : f"{prob*100:.1f}%",
                "H2H Meetings"   : len(all_matches[
                    ((all_matches["team1"]==t1)&(all_matches["team2"]==t2))|
                    ((all_matches["team1"]==t2)&(all_matches["team2"]==t1))
                ]),
            })

    standings = pd.DataFrame({
        "Team"            : list(points.keys()),
        "Expected Points" : [round(v, 1) for v in points.values()],
    }).sort_values("Expected Points", ascending=False).reset_index(drop=True)
    standings.index += 1
    return standings, pd.DataFrame(matchups)


def simulate_playoffs(model, features, top4, all_matches):
    """
    IPL playoff format:
      Q1 → 1st vs 2nd   (winner → Final directly)
      Elim → 3rd vs 4th
      Q2  → Q1 loser vs Elim winner
      Final
    """
    def play(t1, t2, label):
        f1, f2   = features.loc[t1], features.loc[t2]
        h2h      = compute_h2h_win_rate(all_matches, t1, t2)
        p        = predict_win_prob(model, f1, f2, h2h_rate=h2h)
        winner   = t1 if p >= 0.5 else t2
        loser    = t2 if p >= 0.5 else t1
        print(f"   {label:<20} {t1} ({p*100:.0f}%)  vs  {t2} ({(1-p)*100:.0f}%)  →  ✅ {winner}")
        return winner, loser, p

    print("\n  📍 QUALIFIER 1 (1st vs 2nd):")
    q1_win, q1_lose, _   = play(top4[0], top4[1], "Q1")

    print("\n  📍 ELIMINATOR (3rd vs 4th):")
    el_win, _, _         = play(top4[2], top4[3], "Eliminator")

    print("\n  📍 QUALIFIER 2 (Q1 loser vs Eliminator winner):")
    q2_win, _, _         = play(q1_lose, el_win, "Q2")

    print("\n  🏆 FINAL:")
    champion, runner_up, final_p = play(q1_win, q2_win, "Final")

    return champion, runner_up, final_p


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 4: Predicting IPL 2026 Winner  (v2 — XGBoost + H2H)")
    print("=" * 60)

    # ── Load model bundle ──
    try:
        with open("ipl_model.pkl", "rb") as f:
            bundle = pickle.load(f)

        # Support both old (model only) and new (dict bundle) formats
        if isinstance(bundle, dict):
            model    = bundle["model"]
            acc      = bundle.get("accuracy", "?")
            cv_mean  = bundle.get("cv_mean", "?")
            print(f"\n✅ XGBoost model loaded  |  Test acc: {acc:.2%}  |  CV acc: {cv_mean:.2%}")
        else:
            model = bundle
            print("\n✅ Model loaded (legacy format)")
    except FileNotFoundError:
        print("\n❌ ipl_model.pkl not found. Run step3_train.py first!")
        exit(1)

    # ── Data & features ──
    from data_loader import load_matches
    matches  = load_matches()
    features = compute_team_features(matches)

    valid    = [t for t in IPL_2026_TEAMS if t in features.index]
    missing  = [t for t in IPL_2026_TEAMS if t not in features.index]
    if missing:
        print(f"\n⚠️  Teams not in dataset (renamed?): {missing}")

    feat2026 = features.loc[valid]

    # ── League Stage ──
    print(f"\n🏏 Simulating IPL 2026 League Stage ({len(valid)} teams)...")
    standings, matchups = simulate_league_stage(model, feat2026, valid, matches)

    print("\n📊 IPL 2026 Predicted Points Table:")
    print("  " + "─" * 52)
    for rank, row in standings.iterrows():
        q = " ← QUALIFIES" if rank <= 4 else ""
        print(f"  #{rank:<4} {row['Team']:<35} {row['Expected Points']:.1f} pts{q}")
    print("  " + "─" * 52)

    # ── Playoffs ──
    top4 = standings["Team"].head(4).tolist()
    print(f"\n🏟️  Playoff Teams: {top4}")
    print("\n" + "─" * 60)
    print("  PLAYOFF SIMULATION")
    print("─" * 60)

    champion, runner_up, final_p = simulate_playoffs(model, feat2026, top4, matches)

    print("\n" + "=" * 60)
    print(f"  🏆 PREDICTED IPL 2026 CHAMPION: {champion.upper()} 🏆")
    print(f"  🥈 Runner-Up: {runner_up}")
    print(f"  Final Win Probability: {final_p*100:.0f}% for {champion}")
    print("=" * 60)

    # ── Visualization ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("🏏 IPL 2026 Winner Prediction  (XGBoost v2)", fontsize=16, fontweight="bold")

    colors = ["gold" if t == champion else ("steelblue" if i < 4 else "lightgray")
              for i, t in enumerate(standings["Team"])]
    axes[0].barh(standings["Team"].iloc[::-1], standings["Expected Points"].iloc[::-1],
                 color=colors[::-1], edgecolor="black", linewidth=0.5)
    axes[0].set_title("Predicted Points Table", fontweight="bold")
    axes[0].set_xlabel("Expected Points")
    if len(standings) > 4:
        cutoff = (standings["Expected Points"].iloc[3] + standings["Expected Points"].iloc[4]) / 2
        axes[0].axvline(cutoff, color="red", linestyle="--", linewidth=1.5, label="Playoff cutoff")
        axes[0].legend()

    probs      = [final_p * 100, (1 - final_p) * 100]
    bar_colors = ["gold", "silver"]
    bars = axes[1].bar([champion, runner_up], probs, color=bar_colors, edgecolor="black", width=0.4)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("🏆 Final — Win Probability", fontweight="bold")
    axes[1].set_ylabel("Win Probability (%)")
    for bar, prob in zip(bars, probs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{prob:.0f}%", ha="center", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig("step4_prediction.png", dpi=150, bbox_inches="tight")
    print("\n✅ Chart saved: step4_prediction.png")
    plt.show()
