"""
app.py — IPL 2026 Winner Predictor  (v2 — XGBoost + H2H + Dark Theme + Live Schedule)
========================================================================================
Run locally:  streamlit run app.py
Deploy:       Push to GitHub → connect to Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import requests
import base64
import mimetypes
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IPL 2026 Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ★ DARK IPL THEME — CSS Injection
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary   : #080c14;
    --bg-card      : #0f1623;
    --bg-sidebar   : #0a0f1a;
    --accent       : #FF6B1A;
    --accent-gold  : #FFB800;
    --accent-glow  : rgba(255, 107, 26, 0.25);
    --text-primary : #EAEAEA;
    --text-muted   : #7A8BA0;
    --border       : rgba(255, 107, 26, 0.2);
    --success      : #22c55e;
    --danger       : #ef4444;
}

/* ── Global background ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Cards / containers ── */
[data-testid="stVerticalBlock"] > div > div {
    border-radius: 12px;
}

/* ── Metric widgets ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--accent-gold) !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #FF9A4A) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px var(--accent-glow) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Radio ── */
.stRadio > div { gap: 8px; }
.stRadio label { color: var(--text-primary) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-gold)) !important;
    border-radius: 999px !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 999px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Section headers ── */
h1, h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.5px;
}
h1 { color: var(--accent-gold) !important; font-size: 2.2rem !important; }
h2 { color: var(--text-primary) !important; }
h3 { color: var(--accent) !important; }

/* ── Info / Success / Warning boxes ── */
.stAlert { border-radius: 10px !important; }
.stSuccess { border-left: 4px solid var(--success) !important; }
.stInfo    { border-left: 4px solid var(--accent) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Custom classes ── */
.team-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.team-card:hover { border-color: var(--accent); }

.champion-banner {
    background: linear-gradient(135deg, #1a1200, #2d2000);
    border: 2px solid var(--accent-gold);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 40px rgba(255, 184, 0, 0.2);
}

.match-row {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 18px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.stat-pill {
    background: rgba(255,107,26,0.12);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.85rem;
    color: var(--accent);
    display: inline-block;
}

.prob-bar-container {
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
PREDICTOR_LOGO_URL = "https://documents.iplt20.com//ipl/assets/images/ipl-logo-new-old.png"
LOGO_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "logos")
TEAM_INFO = {
    "Mumbai Indians"              : {"emoji": "MI", "color": "#004BA0", "short": "MI", "logo_file": "mi.png", "logo": "https://documents.iplt20.com/ipl/MI/Logos/Logooutline/MIoutline.png"},
    "Chennai Super Kings"         : {"emoji": "CSK", "color": "#F9CD05", "short": "CSK", "logo_file": "csk.png", "logo": "https://documents.iplt20.com/ipl/CSK/logos/Logooutline/CSKoutline.png"},
    "Royal Challengers Bengaluru" : {"emoji": "RCB", "color": "#EC1C24", "short": "RCB", "logo_file": "rcb.png", "logo": "https://documents.iplt20.com/ipl/RCB/Logos/Logooutline/RCBoutline.png"},
    "Royal Challengers Bangalore" : {"emoji": "RCB", "color": "#EC1C24", "short": "RCB", "logo_file": "rcb.png", "logo": "https://documents.iplt20.com/ipl/RCB/Logos/Logooutline/RCBoutline.png"},
    "Kolkata Knight Riders"       : {"emoji": "KKR", "color": "#3A225D", "short": "KKR", "logo_file": "kkr.png", "logo": "https://documents.iplt20.com/ipl/KKR/Logos/Logooutline/KKRoutline.png"},
    "Delhi Capitals"              : {"emoji": "DC", "color": "#00008B", "short": "DC", "logo_file": "dc.png", "logo": "https://documents.iplt20.com/ipl/DC/Logos/LogoOutline/DCoutline.png"},
    "Delhi Daredevils"            : {"emoji": "DD", "color": "#00008B", "short": "DD", "logo_file": "dc.png", "logo": "https://documents.iplt20.com/ipl/DC/Logos/LogoOutline/DCoutline.png"},
    "Rajasthan Royals"            : {"emoji": "RR", "color": "#EA1A85", "short": "RR", "logo_file": "rr.png", "logo": "https://documents.iplt20.com/ipl/RR/Logos/RR_Logo.png"},
    "Sunrisers Hyderabad"         : {"emoji": "SRH", "color": "#F7A721", "short": "SRH", "logo_file": "srh.png", "logo": "https://documents.iplt20.com/ipl/SRH/Logos/Logooutline/SRHoutline.png"},
    "Punjab Kings"                : {"emoji": "PBKS", "color": "#ED1B24", "short": "PBKS", "logo_file": "pbks.png", "logo": "https://documents.iplt20.com/ipl/PBKS/Logos/Logooutline/PBKSoutline.png"},
    "Kings XI Punjab"             : {"emoji": "KXIP", "color": "#ED1B24", "short": "KXIP", "logo_file": "pbks.png", "logo": "https://documents.iplt20.com/ipl/PBKS/Logos/Logooutline/PBKSoutline.png"},
    "Lucknow Super Giants"        : {"emoji": "LSG", "color": "#A2D4F5", "short": "LSG", "logo_file": "lsg.png", "logo": "https://documents.iplt20.com/ipl/LSG/Logos/Logooutline/LSGoutline.png"},
    "Gujarat Titans"              : {"emoji": "GT", "color": "#1C4C8C", "short": "GT", "logo_file": "gt.png", "logo": "https://documents.iplt20.com/ipl/GT/Logos/Logooutline/GToutline.png"},
    "Deccan Chargers"             : {"emoji": "DC", "color": "#555555", "short": "DC"},
    "Pune Warriors"               : {"emoji": "PW", "color": "#8B4513", "short": "PW"},
    "Rising Pune Supergiant"      : {"emoji": "RPS", "color": "#6A0DAD", "short": "RPS"},
    "Rising Pune Supergiants"     : {"emoji": "RPS", "color": "#6A0DAD", "short": "RPS"},
    "Gujarat Lions"               : {"emoji": "GL", "color": "#FF6600", "short": "GL"},
}

IPL_2026_TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru",
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
    "Sunrisers Hyderabad", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans",
]

FEATURE_COLS = [
    "t1_win_rate", "t1_recent_win_rate", "t1_titles",
    "t1_toss_win_rate", "t1_toss2win_rate", "t1_seasons",
    "t2_win_rate", "t2_recent_win_rate", "t2_titles",
    "t2_toss_win_rate", "t2_toss2win_rate", "t2_seasons",
    "toss_winner_is_t1", "t1_h2h_win_rate",
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_team_info(team):
    for key in TEAM_INFO:
        if key.lower() in team.lower() or team.lower() in key.lower():
            return TEAM_INFO[key]
    return {"emoji": "TEAM", "color": "#FF6B1A", "short": team[:3].upper()}


def _file_to_data_uri(path):
    if not os.path.exists(path):
        return None
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def predictor_logo_src():
    local = os.path.join(LOGO_ASSETS_DIR, "predictor.png")
    return _file_to_data_uri(local) or PREDICTOR_LOGO_URL


def predictor_logo_html(size=58):
    src = predictor_logo_src()
    if src:
        return f"<img src='{src}' alt='IPL logo' style='height:{size}px; width:auto;' />"
    return f"<span style='font-size:{max(24, int(size * 0.55))}px; font-weight:700; color:#FFB800;'>IPL</span>"


def team_logo_html(team, size=44):
    info = get_team_info(team)
    local_src = None
    if info.get("logo_file"):
        local_src = _file_to_data_uri(os.path.join(LOGO_ASSETS_DIR, info["logo_file"]))
    logo = local_src or info.get("logo")

    if logo:
        return (
            "<span style='display:inline-flex; align-items:center;'>"
            f"<img src='{logo}' alt='{team} logo' "
            f"style='height:{size}px; width:auto; max-width:{int(size * 1.8)}px; object-fit:contain; vertical-align:middle;' "
            "onerror=\"this.style.display='none'; this.nextElementSibling.style.display='inline';\"/>"
            f"<span style='display:none; font-size:{max(16, int(size * 0.55))}px; font-weight:700; color:{info['color']};'>{info['short']}</span>"
            "</span>"
        )

    return f"<span style='font-size:{max(16, int(size * 0.55))}px; font-weight:700; color:{info['color']}; vertical-align:middle;'>{info['short']}</span>"


def h2h_win_rate(all_matches, t1, t2):
    h2h = all_matches[
        ((all_matches["team1"] == t1) & (all_matches["team2"] == t2)) |
        ((all_matches["team1"] == t2) & (all_matches["team2"] == t1))
    ]
    if len(h2h) == 0:
        return 0.5
    return len(h2h[h2h["winner"] == t1]) / len(h2h)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    for fname in ["IPL.csv", "ipl.csv", "matches.csv", "IPL_2008_2025.csv"]:
        if os.path.exists(fname):
            df = pd.read_csv(fname, low_memory=False)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            if "winner" in df.columns and "team1" in df.columns:
                return df.dropna(subset=["winner"])
            if "match_id" in df.columns and "batting_team" in df.columns:
                return _convert_bbb(df)
    return None


def _convert_bbb(df):
    match_info = df.drop_duplicates(subset=["match_id"]).copy()
    inn1 = df[df["innings"] == 1].drop_duplicates(subset=["match_id"])[
        ["match_id", "batting_team", "bowling_team"]
    ].rename(columns={"batting_team": "team1", "bowling_team": "team2"})
    match_info = match_info.merge(inn1, on="match_id", how="left")
    team_names = set(df["batting_team"].dropna().unique()) | set(df["bowling_team"].dropna().unique())
    winner_col = None
    for col in ["match_won_by", "win_outcome", "winner", "winning_team"]:
        if col in match_info.columns:
            vals = match_info[col].dropna().astype(str).unique()
            if sum(1 for v in vals if v.strip() in team_names) > 2:
                winner_col = col
                break
    if winner_col is None:
        return None
    rows = []
    for _, row in match_info.iterrows():
        season = row.get("season") or row.get("year")
        try: season = int(str(season).split("/")[0].strip())
        except: continue
        t1, t2, winner = row.get("team1"), row.get("team2"), row.get(winner_col)
        if pd.isna(t1) or pd.isna(t2) or pd.isna(winner): continue
        if str(winner).strip().lower() in ["no result", "tie", "draw", "nan", ""]: continue
        rows.append({
            "match_id": row.get("match_id"), "season": season,
            "team1": str(t1).strip(), "team2": str(t2).strip(),
            "winner": str(winner).strip(),
            "toss_winner": str(row.get("toss_winner", "")).strip(),
            "toss_decision": str(row.get("toss_decision", "")).strip(),
            "venue": str(row.get("venue", "")), "city": str(row.get("city", "")),
            "stage": str(row.get("stage", "")).lower(),
        })
    return pd.DataFrame(rows).sort_values(["season", "match_id"]).reset_index(drop=True)


def get_season_winners(matches):
    winners = {}
    for season in sorted(matches["season"].unique()):
        sm = matches[matches["season"] == season]
        final = sm[sm.get("stage", pd.Series(dtype=str)).str.contains("final", na=False)]
        winners[season] = final.iloc[-1]["winner"] if len(final) > 0 else sm.iloc[-1]["winner"]
    return winners


@st.cache_data
def compute_features(_matches):
    matches = _matches
    season_winners = get_season_winners(matches)
    teams = set(matches["team1"].unique()) | set(matches["team2"].unique())
    rows = []
    for team in teams:
        tm    = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        total = len(tm)
        if total == 0: continue
        wins  = len(tm[tm["winner"] == team])
        last3 = sorted(matches["season"].unique())[-3:]
        rtm   = matches[matches["season"].isin(last3)]
        rtm   = rtm[(rtm["team1"] == team) | (rtm["team2"] == team)]
        rwr   = len(rtm[rtm["winner"] == team]) / len(rtm) if len(rtm) > 0 else 0
        titles = sum(1 for w in season_winners.values() if w == team)
        tw     = len(tm[tm["toss_winner"] == team])
        wt     = tm[tm["toss_winner"] == team]
        t2w    = len(wt[wt["winner"] == team]) / len(wt) if len(wt) > 0 else 0
        rows.append({
            "team": team, "win_rate": wins/total, "recent_win_rate": rwr,
            "titles": titles, "toss_win_rate": tw/total,
            "toss_to_win_rate": t2w, "total_matches": total,
            "seasons_played": tm["season"].nunique(),
        })
    return pd.DataFrame(rows).set_index("team")


@st.cache_data
def train_model_cached(_matches):
    """Train XGBoost with H2H feature. Cached so it runs once per session."""
    matches = _matches
    rows    = []

    def _sw(df):
        w = {}
        for s in sorted(df["season"].unique()):
            sm = df[df["season"] == s]
            fn = sm[sm.get("stage", pd.Series(dtype=str)).str.contains("final", na=False)]
            w[s] = fn.iloc[-1]["winner"] if len(fn) > 0 else sm.iloc[-1]["winner"]
        return w

    for season in sorted(matches["season"].unique()):
        if season < 2012:
            continue
        past = matches[matches["season"] < season]
        if len(past) == 0:
            continue
        pw    = _sw(past)
        teams = set(past["team1"].unique()) | set(past["team2"].unique())
        feat  = {}
        for team in teams:
            tm    = past[(past["team1"] == team) | (past["team2"] == team)]
            total = len(tm)
            if total == 0: continue
            wins  = len(tm[tm["winner"] == team])
            last3 = sorted(past["season"].unique())[-3:]
            rtm   = past[past["season"].isin(last3)]
            rtm   = rtm[(rtm["team1"] == team) | (rtm["team2"] == team)]
            rwr   = len(rtm[rtm["winner"] == team]) / len(rtm) if len(rtm) > 0 else 0
            titles = sum(1 for w in pw.values() if w == team)
            tw     = len(tm[tm["toss_winner"] == team])
            wt     = tm[tm["toss_winner"] == team]
            t2w    = len(wt[wt["winner"] == team]) / len(wt) if len(wt) > 0 else 0
            feat[team] = {
                "win_rate": wins/total, "recent_win_rate": rwr,
                "titles": titles, "toss_win_rate": tw/total,
                "toss_to_win_rate": t2w, "seasons_played": tm["season"].nunique(),
            }
        for _, m in matches[matches["season"] == season].iterrows():
            t1, t2 = m["team1"], m["team2"]
            if t1 not in feat or t2 not in feat: continue
            f1, f2 = feat[t1], feat[t2]
            # H2H rate using only past data
            h2h    = past[
                ((past["team1"]==t1)&(past["team2"]==t2))|
                ((past["team1"]==t2)&(past["team2"]==t1))
            ]
            h2h_rate = len(h2h[h2h["winner"]==t1]) / len(h2h) if len(h2h) > 0 else 0.5
            rows.append([
                f1["win_rate"], f1["recent_win_rate"], f1["titles"],
                f1["toss_win_rate"], f1["toss_to_win_rate"], f1["seasons_played"],
                f2["win_rate"], f2["recent_win_rate"], f2["titles"],
                f2["toss_win_rate"], f2["toss_to_win_rate"], f2["seasons_played"],
                1 if m["toss_winner"] == t1 else 0,
                h2h_rate,
                1 if m["winner"] == t1 else 0,
            ])

    data = pd.DataFrame(rows, columns=FEATURE_COLS + ["label"])
    X, y = data[FEATURE_COLS], data["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Try to load pre-trained model first (faster)
    if os.path.exists("ipl_model.pkl"):
        try:
            bundle = pickle.load(open("ipl_model.pkl", "rb"))
            if isinstance(bundle, dict) and "model" in bundle:
                clf = bundle["model"]
                acc = bundle.get("accuracy", accuracy_score(y_te, clf.predict(X_te)))
                return clf, round(acc, 4)
        except Exception:
            pass

    # Train XGBoost with quick tuning (5 iter for app speed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42, n_jobs=-1),
        param_distributions={
            "n_estimators": [200, 300], "max_depth": [4, 5],
            "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0],
        },
        n_iter=5, cv=cv, scoring="accuracy", random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_tr, y_tr)
    clf = search.best_estimator_
    acc = accuracy_score(y_te, clf.predict(X_te))
    return clf, round(acc, 4)


def win_prob(model, all_matches, f1_dict, f2_dict, t1_name, t2_name, toss=0.5):
    h2h  = h2h_win_rate(all_matches, t1_name, t2_name)
    row  = [[
        f1_dict["win_rate"], f1_dict["recent_win_rate"], f1_dict["titles"],
        f1_dict["toss_win_rate"], f1_dict["toss_to_win_rate"], f1_dict["seasons_played"],
        f2_dict["win_rate"], f2_dict["recent_win_rate"], f2_dict["titles"],
        f2_dict["toss_win_rate"], f2_dict["toss_to_win_rate"], f2_dict["seasons_played"],
        toss, h2h,
    ]]
    return model.predict_proba(row)[0][1]


def simulate_tournament(model, features, teams, all_matches):
    pts = {t: 0.0 for t in teams}
    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            if i >= j: continue
            if t1 not in features.index or t2 not in features.index: continue
            f1, f2 = features.loc[t1], features.loc[t2]
            p      = win_prob(model, all_matches, f1, f2, t1, t2)
            pts[t1] += p * 2
            pts[t2] += (1 - p) * 2
    return pd.DataFrame({
        "Team": list(pts.keys()),
        "Expected Points": [round(v, 1) for v in pts.values()]
    }).sort_values("Expected Points", ascending=False).reset_index(drop=True).rename(lambda x: x+1)


# ─────────────────────────────────────────────
# ★ NEW: Live IPL 2026 Schedule Fetcher
# ─────────────────────────────────────────────
@st.cache_data(ttl=900)  # Refresh every 15 minutes
def fetch_ipl_2026_schedule():
    """
    Tries to fetch IPL 2026 match schedule from ESPN Cricinfo's public API.
    Returns list of match dicts or None if unavailable.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    # ESPN Cricinfo public API — leagueId=6 is IPL
    endpoints = [
        "https://hs-consumer-api.espncricinfo.com/v1/pages/matches/current?lang=en&leagueId=6",
        "https://hs-consumer-api.espncricinfo.com/v1/pages/matches/upcoming?lang=en&leagueId=6",
        "https://hs-consumer-api.espncricinfo.com/v1/pages/matches/recent?lang=en&leagueId=6",
    ]
    all_matches = []
    for url in endpoints:
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code != 200:
                continue
            data = resp.json()
            content = data.get("content", []) or data.get("matches", []) or []
            for item in content:
                match = item.get("match", item)
                teams = match.get("teams", [])
                if len(teams) < 2:
                    continue
                t1    = teams[0].get("team", {}).get("longName", teams[0].get("team", {}).get("name", ""))
                t2    = teams[1].get("team", {}).get("longName", teams[1].get("team", {}).get("name", ""))
                status = match.get("statusText", "") or match.get("status", "")
                date   = match.get("startDate", {})
                if isinstance(date, dict):
                    date = date.get("iso", "")
                winner_id = match.get("winnerTeamId")
                winner = ""
                if winner_id:
                    for t in teams:
                        if t.get("team", {}).get("id") == winner_id:
                            winner = t.get("team", {}).get("longName", "")
                            break
                venue = (match.get("ground") or {}).get("longName", "")
                all_matches.append({
                    "team1"  : t1,
                    "team2"  : t2,
                    "status" : status,
                    "date"   : str(date)[:10] if date else "",
                    "winner" : winner,
                    "venue"  : venue,
                    "series" : match.get("series", [{}])[0].get("longName", "") if match.get("series") else "",
                })
        except Exception:
            continue

    # Filter IPL 2026 only and deduplicate
    ipl = [m for m in all_matches if "2026" in m.get("series", "") or "Indian Premier League 2026" in m.get("series", "")]
    if not ipl:
        ipl = all_matches  # Fallback: show whatever we got

    seen = set()
    deduped = []
    for m in ipl:
        key = (m["team1"], m["team2"], m["date"])
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped if deduped else None


def normalize_team_name(name, known_teams):
    """Match ESPN team name to our dataset team names."""
    name = name.strip().lower()
    aliases = {
        "rcb": "Royal Challengers Bengaluru",
        "royal challengers bangalore": "Royal Challengers Bengaluru",
        "royal challengers bengaluru": "Royal Challengers Bengaluru",
        "mi": "Mumbai Indians",
        "csk": "Chennai Super Kings",
        "kkr": "Kolkata Knight Riders",
        "dc": "Delhi Capitals",
        "rr": "Rajasthan Royals",
        "srh": "Sunrisers Hyderabad",
        "pbks": "Punjab Kings",
        "lsg": "Lucknow Super Giants",
        "gt": "Gujarat Titans",
        "kings xi punjab": "Punjab Kings",
        "delhi daredevils": "Delhi Capitals",
    }
    if name in aliases:
        return aliases[name]
    for team in known_teams:
        if team.lower() in name or name in team.lower():
            return team
        # Word overlap
        words = set(name.split())
        team_words = set(team.lower().split())
        if len(words & team_words) >= 2:
            return team
    return None


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        {predictor_logo_html(58)}
        <div style='font-family:Rajdhani,sans-serif; font-size:1.4rem; font-weight:700; color:#FFB800; letter-spacing:1px;'>
            IPL 2026
        </div>
        <div style='color:#7A8BA0; font-size:0.8rem;'>Winner Predictor</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "🏠 Home & Predict",
        "📊 2026 Standings",
        "📅 Live Schedule",
        "📈 History",
        "ℹ️ About",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='color:#7A8BA0; font-size:0.78rem;'>Upload Dataset</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    if uploaded:
        with open("IPL.csv", "wb") as f:
            f.write(uploaded.read())
        st.success("✅ Uploaded!")
        st.cache_data.clear()

# ── Load Data ──
matches = load_data()

if matches is None:
    st.title("🏏 IPL 2026 Winner Predictor")
    st.error("⚠️ Dataset not found! Upload your `IPL.csv` file using the sidebar ←")
    st.info("**Get the dataset:**\n1. Go to Kaggle → search 'IPL Dataset 2008-2025'\n2. Download CSV\n3. Upload in sidebar")
    st.stop()

with st.spinner("🤖 Training XGBoost model..."):
    features      = compute_features(matches)
    model, accuracy = train_model_cached(matches)
    season_winners  = get_season_winners(matches)

all_teams = sorted(set(matches["team1"].unique()) | set(matches["team2"].unique()))


# ══════════════════════════════════════════════════════════════
# PAGE 1: HOME & PREDICT
# ══════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown(f"""
    <div style='margin-bottom:8px;'>
        <span style='font-family:Rajdhani,sans-serif; font-size:2.4rem; font-weight:700; color:#FFB800; letter-spacing:1px;'>
{predictor_logo_html(56)} IPL 2026 WINNER PREDICTOR
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div style='color:#7A8BA0; margin-bottom:1.5rem;'>XGBoost model · <b>{len(matches):,}</b> matches ({int(matches['season'].min())}–{int(matches['season'].max())}) · Accuracy: <b style='color:#22c55e'>{accuracy:.1%}</b> · H2H feature enabled</div>", unsafe_allow_html=True)

    # ── Stat cards ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Seasons",    matches["season"].nunique())
    c2.metric("🏟️ Matches",    f"{len(matches):,}")
    c3.metric("🏆 Teams",      len(all_teams))
    c4.metric("🤖 Accuracy",   f"{accuracy:.1%}")

    st.markdown("---")
    st.subheader("⚡ Head-to-Head Match Predictor")
    st.markdown("<div style='color:#7A8BA0; margin-bottom:1rem;'>Pick any two teams — the model uses win rates, recent form, titles won, and head-to-head history.</div>", unsafe_allow_html=True)

    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        t1_idx = all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0
        team1  = st.selectbox("🏏 Team 1", all_teams, index=t1_idx)
    with col_vs:
        st.markdown("<div style='text-align:center; padding-top:1.8rem; font-size:1.4rem; font-weight:700; color:#FF6B1A;'>VS</div>", unsafe_allow_html=True)
    with col_b:
        others  = [t for t in all_teams if t != team1]
        t2_def  = "Chennai Super Kings" if "Chennai Super Kings" in others else others[0]
        t2_idx  = others.index(t2_def)
        team2   = st.selectbox("🏏 Team 2", others, index=t2_idx)

    toss_opt = st.radio("🪙 Toss Winner", [team1, team2, "Not decided (50/50)"], horizontal=True)
    toss_val = 1.0 if toss_opt == team1 else (0.0 if toss_opt == team2 else 0.5)

    if st.button("🔮 Predict Match Winner", use_container_width=True, type="primary"):
        if team1 not in features.index or team2 not in features.index:
            st.error("Not enough historical data for one of these teams.")
        else:
            f1 = features.loc[team1]
            f2 = features.loc[team2]
            p1 = win_prob(model, matches, f1, f2, team1, team2, toss_val)
            p2 = 1 - p1

            winner = team1 if p1 >= p2 else team2
            t1i    = get_team_info(team1)
            t2i    = get_team_info(team2)
            wp     = max(p1, p2)

            st.markdown("---")

            # Champion banner
            st.markdown(f"""
            <div class='champion-banner'>
                <div style='margin-bottom:8px;'>{team_logo_html(winner, 72)}</div>
                <div style='font-family:Rajdhani,sans-serif; font-size:2rem; font-weight:700; color:#FFB800;'>
                    {winner}
                </div>
                <div style='color:#7A8BA0; margin-top:4px;'>wins with <b style='color:#FFB800'>{wp*100:.1f}%</b> probability</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Probability bars
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"""
                <div class='team-card'>
                    <div>{team_logo_html(team1, 52)}</div>
                    <div style='font-weight:700; font-size:1.1rem; margin:8px 0;'>{team1}</div>
                    <div style='font-size:2rem; font-weight:700; color:{"#FFB800" if p1>p2 else "#7A8BA0"}'>{p1*100:.1f}%</div>
                    <div class='prob-bar-container'>
                        <div style='background:{"#FF6B1A" if p1>p2 else "#4A5568"}; height:100%; width:{p1*100:.0f}%; border-radius:999px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_r:
                st.markdown(f"""
                <div class='team-card'>
                    <div>{team_logo_html(team2, 52)}</div>
                    <div style='font-weight:700; font-size:1.1rem; margin:8px 0;'>{team2}</div>
                    <div style='font-size:2rem; font-weight:700; color:{"#FFB800" if p2>p1 else "#7A8BA0"}'>{p2*100:.1f}%</div>
                    <div class='prob-bar-container'>
                        <div style='background:{"#FF6B1A" if p2>p1 else "#4A5568"}; height:100%; width:{p2*100:.0f}%; border-radius:999px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # H2H history
            h2h_df = matches[
                ((matches["team1"] == team1) & (matches["team2"] == team2)) |
                ((matches["team1"] == team2) & (matches["team2"] == team1))
            ]
            t1w_h2h = len(h2h_df[h2h_df["winner"] == team1])
            t2w_h2h = len(h2h_df[h2h_df["winner"] == team2])
            h2h_rate = t1w_h2h / len(h2h_df) * 100 if len(h2h_df) > 0 else 50

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.subheader("📊 Team Stats")
                comp = pd.DataFrame({
                    "Stat"    : ["Overall Win Rate", "Recent Form (3y)", "IPL Titles", "Win Rate After Toss"],
                    team1     : [f"{f1['win_rate']:.1%}", f"{f1['recent_win_rate']:.1%}", int(f1['titles']), f"{f1['toss_to_win_rate']:.1%}"],
                    team2     : [f"{f2['win_rate']:.1%}", f"{f2['recent_win_rate']:.1%}", int(f2['titles']), f"{f2['toss_to_win_rate']:.1%}"],
                })
                st.dataframe(comp, use_container_width=True, hide_index=True)

            with col_s2:
                st.subheader(f"🤝 Head-to-Head Record")
                st.markdown(f"""
                <div style='text-align:center; padding:16px;'>
                    <div style='font-size:2.5rem; font-weight:700;'>
                        <span style='color:#FF6B1A'>{t1w_h2h}</span>
                        <span style='color:#7A8BA0; margin:0 12px;'>–</span>
                        <span style='color:#FF6B1A'>{t2w_h2h}</span>
                    </div>
                    <div style='color:#7A8BA0; margin:8px 0;'>{len(h2h_df)} total meetings</div>
                    <div class='prob-bar-container' style='height:12px;'>
                        <div style='background:linear-gradient(90deg,#FF6B1A,#FFB800); height:100%; width:{h2h_rate:.0f}%; border-radius:999px;'></div>
                    </div>
                    <div style='color:#7A8BA0; font-size:0.8rem; margin-top:6px;'>{team1} won {h2h_rate:.0f}% of meetings</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: 2026 STANDINGS
# ══════════════════════════════════════════════════════════════
elif "Standings" in page:
    st.markdown("<h1>📊 IPL 2026 Predicted Standings</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#7A8BA0; margin-bottom:1.5rem;'>Full round-robin simulation using XGBoost + H2H features.</div>", unsafe_allow_html=True)

    # Resolve team names
    actual_teams = []
    for t in IPL_2026_TEAMS:
        if t in features.index:
            actual_teams.append(t)
        else:
            for ft in features.index:
                if any(w in ft for w in t.split() if len(w) > 4):
                    actual_teams.append(ft)
                    break
    actual_teams = list(dict.fromkeys(actual_teams))

    with st.spinner("⚙️ Simulating full IPL 2026 season..."):
        pts = {t: 0.0 for t in actual_teams}
        for i, t1 in enumerate(actual_teams):
            for j, t2 in enumerate(actual_teams):
                if i >= j: continue
                if t1 not in features.index or t2 not in features.index: continue
                f1, f2 = features.loc[t1], features.loc[t2]
                p = win_prob(model, matches, f1, f2, t1, t2)
                pts[t1] += p * 2
                pts[t2] += (1 - p) * 2
        standings = pd.DataFrame({
            "Team": list(pts.keys()),
            "Expected Points": [round(v, 1) for v in pts.values()]
        }).sort_values("Expected Points", ascending=False).reset_index(drop=True)
        standings.index = standings.index + 1

    # Points table
    st.subheader("🏆 Points Table")
    for idx, row in standings.iterrows():
        team  = row["Team"]
        pts_v = row["Expected Points"]
        info  = get_team_info(team)
        max_pts = standings["Expected Points"].max()
        bar_w = pts_v / max_pts * 100

        if idx == 1:    rank_badge = "🥇"; bg = "rgba(255,184,0,0.08)"; border = "#FFB800"
        elif idx == 2:  rank_badge = "🥈"; bg = "rgba(255,107,26,0.08)"; border = "#FF6B1A"
        elif idx == 3:  rank_badge = "🥉"; bg = "rgba(255,107,26,0.05)"; border = "rgba(255,107,26,0.3)"
        elif idx <= 4:  rank_badge = f"#{idx}"; bg = "rgba(255,107,26,0.04)"; border = "rgba(255,107,26,0.2)"
        else:           rank_badge = f"#{idx}"; bg = "transparent"; border = "rgba(255,255,255,0.06)"

        qual_badge = "<span class='stat-pill'>✅ QUALIFIES</span>" if idx <= 4 else "<span style='color:#4A5568;font-size:0.8rem;'>eliminated</span>"

        st.markdown(f"""
        <div style='background:{bg}; border:1px solid {border}; border-radius:12px; padding:14px 20px; margin:5px 0; display:flex; align-items:center; gap:16px;'>
            <div style='font-size:1.3rem; width:36px; text-align:center;'>{rank_badge}</div>
            <div>{team_logo_html(team, 38)}</div>
            <div style='flex:1;'>
                <div style='font-weight:700; font-size:1rem;'>{team}</div>
                <div class='prob-bar-container' style='width:200px; margin-top:5px;'>
                    <div style='background:linear-gradient(90deg,#FF6B1A,#FFB800); height:100%; width:{bar_w:.0f}%; border-radius:999px;'></div>
                </div>
            </div>
            <div style='font-size:1.3rem; font-weight:700; color:#FFB800; min-width:60px; text-align:right;'>{pts_v:.1f}</div>
            <div style='min-width:120px; text-align:right;'>{qual_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Plotly bar chart
    fig = go.Figure()
    colors_bar = ["#FFB800" if i <= 4 else "#2A3A4A" for i in standings.index]
    fig.add_trace(go.Bar(
        x=standings["Expected Points"], y=standings["Team"],
        orientation="h", marker_color=colors_bar,
        marker_line_color="rgba(255,107,26,0.3)", marker_line_width=1,
        text=[f"{v:.1f}" for v in standings["Expected Points"]],
        textposition="outside", textfont=dict(color="white"),
    ))
    if len(standings) > 4:
        cutoff = (standings["Expected Points"].iloc[3] + standings["Expected Points"].iloc[4]) / 2
        fig.add_vline(x=cutoff, line_dash="dash", line_color="#FF6B1A",
                      annotation_text="Playoff cutoff", annotation_font_color="#FF6B1A")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, xaxis_title="Expected Points",
        yaxis=dict(categoryorder="array", categoryarray=standings["Team"].tolist()[::-1]),
        margin=dict(l=0, r=60, t=20, b=30), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Playoff bracket
    st.subheader("🏟️ Predicted Playoff Bracket")
    top4 = standings["Team"].head(4).tolist()
    if len(top4) == 4:
        def pm(t1, t2):
            if t1 not in features.index or t2 not in features.index: return t1, 0.5
            f1, f2 = features.loc[t1], features.loc[t2]
            p = win_prob(model, matches, f1, f2, t1, t2)
            return (t1 if p >= 0.5 else t2), max(p, 1-p)

        q1w, q1p   = pm(top4[0], top4[1])
        q1l        = top4[1] if q1w == top4[0] else top4[0]
        ew,  ep    = pm(top4[2], top4[3])
        q2w, q2p   = pm(q1l, ew)
        champ, fp  = pm(q1w, q2w)
        runner_up  = q2w if champ == q1w else q1w

        bracket_data = [
            ("Qualifier 1",  top4[0], top4[1], q1w,    q1p),
            ("Eliminator",   top4[2], top4[3], ew,      ep),
            ("Qualifier 2",  q1l,      ew,     q2w,    q2p),
            ("🏆 FINAL",     q1w,      q2w,    champ,  fp),
        ]
        for label, t1, t2, winner, prob in bracket_data:
            is_final = "FINAL" in label
            border   = "#FFB800" if is_final else "rgba(255,107,26,0.25)"
            bg       = "rgba(255,184,0,0.06)" if is_final else "var(--bg-card)"
            st.markdown(f"""
            <div style='background:{bg}; border:1px solid {border}; border-radius:12px; padding:14px 20px; margin:6px 0; display:flex; align-items:center; gap:12px;'>
                <div style='min-width:120px; color:#7A8BA0; font-size:0.85rem;'>{label}</div>
                <div style='flex:1; font-size:0.95rem;'>{team_logo_html(t1, 28)} {t1}  <span style='color:#4A5568;'>vs</span>  {team_logo_html(t2, 28)} {t2}</div>
                <div style='font-weight:700; color:{"#FFB800" if is_final else "#FF6B1A"};'>✅ {winner}</div>
                <div class='stat-pill'>{prob*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='champion-banner' style='margin-top:1.5rem;'>
            <div style='margin-bottom:8px;'>{team_logo_html(champ, 72)}</div>
            <div style='font-family:Rajdhani,sans-serif; font-size:2.2rem; font-weight:700; color:#FFB800;'>
                🏆 {champ.upper()} 🏆
            </div>
            <div style='color:#7A8BA0; margin-top:4px;'>Predicted IPL 2026 Champions · {fp*100:.0f}% Final win probability</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3: LIVE SCHEDULE
# ══════════════════════════════════════════════════════════════
elif "Schedule" in page:
    st.markdown("<h1>📅 IPL 2026 Live Schedule & Predictions</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#7A8BA0; margin-bottom:1.5rem;'>Fetches live match schedule from ESPN Cricinfo · Each upcoming match is run through the model automatically.</div>", unsafe_allow_html=True)

    col_refresh, col_status = st.columns([3, 7])
    with col_refresh:
        if st.button("🔄 Refresh Schedule", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("📡 Fetching live IPL 2026 schedule..."):
        schedule = fetch_ipl_2026_schedule()

    if schedule:
        with col_status:
            st.success(f"✅ Fetched {len(schedule)} matches from ESPN Cricinfo")

        # Separate completed vs upcoming
        completed = [m for m in schedule if m.get("winner")]
        upcoming  = [m for m in schedule if not m.get("winner") and m.get("team1") and m.get("team2")]
        live      = [m for m in schedule if "live" in m.get("status", "").lower() or "innings" in m.get("status", "").lower()]

        # ── Live matches ──
        if live:
            st.markdown("### 🔴 Live Now")
            for m in live:
                t1n = normalize_team_name(m["team1"], all_teams) or m["team1"]
                t2n = normalize_team_name(m["team2"], all_teams) or m["team2"]
                i1, i2 = get_team_info(t1n), get_team_info(t2n)
                st.markdown(f"""
                <div style='background:rgba(239,68,68,0.08); border:1px solid #ef4444; border-radius:12px; padding:16px 20px; margin:6px 0;'>
                    <span style='background:#ef4444; color:white; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;'>● LIVE</span>
                    <span style='margin-left:12px; font-weight:700;'>{team_logo_html(t1n, 26)} {t1n}  vs  {team_logo_html(t2n, 26)} {t2n}</span>
                    <span style='color:#7A8BA0; margin-left:12px; font-size:0.85rem;'>{m.get("status","")}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Upcoming matches with predictions ──
        if upcoming:
            st.markdown("### 🔮 Upcoming Matches — Model Predictions")
            for m in upcoming:
                t1_raw = m["team1"]
                t2_raw = m["team2"]
                t1n = normalize_team_name(t1_raw, all_teams)
                t2n = normalize_team_name(t2_raw, all_teams)

                date_str = m.get("date", "")
                venue    = m.get("venue", "")

                if t1n and t2n and t1n in features.index and t2n in features.index:
                    f1 = features.loc[t1n]
                    f2 = features.loc[t2n]
                    p1 = win_prob(model, matches, f1, f2, t1n, t2n)
                    p2 = 1 - p1
                    fav = t1n if p1 >= p2 else t2n
                    fav_prob = max(p1, p2)
                    i1, i2   = get_team_info(t1n), get_team_info(t2n)

                    bar1 = p1 * 100
                    bar2 = p2 * 100

                    st.markdown(f"""
                    <div style='background:#0f1623; border:1px solid rgba(255,107,26,0.2); border-radius:14px; padding:18px 22px; margin:8px 0;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
                            <div>
                                <span style='font-weight:700; font-size:1.05rem;'>{team_logo_html(t1n, 30)} {t1n}  <span style="color:#4A5568;">vs</span>  {team_logo_html(t2n, 30)} {t2n}</span>
                            </div>
                            <div style='text-align:right;'>
                                <span style='color:#7A8BA0; font-size:0.8rem;'>{date_str}</span>
                                {"<br><span style='color:#7A8BA0; font-size:0.75rem;'>📍 " + venue + "</span>" if venue else ""}
                            </div>
                        </div>
                        <div style='display:flex; align-items:center; gap:12px;'>
                            <div style='min-width:120px; font-size:0.85rem; color:{"#FFB800" if p1>p2 else "#7A8BA0"}; font-weight:{"700" if p1>p2 else "400"};'>{t1n.split()[-1]} {p1*100:.0f}%</div>
                            <div style='flex:1;'>
                                <div style='background:rgba(255,255,255,0.06); border-radius:999px; height:8px; overflow:hidden;'>
                                    <div style='background:linear-gradient(90deg,#FF6B1A,#FFB800); height:100%; width:{bar1:.0f}%; border-radius:999px;'></div>
                                </div>
                            </div>
                            <div style='min-width:120px; text-align:right; font-size:0.85rem; color:{"#FFB800" if p2>p1 else "#7A8BA0"}; font-weight:{"700" if p2>p1 else "400"};'>{p2*100:.0f}% {t2n.split()[-1]}</div>
                        </div>
                        <div style='margin-top:8px; color:#7A8BA0; font-size:0.8rem;'>🔮 Predicted winner: <b style='color:#FFB800;'>{team_logo_html(fav, 24)} {fav}</b> ({fav_prob*100:.0f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Can't predict — show raw data
                    st.markdown(f"""
                    <div style='background:#0f1623; border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px 20px; margin:6px 0; color:#7A8BA0;'>
                        🏏 {t1_raw}  vs  {t2_raw}  ·  {date_str}  <span style='font-size:0.8rem;'>(prediction unavailable for this matchup)</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Completed matches ──
        if completed:
            st.markdown("---")
            with st.expander(f"✅ Completed Matches ({len(completed)})", expanded=False):
                for m in completed[-20:]:  # Show last 20
                    t1n = normalize_team_name(m["team1"], all_teams) or m["team1"]
                    t2n = normalize_team_name(m["team2"], all_teams) or m["team2"]
                    wn  = normalize_team_name(m["winner"], all_teams) or m["winner"]
                    i_w = get_team_info(wn)
                    st.markdown(f"""
                    <div style='background:#0f1623; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px 16px; margin:4px 0; display:flex; justify-content:space-between; align-items:center;'>
                        <span style='color:#7A8BA0; font-size:0.85rem;'>{m.get("date","")}</span>
                        <span>{t1n}  vs  {t2n}</span>
                        <span style='color:#22c55e; font-weight:700;'>{team_logo_html(wn, 22)} {wn} won</span>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        with col_status:
            st.warning("⚠️ Could not fetch live schedule — using manual predictor below")

        st.markdown("---")
        st.subheader("🔮 Manual Match Predictor")
        st.markdown("<div style='color:#7A8BA0;'>Select any two 2026 teams to get an instant prediction.</div>", unsafe_allow_html=True)

        valid_2026 = [t for t in IPL_2026_TEAMS if t in features.index]
        c1, c2 = st.columns(2)
        with c1:
            pt1 = st.selectbox("Team 1", valid_2026, key="sched_t1")
        with c2:
            pt2 = st.selectbox("Team 2", [t for t in valid_2026 if t != pt1], key="sched_t2")

        if pt1 and pt2 and pt1 != pt2:
            f1, f2 = features.loc[pt1], features.loc[pt2]
            p1     = win_prob(model, matches, f1, f2, pt1, pt2)
            p2     = 1 - p1
            fav    = pt1 if p1 >= p2 else pt2
            i1, i2 = get_team_info(pt1), get_team_info(pt2)

            st.markdown(f"""
            <div class='champion-banner' style='margin-top:1rem;'>
                <div style='font-size:2rem; font-weight:700;'>{team_logo_html(pt1, 32)} {pt1}  <span style='color:#4A5568; font-size:1.5rem;'>vs</span>  {team_logo_html(pt2, 32)} {pt2}</div>
                <div style='margin-top:12px; font-family:Rajdhani,sans-serif; font-size:1.8rem; color:#FFB800;'>🔮 {fav} wins</div>
                <div style='color:#7A8BA0; margin-top:4px;'>{pt1} {p1*100:.0f}%  ·  {pt2} {p2*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4: HISTORY
# ══════════════════════════════════════════════════════════════
elif "History" in page:
    st.markdown("<h1>📈 IPL Historical Analysis</h1>", unsafe_allow_html=True)

    champs = pd.DataFrame(list(season_winners.items()), columns=["Season", "Champion"]).sort_values("Season")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Titles Won")
        title_counts = champs["Champion"].value_counts().reset_index()
        title_counts.columns = ["Team", "Titles"]
        fig = px.bar(title_counts, x="Titles", y="Team", orientation="h",
                     color="Titles", color_continuous_scale=["#1a2a3a", "#FF6B1A", "#FFB800"],
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(categoryorder="total ascending"), height=350,
                          margin=dict(l=0, r=0, t=10, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 All-Time Win Rate")
        top_wr = features.sort_values("win_rate", ascending=False).head(12).reset_index()
        fig = px.bar(top_wr, x="win_rate", y="team", orientation="h",
                     color="win_rate", color_continuous_scale=["#1a2a3a", "#FF6B1A", "#FFB800"],
                     template="plotly_dark", text=top_wr["win_rate"].map("{:.1%}".format))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, coloraxis_showscale=False,
                          xaxis_tickformat=".0%",
                          yaxis=dict(categoryorder="total ascending"), height=350,
                          margin=dict(l=0, r=60, t=10, b=30))
        fig.update_traces(textposition="outside", textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)

    # Season-by-season champions table
    st.subheader("🗓️ Season-by-Season Champions")
    st.dataframe(champs.sort_values("Season", ascending=False), use_container_width=True, hide_index=True)

    # Matches per season
    st.subheader("📅 Matches Per Season")
    mps = matches.groupby("season").size().reset_index(name="Matches")
    fig = px.bar(mps, x="season", y="Matches", color="Matches",
                 color_continuous_scale=["#1a3a2a", "#FF6B1A", "#FFB800"],
                 template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, showlegend=False, height=300,
                      margin=dict(l=0, r=0, t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Toss analysis
    st.subheader("🪙 Toss Impact")
    toss_win = len(matches[matches["toss_winner"] == matches["winner"]])
    toss_pct = toss_win / len(matches) * 100
    c1, c2 = st.columns(2)
    c1.metric("Toss winner also won match",  f"{toss_pct:.1f}%")
    c2.metric("Toss winner lost the match",  f"{100-toss_pct:.1f}%")


# ══════════════════════════════════════════════════════════════
# PAGE 5: ABOUT
# ══════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("<h1>ℹ️ About This App</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0f1623; border:1px solid rgba(255,107,26,0.2); border-radius:14px; padding:28px; max-width:800px;'>

    <h3>🏏 IPL 2026 Winner Predictor — v2</h3>
    <p style='color:#7A8BA0;'>Machine Learning-powered IPL 2026 prediction using 17+ seasons of historical data.</p>

    <hr style='border-color:rgba(255,107,26,0.15); margin:20px 0;'>

    <h3>🤖 What's New in v2</h3>

    <div style='margin:12px 0;'>
        <span class='stat-pill'>★ XGBoost</span>&nbsp;
        Replaced Random Forest with gradient boosting — typically 3–8% more accurate on sports tabular data
    </div>
    <div style='margin:12px 0;'>
        <span class='stat-pill'>★ H2H Feature</span>&nbsp;
        Head-to-head win rate between any two teams, computed on past data only (no leakage)
    </div>
    <div style='margin:12px 0;'>
        <span class='stat-pill'>★ Hyperparameter Tuning</span>&nbsp;
        RandomizedSearchCV with StratifiedKFold finds optimal XGBoost parameters
    </div>
    <div style='margin:12px 0;'>
        <span class='stat-pill'>★ Live Schedule</span>&nbsp;
        Fetches IPL 2026 fixtures from ESPN Cricinfo's public API and runs predictions on each match
    </div>
    <div style='margin:12px 0;'>
        <span class='stat-pill'>★ Dark Theme</span>&nbsp;
        Full IPL-branded dark UI with Plotly interactive charts
    </div>

    <hr style='border-color:rgba(255,107,26,0.15); margin:20px 0;'>

    <h3>📊 All Features Used</h3>
    <ul style='color:#7A8BA0; line-height:1.9;'>
        <li>Overall win rate</li>
        <li>Recent form (last 3 seasons)</li>
        <li>IPL titles won historically</li>
        <li>Toss win rate</li>
        <li>Win rate after winning toss</li>
        <li>Seasons of experience</li>
        <li><b style='color:#FF6B1A;'>★ Head-to-head win rate vs opponent</b></li>
    </ul>

    <hr style='border-color:rgba(255,107,26,0.15); margin:20px 0;'>

    <h3>🛠️ Tech Stack</h3>
    <p style='color:#7A8BA0;'>Python · Streamlit · XGBoost · Scikit-learn · Plotly · Pandas · Requests</p>

    <h3>📁 Dataset</h3>
    <p style='color:#7A8BA0;'>IPL Complete Dataset (2008–2025) · Kaggle · Chaitanya · Usability 10.0</p>


    <hr style='border-color:rgba(255,107,26,0.15); margin:20px 0;'>

    <h3>????? Creator Profile</h3>
    <p style='color:#7A8BA0; line-height:1.8;'>
        Built by a Data Science and Machine Learning enthusiast focused on practical AI projects,
        predictive modeling, analytics dashboards, and end-to-end Python applications.
        This project reflects a hands-on approach to solving real-world sports prediction problems.
    </p>


    <hr style='border-color:rgba(255,107,26,0.15); margin:20px 0;'>

    <h3>?? Connect</h3>
    <p style='color:#7A8BA0; line-height:1.8; margin-bottom:6px;'>
        Name: <b style='color:#EAEAEA;'>Tushar Magar</b>
    </p>
    <p style='color:#7A8BA0; line-height:1.8;'>
        LinkedIn: <a href='https://www.linkedin.com/in/tushar-magar-7b80a2255' target='_blank' style='color:#FFB800; text-decoration:none;'>linkedin.com/in/tushar-magar-7b80a2255</a>
    </p>
    </div>
    """, unsafe_allow_html=True)
