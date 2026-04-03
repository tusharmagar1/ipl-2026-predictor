"""
STEP 3 — Train the ML Model  (v2 — XGBoost + Hyperparameter Tuning)
=====================================================================
Upgrades from v1:
  ★ XGBoost  instead of Random Forest (typically 3–8% more accurate on sports data)
  ★ StratifiedKFold cross-validation  (more reliable accuracy estimate)
  ★ RandomizedSearchCV               (finds best hyperparameters automatically)
  ★ Head-to-Head feature              (from step2 v2)

XGBoost builds trees sequentially — each new tree corrects the mistakes
of the previous ones. This "boosting" strategy tends to outperform
Random Forest, especially on structured tabular data like ours.

Run: python step3_train.py
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ─────────────────────────────────────────────
# Feature columns — MUST match step2_features.py
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "t1_win_rate", "t1_recent_win_rate", "t1_titles",
    "t1_toss_win_rate", "t1_toss2win_rate", "t1_seasons",
    "t2_win_rate", "t2_recent_win_rate", "t2_titles",
    "t2_toss_win_rate", "t2_toss2win_rate", "t2_seasons",
    "toss_winner_is_t1",
    "t1_h2h_win_rate",       # ★ NEW
]

FEATURE_LABELS = [
    "T1 Win Rate", "T1 Recent Win Rate", "T1 Titles Won",
    "T1 Toss Win Rate", "T1 Win After Toss", "T1 Seasons Played",
    "T2 Win Rate", "T2 Recent Win Rate", "T2 Titles Won",
    "T2 Toss Win Rate", "T2 Win After Toss", "T2 Seasons Played",
    "Toss Winner is Team1",
    "T1 Head-to-Head Win Rate",   # ★ NEW
]


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 3: Training XGBoost Model (v2)")
    print("=" * 60)

    # ── Load training data ──
    try:
        data = pd.read_csv("training_data.csv")
    except FileNotFoundError:
        print("\n❌ training_data.csv not found. Run step2_features.py first!")
        exit(1)

    # Back-compat: if built with old step2 (no H2H column), add neutral default
    if "t1_h2h_win_rate" not in data.columns:
        print("   ⚠️  H2H column missing — adding neutral 0.5 (re-run step2 for full benefit)")
        data["t1_h2h_win_rate"] = 0.5

    print(f"\n✅ Loaded training data: {len(data)} matches")

    X = data[FEATURE_COLS]
    y = data["team1_wins"]

    # ── Train / Test split (stratified to keep label balance) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples : {len(X_train)}")
    print(f"   Test samples     : {len(X_test)}")

    # ═══════════════════════════════════════════
    # BASELINE: Random Forest (for comparison)
    # ═══════════════════════════════════════════
    print("\n🌲 Training baseline Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"   Random Forest test accuracy: {rf_acc:.2%}")

    # ═══════════════════════════════════════════
    # STEP A: Hyperparameter search with XGBoost
    # ═══════════════════════════════════════════
    print("\n🔍 Running RandomizedSearchCV on XGBoost (30 iterations × 5-fold CV)...")
    print("   This may take 1–3 minutes — worth it for better predictions!\n")

    param_dist = {
        "n_estimators"     : [100, 200, 300, 500],
        "max_depth"        : [3, 4, 5, 6],
        "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
        "subsample"        : [0.6, 0.8, 1.0],
        "colsample_bytree" : [0.6, 0.8, 1.0],
        "min_child_weight" : [1, 3, 5],
        "gamma"            : [0, 0.1, 0.3],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_base = XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator          = xgb_base,
        param_distributions = param_dist,
        n_iter             = 30,           # Try 30 random combos
        cv                 = cv_strategy,
        scoring            = "accuracy",
        random_state       = 42,
        n_jobs             = -1,
        verbose            = 1,
        refit              = True,         # Re-train best model on full train set
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    print(f"\n✅ Best hyperparameters found:")
    for k, v in search.best_params_.items():
        print(f"   {k:<25} = {v}")

    # ═══════════════════════════════════════════
    # STEP B: Evaluate best model
    # ═══════════════════════════════════════════
    y_pred = model.predict(X_test)
    xgb_acc = accuracy_score(y_test, y_pred)

    # Full cross-validation score on the best params
    cv_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring="accuracy")

    print(f"\n📊 Model Performance:")
    print(f"   Random Forest  test accuracy : {rf_acc:.2%}  (baseline)")
    print(f"   XGBoost        test accuracy : {xgb_acc:.2%}  ★")
    print(f"   XGBoost 5-Fold CV accuracy   : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    improvement = (xgb_acc - rf_acc) * 100
    sign = "+" if improvement >= 0 else ""
    print(f"   Improvement over baseline    : {sign}{improvement:.1f}%")
    print(f"\n   (A coin flip gives ~50%. Any score above 55% is meaningful!)")

    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Team 2 Wins", "Team 1 Wins"]))

    # ═══════════════════════════════════════════
    # Feature Importances
    # ═══════════════════════════════════════════
    importances = pd.Series(model.feature_importances_, index=FEATURE_LABELS)
    importances = importances.sort_values(ascending=False)

    print("\n🔍 Feature Importances (XGBoost):")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 80)
        print(f"   {feat:<35} {bar}  {imp:.3f}")

    # ── Save the model + metadata ──
    model_bundle = {
        "model"       : model,
        "features"    : FEATURE_COLS,
        "accuracy"    : xgb_acc,
        "cv_mean"     : cv_scores.mean(),
        "best_params" : search.best_params_,
    }
    with open("ipl_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)
    print("\n✅ Model bundle saved: ipl_model.pkl")

    # ═══════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("XGBoost Model Evaluation (v2)", fontsize=14, fontweight="bold")

    # Plot 1: Feature Importances
    importances.sort_values().plot(kind="barh", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("Feature Importances")
    axes[0].set_xlabel("Importance Score")

    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["T2 Wins", "T1 Wins"],
                yticklabels=["T2 Wins", "T1 Wins"])
    axes[1].set_title("Confusion Matrix")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    # Plot 3: RF vs XGBoost accuracy comparison
    bars = axes[2].bar(
        ["Random Forest\n(baseline)", f"XGBoost\n(tuned)"],
        [rf_acc * 100, xgb_acc * 100],
        color=["lightcoral", "steelblue"],
        edgecolor="black", width=0.5
    )
    axes[2].set_ylim(40, 80)
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("Model Comparison")
    axes[2].axhline(50, color="gray", linestyle="--", linewidth=1, label="Coin flip (50%)")
    axes[2].legend()
    for bar, val in zip(bars, [rf_acc * 100, xgb_acc * 100]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("step3_model_eval.png", dpi=150, bbox_inches="tight")
    print("✅ Chart saved: step3_model_eval.png")
    plt.show()

    print("\n✅ Step 3 complete! Run: python step4_predict.py\n")
