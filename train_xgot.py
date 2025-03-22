import pandas as pd
import numpy as np
import joblib
import logging
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from statsbombpy import sb
import shap

DEBUG_MODE = False  # Set to False to use all data
MAX_MATCHES = 5 if DEBUG_MODE else None

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.info("🚀 Starting Enhanced xG Model Training")

competitions = [
    (37, 90),  # Premier League 2020/21
    (37, 42),  # Premier League 2019/20
    (43, 106), # Champions League 2020/21
    (11, 42)   # La Liga
]

shots_df = pd.DataFrame()
for comp_id, season_id in competitions:
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    if DEBUG_MODE:
        matches = matches.head(MAX_MATCHES)
    for match_id in matches["match_id"]:
        try:
            events = sb.events(match_id=match_id)
            shots = events[events["type"] == "Shot"].copy()

            if shots.empty:
                continue

            shot_locs = pd.DataFrame(shots["location"].tolist(), columns=["x", "y"])
            shots = pd.concat([shots.reset_index(drop=True), shot_locs], axis=1)
            shots_df = pd.concat([shots_df, shots], ignore_index=True)
            time.sleep(1.5)
        except Exception as e:
            logging.warning(f"Skipping match {match_id}: {e}")

logging.info(f"Collected {len(shots_df)} shots")

# Feature Engineering
shots_df = shots_df.copy()
goal_x, goal_y = 120, 40

# Ensure only existing columns are used
available_columns = shots_df.columns.tolist()

additional_features = {}
if "minute" in available_columns:
    additional_features["time_remaining"] = 90 - shots_df["minute"]
if "x" in available_columns and "y" in available_columns:
    additional_features["distance_to_goal"] = np.sqrt((goal_x - shots_df["x"])**2 + (goal_y - shots_df["y"])**2)
    additional_features["angle_to_goal"] = np.arctan2(abs(goal_y - shots_df["y"]), goal_x - shots_df["x"])
if "shot_outcome" in available_columns:
    additional_features["is_goal"] = (shots_df["shot_outcome"] == "Goal").astype(int)
if "under_pressure" in available_columns:
    additional_features["under_pressure"] = shots_df["under_pressure"].fillna(False).astype(int)
if "shot_first_time" in available_columns:
    additional_features["rebound"] = shots_df["shot_first_time"].fillna(False).astype(int)
if "freeze_frame" in available_columns:
    additional_features["goalkeeper_distance"] = shots_df["freeze_frame"].apply(
        lambda x: np.mean([np.sqrt((goal_x - p["location"][0])**2 + (goal_y - p["location"][1])**2) 
                              for p in x if isinstance(p, dict) and "position" in p and p["position"].get("name") == "Goalkeeper"])
        if isinstance(x, list) else np.nan
    )

additional_features_df = pd.DataFrame(additional_features)
shots_df = pd.concat([shots_df.reset_index(drop=True), additional_features_df], axis=1)

categorical_cols = [col for col in ["shot_type", "shot_body_part", "play_pattern"] if col in available_columns]
shots_df = pd.get_dummies(shots_df, columns=categorical_cols)

features = [col for col in ["x", "y", "distance_to_goal", "angle_to_goal", "time_remaining", "under_pressure", "rebound", "goalkeeper_distance"] if col in shots_df.columns] + [
    col for col in shots_df.columns if any(prefix in col for prefix in ["shot_type_", "shot_body_part_", "play_pattern_"])
]

X = shots_df[features].fillna(0)
y = shots_df["is_goal"] if "is_goal" in shots_df.columns else None

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.joblib")

# Handling class imbalance
if y is not None:
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    # Hyperparameter tuning using RandomizedSearchCV
    param_grid = {
        "n_estimators": [200, 500, 1000],
        "max_depth": [10, 20, 30, None],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "gamma": [0, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    grid_search = RandomizedSearchCV(model, param_grid, cv=5, scoring="roc_auc", n_iter=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_balanced, y_balanced)
    best_model = grid_search.best_estimator_

    # Cross-validation
    scores = cross_val_score(best_model, X_balanced, y_balanced, cv=10, scoring='roc_auc')
    logging.info(f"Average ROC AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    # SHAP feature importance
    explainer = shap.Explainer(best_model, X_balanced)
    shap_values = explainer(X_balanced)
    shap.summary_plot(shap_values, X_balanced, feature_names=features)

    # Save model and features
    joblib.dump(best_model, "xg_model.joblib")
    joblib.dump(features, "features.joblib")

    logging.info("✅ Enhanced xG Model Trained and Saved")
else:
    logging.warning("⚠️ 'is_goal' column is missing, skipping model training.")
