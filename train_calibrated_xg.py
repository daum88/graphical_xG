"""
Fixed Realistic xG Model with Better Calibration
"""
import pandas as pd
import numpy as np
import joblib
import logging
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def create_calibrated_xg_data(n_samples=20000):
    """Create properly calibrated xG data"""
    logger.info(f"🎯 Creating {n_samples} calibrated xG samples...")
    
    np.random.seed(42)
    shots = []
    
    for _ in range(n_samples):
        # Generate position
        if np.random.random() < 0.05:  # 5% penalties
            x, y = 108, 40 + np.random.normal(0, 0.5)
            shot_type = 'Penalty'
            under_pressure = 0
            base_xg = 0.76  # Real penalty conversion rate
        else:
            # Non-penalty shots
            shot_type = np.random.choice(['Open Play', 'Free Kick', 'Corner'], p=[0.85, 0.1, 0.05])
            
            # Generate realistic positions
            if shot_type == 'Free Kick':
                distance = np.random.gamma(2, 8) + 16
                angle = np.random.normal(0, 0.7)
            else:
                distance = np.random.exponential(10) + 6
                angle = np.random.normal(0, 0.5)
            
            x = max(75, min(119, 120 - distance * np.cos(angle)))
            y = max(5, min(75, 40 + distance * np.sin(angle)))
            
            under_pressure = int(np.random.random() < 0.35)
            
            # Calculate distance and angle
            dist = np.sqrt((120 - x)**2 + (40 - y)**2)
            angle_deg = np.degrees(np.arctan2(abs(40 - y), 120 - x))
            
            # Realistic xG based on StatsBomb data patterns
            if dist < 6:
                base_xg = 0.35
            elif dist < 10:
                base_xg = 0.20
            elif dist < 12:
                base_xg = 0.12
            elif dist < 16:
                base_xg = 0.08
            elif dist < 20:
                base_xg = 0.05
            elif dist < 25:
                base_xg = 0.03
            else:
                base_xg = 0.015
            
            # Angle penalty
            angle_factor = max(0.3, 1 - angle_deg / 45)
            base_xg *= angle_factor
            
            # Pressure penalty
            if under_pressure:
                base_xg *= 0.8
        
        # Other features
        body_part = np.random.choice(['Right Foot', 'Left Foot', 'Head', 'Other'], 
                                   p=[0.45, 0.35, 0.15, 0.05])
        
        if body_part == 'Head':
            base_xg *= 0.75
        elif body_part == 'Other':
            base_xg *= 0.6
            
        minute = np.random.randint(1, 91)
        rebound = int(np.random.random() < 0.08)
        
        # Ensure realistic bounds
        final_xg = max(0.001, min(0.95, base_xg + np.random.normal(0, 0.01)))
        
        # Generate goal based on xG
        is_goal = int(np.random.random() < final_xg)
        
        shots.append({
            'x': x, 'y': y,
            'shot_type': shot_type,
            'shot_body_part': body_part,
            'under_pressure': under_pressure,
            'minute': minute,
            'time_remaining': 90 - minute,
            'rebound': rebound,
            'is_goal': is_goal,
            'true_xg': final_xg  # Store for calibration
        })
    
    df = pd.DataFrame(shots)
    
    logger.info(f"✅ Created data:")
    logger.info(f"   Conversion rate: {df['is_goal'].mean():.1%}")
    logger.info(f"   Average xG: {df['true_xg'].mean():.3f}")
    logger.info(f"   Penalty conversion: {df[df['shot_type'] == 'Penalty']['is_goal'].mean():.1%}")
    
    return df

def create_simple_features(df):
    """Create simple, reliable features"""
    logger.info("🔧 Creating features...")
    
    df = df.copy()
    
    # Basic features
    df['distance_to_goal'] = np.sqrt((120 - df['x'])**2 + (40 - df['y'])**2)
    df['angle_to_goal'] = np.arctan2(np.abs(40 - df['y']), 120 - df['x'])
    df['angle_degrees'] = np.degrees(df['angle_to_goal'])
    
    # Simple bins
    df['is_penalty'] = (df['shot_type'] == 'Penalty').astype(int)
    df['is_close'] = (df['distance_to_goal'] < 12).astype(int)
    df['is_central'] = (df['angle_degrees'] < 20).astype(int)
    df['is_header'] = (df['shot_body_part'] == 'Head').astype(int)
    df['is_weak_foot'] = (df['shot_body_part'].isin(['Other'])).astype(int)
    
    # One-hot encode main categoricals
    df_encoded = pd.get_dummies(df, columns=['shot_type', 'shot_body_part'], drop_first=True)
    
    # Select final features
    feature_cols = [
        'x', 'y', 'distance_to_goal', 'angle_degrees', 
        'under_pressure', 'time_remaining', 'rebound',
        'is_penalty', 'is_close', 'is_central', 'is_header', 'is_weak_foot'
    ]
    
    # Add encoded features
    encoded_cols = [col for col in df_encoded.columns 
                   if col.startswith(('shot_type_', 'shot_body_part_'))]
    feature_cols.extend(encoded_cols)
    
    # Ensure all exist
    feature_cols = [col for col in feature_cols if col in df_encoded.columns]
    
    logger.info(f"✅ Features: {len(feature_cols)}")
    return df_encoded, feature_cols

def train_calibrated_model(df, features):
    """Train and calibrate the model"""
    logger.info("🎯 Training calibrated model...")
    
    X = df[features].fillna(0)
    y = df['is_goal']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Light SMOTE to balance but not overwhelm
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    logger.info(f"   Balanced samples: {len(X_balanced):,}")
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=12, 
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_balanced, y_balanced)
    
    # Calibrate probabilities using test set
    logger.info("🎯 Calibrating probabilities...")
    
    calibrated_clf = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    calibrated_clf.fit(X_train_scaled, y_train)  # Use original training data
    
    # Test performance
    y_pred_proba = calibrated_clf.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"✅ Test AUC: {test_auc:.3f}")
    
    return calibrated_clf, scaler, features

def test_realistic_scenarios(model, scaler, features):
    """Test with hand-crafted realistic scenarios"""
    logger.info("🧪 Testing realistic scenarios...")
    
    scenarios = [
        # Basic info for creating feature vector
        ('Penalty', {'x': 108, 'y': 40, 'shot_type': 'Penalty', 'shot_body_part': 'Right Foot', 'under_pressure': 0}, 0.72, 0.82),
        ('Tap-in', {'x': 118, 'y': 40, 'shot_type': 'Open Play', 'shot_body_part': 'Right Foot', 'under_pressure': 0}, 0.45, 0.75),
        ('Close central', {'x': 112, 'y': 40, 'shot_type': 'Open Play', 'shot_body_part': 'Right Foot', 'under_pressure': 0}, 0.15, 0.30),
        ('Box edge', {'x': 102, 'y': 40, 'shot_type': 'Open Play', 'shot_body_part': 'Right Foot', 'under_pressure': 0}, 0.06, 0.12),
        ('Wide angle', {'x': 110, 'y': 55, 'shot_type': 'Open Play', 'shot_body_part': 'Right Foot', 'under_pressure': 0}, 0.03, 0.08),
        ('Long range', {'x': 90, 'y': 40, 'shot_type': 'Open Play', 'shot_body_part': 'Right Foot', 'under_pressure': 0}, 0.01, 0.04),
        ('Header close', {'x': 112, 'y': 40, 'shot_type': 'Open Play', 'shot_body_part': 'Head', 'under_pressure': 0}, 0.10, 0.22),
        ('Under pressure', {'x': 112, 'y': 40, 'shot_type': 'Open Play', 'shot_body_part': 'Right Foot', 'under_pressure': 1}, 0.10, 0.20),
    ]
    
    realistic_count = 0
    
    for name, base_features, min_xg, max_xg in scenarios:
        # Create feature vector matching training data
        feature_vector = {col: 0 for col in features}
        
        # Set basic features
        x, y = base_features['x'], base_features['y']
        feature_vector.update({
            'x': x,
            'y': y,
            'distance_to_goal': np.sqrt((120 - x)**2 + (40 - y)**2),
            'angle_degrees': np.degrees(np.arctan2(abs(40 - y), 120 - x)),
            'under_pressure': base_features.get('under_pressure', 0),
            'time_remaining': 45,
            'rebound': 0,
            'is_penalty': 1 if base_features.get('shot_type') == 'Penalty' else 0,
            'is_close': 1 if np.sqrt((120 - x)**2 + (40 - y)**2) < 12 else 0,
            'is_central': 1 if np.degrees(np.arctan2(abs(40 - y), 120 - x)) < 20 else 0,
            'is_header': 1 if base_features.get('shot_body_part') == 'Head' else 0,
            'is_weak_foot': 1 if base_features.get('shot_body_part') == 'Other' else 0,
        })
        
        # Set categorical features
        shot_type = base_features.get('shot_type', 'Open Play')
        body_part = base_features.get('shot_body_part', 'Right Foot')
        
        for feature in features:
            if f'shot_type_{shot_type}' in feature and shot_type != 'Open Play':  # Open Play is reference
                feature_vector[feature] = 1
            if f'shot_body_part_{body_part}' in feature and body_part != 'Right Foot':  # Right Foot is reference
                feature_vector[feature] = 1
        
        # Predict
        df_test = pd.DataFrame([feature_vector])
        X_scaled = scaler.transform(df_test[features])
        xg_pred = model.predict_proba(X_scaled)[0][1]
        
        # Check realism
        is_realistic = min_xg <= xg_pred <= max_xg
        if is_realistic:
            realistic_count += 1
        
        status = "✅" if is_realistic else "❌"
        logger.info(f"   {status} {name}: {xg_pred:.3f} (expected {min_xg:.2f}-{max_xg:.2f})")
    
    realism_score = realistic_count / len(scenarios)
    logger.info(f"🎯 Realism: {realism_score:.1%} ({realistic_count}/{len(scenarios)})")
    
    return realism_score >= 0.75  # Need 75% to pass

def main():
    """Main pipeline"""
    logger.info("🚀 Training Calibrated Realistic xG Model")
    
    try:
        # Create data
        df = create_calibrated_xg_data(20000)
        
        # Create features
        df_featured, features = create_simple_features(df)
        
        # Train model
        model, scaler, feature_list = train_calibrated_model(df_featured, features)
        
        # Test realism
        is_realistic = test_realistic_scenarios(model, scaler, feature_list)
        
        if is_realistic:
            # Save artifacts
            joblib.dump(model, "xg_model.joblib")
            joblib.dump(scaler, "scaler.joblib")
            joblib.dump(feature_list, "features.joblib")
            
            logger.info("🎉 SUCCESS! Realistic calibrated xG model saved!")
            logger.info("📊 Summary:")
            logger.info(f"   Model type: Calibrated {type(model.base_estimator).__name__}")
            logger.info(f"   Features: {len(feature_list)}")
            logger.info(f"   Training samples: {len(df):,}")
            logger.info("🎯 Ready to use!")
            
        else:
            logger.error("❌ Model not realistic enough")
            
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
