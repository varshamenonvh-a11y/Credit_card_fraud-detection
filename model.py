# model.py
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# === CONFIG ===
CSV_PATH = r"C:/Users/user/OneDrive/Documents/VaStUfFs/DATA SCIENCE AND AI/AI/PROJECT_credit_fraud/credit_card_fraud_dataset.csv"
OUTPUT_PICKLE = "model.pkl"

def train_and_save():
    # 1. Load dataset
    credit = pd.read_csv(CSV_PATH)
    print("✅ Dataset loaded with shape:", credit.shape)
    
    # 2. Encode categorical features
    encoders = {}
    for col in ["TransactionType", "Location"]:
        le = LabelEncoder()
        credit[col] = le.fit_transform(credit[col].astype(str))
        encoders[col] = le
    
    # 3. Features and target
    X = credit.drop(["TransactionDate", "IsFraud"], axis=1, errors="ignore")
    y = credit["IsFraud"]
    
    feature_columns = list(X.columns)
    print("🔎 Feature columns:", feature_columns)
    
    # Show target distribution
    print("\n📊 Target distribution (IsFraud):")
    print(y.value_counts())
    
    # 4. Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.30, random_state=42, stratify=y
    )
    
    # 5. Scale features
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # 6. Hyperparameter tuning with GridSearchCV
    print("\n🔍 Starting hyperparameter tuning...")
    
    # Define parameter grid for LogisticRegression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],  # Regularization strength
        'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization type
        'solver': ['liblinear', 'lbfgs', 'saga'],  # Optimization algorithm
        'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 5}],  # Handle imbalance
        'max_iter': [1000, 2000, 3000]  # Maximum iterations
    }
    
    # Handle incompatible combinations
    # For penalty='elasticnet', we need solver='saga' and l1_ratio
    param_combinations = []
    
    # Standard combinations
    for C in param_grid['C']:
        for class_weight in param_grid['class_weight']:
            for max_iter in param_grid['max_iter']:
                # L2 penalty with lbfgs/liblinear/saga
                for solver in ['lbfgs', 'liblinear', 'saga']:
                    param_combinations.append({
                        'C': C, 'penalty': 'l2', 'solver': solver,
                        'class_weight': class_weight, 'max_iter': max_iter
                    })
                
                # L1 penalty with liblinear/saga only
                for solver in ['liblinear', 'saga']:
                    param_combinations.append({
                        'C': C, 'penalty': 'l1', 'solver': solver,
                        'class_weight': class_weight, 'max_iter': max_iter
                    })
                
                # Elasticnet penalty with saga only
                for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    param_combinations.append({
                        'C': C, 'penalty': 'elasticnet', 'solver': 'saga',
                        'l1_ratio': l1_ratio, 'class_weight': class_weight,
                        'max_iter': max_iter
                    })
    
    # Use StratifiedKFold for better cross-validation with imbalanced data
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Create base model
    base_model = LogisticRegression(random_state=42)
    
    # Perform grid search with a subset of combinations to avoid memory issues
    # Take every 5th combination to make it manageable
    reduced_combinations = param_combinations[::5]
    print(f"🔍 Testing {len(reduced_combinations)} parameter combinations...")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=reduced_combinations,
        cv=cv,
        scoring='roc_auc',  # Good metric for imbalanced data
        n_jobs=-1,  # Use all cores
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(x_train_scaled, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    print("\n🎯 Best Parameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\n🏆 Best Cross-Validation ROC-AUC Score: {grid_search.best_score_:.4f}")
    
    # 7. Evaluate the best model
    train_accuracy = best_model.score(x_train_scaled, y_train)
    test_accuracy = best_model.score(x_test_scaled, y_test)
    
    print(f"\n✅ Training Accuracy: {train_accuracy:.4f}")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    
    # Additional metrics
    y_pred = best_model.predict(x_test_scaled)
    y_pred_proba = best_model.predict_proba(x_test_scaled)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"📈 ROC-AUC Score: {roc_auc:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 8. Sample predictions
    sample_preds = best_model.predict(x_test_scaled[:20])
    print("\n🔎 Sample Predictions (first 20 test rows):", sample_preds.tolist())
    print("🔎 Actual Labels:", y_test[:20].tolist())
    
    # 9. Save artifacts
    artifacts = {
        "model": best_model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": feature_columns,
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_
    }
    
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(artifacts, f)
    
    print(f"\n💾 Enhanced model trained and saved to {OUTPUT_PICKLE}")
    return best_model, scaler

def test_prediction():
    # === Load artifacts ===
    with open(OUTPUT_PICKLE, "rb") as f:
        artifacts = pickle.load(f)
    
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    
    print("\n🔧 Model Configuration:")
    print(f"   Best Parameters: {artifacts.get('best_params', 'N/A')}")
    print(f"   CV ROC-AUC Score: {artifacts.get('cv_score', 'N/A'):.4f}")
    
    # === Example encoded input (already numeric) ===
    # TransactionID, Amount, MerchantID, TransactionType, Location
    input_tuple = (1, 4189.27, 688, 1, 7)
    
    input_array = np.array(input_tuple).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    print(f"\n🎯 Single Prediction: {prediction}")
    print(f"🎯 Prediction Probabilities: [Not Fraud: {prediction_proba[0]:.3f}, Fraud: {prediction_proba[1]:.3f}]")
    
    if prediction == 1:  # Note: I fixed the logic here - 1 should be fraud
        print("❌ Fraud Detected")
    else:
        print("✅ Not Fraud")

if __name__ == "__main__":
    train_and_save()
    test_prediction()