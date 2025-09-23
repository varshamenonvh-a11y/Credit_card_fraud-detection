# model.py
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
        X, y, train_size=0.30, random_state=42
    )

    # 5. Scale features
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 6. Train model (balanced for imbalanced data)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train_scaled, y_train)

    # 7. Accuracy check
    print("\n✅ Training Accuracy:", model.score(x_train_scaled, y_train))
    print("✅ Test Accuracy:", model.score(x_test_scaled, y_test))

    # 8. Sample predictions
    sample_preds = model.predict(x_test_scaled[:20])
    print("\n🔎 Sample Predictions (first 20 test rows):", sample_preds.tolist())
    print("🔎 Actual Labels:", y_test[:20].tolist())

    # 9. Save artifacts
    artifacts = {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": feature_columns,
    }

    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n💾 Model trained and saved to {OUTPUT_PICKLE}")


def test_prediction():
    # === Load artifacts ===
    with open(OUTPUT_PICKLE, "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    scaler = artifacts["scaler"]

    # === Example encoded input (already numeric) ===
    # TransactionID, Amount, MerchantID, TransactionType, Location
    input_tuple = (1, 4189.27, 688, 1, 7)

    input_array = np.array(input_tuple).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    print("\n🎯 Single Prediction:", prediction)
    if prediction == 0:
        print("❌ Fraud Detected")
    else:
        print("✅ Not Fraud")


if __name__ == "__main__":
    train_and_save()
    test_prediction()