# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# === Load all artifacts from model.pkl ===
with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
encoders = artifacts["encoders"]
feature_columns = artifacts["feature_columns"]

# Allowed categories for dropdowns
allowed_transaction_types = list(encoders["TransactionType"].classes_)
allowed_locations = list(encoders["Location"].classes_)


@app.route("/")
def welcome():
    """Welcome page"""
    return render_template("welcome.html")


@app.route("/input", methods=["GET", "POST"])
def input_page():
    """Transaction input form"""
    if request.method == "POST":
        tid = request.form.get("TransactionID", "")
        amount = request.form.get("Amount", "")
        merchant = request.form.get("MerchantID", "")
        ttype = request.form.get("TransactionType", "")
        loc = request.form.get("Location", "")

        # --- Validate numeric input ---
        try:
            amount_val = float(amount)
        except ValueError:
            return render_template(
                "output.html",
                error="❌ Amount must be numeric."
            )

        # --- Encode categorical input ---
        enc_t = encoders["TransactionType"]
        enc_l = encoders["Location"]

        if str(ttype) not in list(enc_t.classes_):
            return render_template(
                "output.html",
                error=f"❌ TransactionType '{ttype}' not recognised."
            )
        if str(loc) not in list(enc_l.classes_):
            return render_template(
                "output.html",
                error=f"❌ Location '{loc}' not recognised."
            )

        ttype_enc = int(enc_t.transform([str(ttype)])[0])
        loc_enc = int(enc_l.transform([str(loc)])[0])

        # --- Construct feature vector ---
        input_dict = {
            "TransactionID": float(tid) if tid != "" else 0.0,
            "Amount": amount_val,
            "MerchantID": float(merchant) if merchant != "" else 0.0,
            "TransactionType": ttype_enc,
            "Location": loc_enc,
        }

        # Check missing features
        missing = [col for col in feature_columns if col not in input_dict]
        if missing:
            return render_template(
                "output.html",
                error=f"❌ Missing features: {missing}"
            )

        # Order features correctly
        features = np.array([[input_dict[col] for col in feature_columns]])

        # --- Scale and predict ---
        try:
            features_scaled = scaler.transform(features)
        except Exception as e:
            return render_template(
                "output.html",
                error=f"❌ Error scaling features: {e}"
            )

        pred = model.predict(features_scaled)[0]
        interpretation = "Not Fraud" if pred == 1 else "Fraud"

        return render_template(
            "output.html",
            prediction=int(pred),
            interpretation=interpretation,
            TransactionID=tid,
            Amount=amount_val,
            MerchantID=merchant,
            TransactionType=ttype,
            Location=loc,
        )

    # GET → show empty form
    return render_template(
        "input.html",
        allowed_transaction_types=allowed_transaction_types,
        allowed_locations=allowed_locations,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for JSON prediction"""
    data = request.json

    try:
        tid = float(data.get("TransactionID", 0))
        amount = float(data.get("Amount", 0))
        merchant = float(data.get("MerchantID", 0))
        ttype = data.get("TransactionType", "")
        loc = data.get("Location", "")
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {e}"}), 400

    # Encode categorical inputs
    try:
        ttype_enc = int(encoders["TransactionType"].transform([str(ttype)])[0])
        loc_enc = int(encoders["Location"].transform([str(loc)])[0])
    except Exception as e:
        return jsonify({"error": f"Encoding failed: {e}"}), 400

    # Build input vector
    input_dict = {
        "TransactionID": tid,
        "Amount": amount,
        "MerchantID": merchant,
        "TransactionType": ttype_enc,
        "Location": loc_enc,
    }
    features = np.array([[input_dict[col] for col in feature_columns]])
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    result = "Not Fraud" if pred == 1 else "Fraud"

    return jsonify({
        "prediction": int(pred),
        "result": result,
        "input": input_dict
    })


if __name__ == "__main__":
    # For local debugging (Render will use gunicorn)
    app.run(debug=True, host="0.0.0.0", port=5000)
