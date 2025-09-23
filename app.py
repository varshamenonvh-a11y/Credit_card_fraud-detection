# app.py
from flask import Flask, render_template, request, redirect, url_for
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
    return render_template("welcome.html")


@app.route("/input", methods=["GET", "POST"])
def input_page():
    if request.method == "POST":
        form = request.form
        params = {
            "TransactionID": form.get("TransactionID", ""),
            "Amount": form.get("Amount", ""),
            "MerchantID": form.get("MerchantID", ""),
            "TransactionType": form.get("TransactionType", ""),
            "Location": form.get("Location", "")
        }
        return redirect(url_for("predict", **params))

    return render_template(
        "input.html",
        allowed_transaction_types=allowed_transaction_types,
        allowed_locations=allowed_locations,
    )


@app.route("/predict")
def predict():
    tid = request.args.get("TransactionID", "")
    amount = request.args.get("Amount", "")
    merchant = request.args.get("MerchantID", "")
    ttype = request.args.get("TransactionType", "")
    loc = request.args.get("Location", "")

    # --- Validate numeric input ---
    try:
        amount_val = float(amount)
    except ValueError:
        return render_template("output.html", error="❌ Amount must be numeric.")

    # --- Encode categorical input ---
    enc_t = encoders["TransactionType"]
    enc_l = encoders["Location"]

    if str(ttype) not in list(enc_t.classes_):
        return render_template("output.html", error=f"❌ TransactionType '{ttype}' not recognised.")
    if str(loc) not in list(enc_l.classes_):
        return render_template("output.html", error=f"❌ Location '{loc}' not recognised.")

    ttype_enc = int(enc_t.transform([str(ttype)])[0])
    loc_enc = int(enc_l.transform([str(loc)])[0])

    # --- Construct feature vector with consistency check ---
    input_dict = {
        "TransactionID": float(tid) if tid != "" else 0.0,
        "Amount": amount_val,
        "MerchantID": float(merchant) if merchant != "" else 0.0,
        "TransactionType": ttype_enc,
        "Location": loc_enc,
    }

    # Check that all required features are present
    missing = [col for col in feature_columns if col not in input_dict]
    if missing:
        return render_template("output.html", error=f"❌ Missing features: {missing}")

    # Order features correctly
    features = np.array([[input_dict[col] for col in feature_columns]])

    # --- Scale and predict ---
    try:
        features_scaled = scaler.transform(features)
    except Exception as e:
        return render_template("output.html", error=f"❌ Error scaling features: {e}")

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


if __name__ == "__main__":
    app.run(debug=True)