Project Description

This project is a Credit Card Fraud Detection system that uses Logistic Regression to classify transactions as Fraud (0) or Not Fraud (1). The model is trained on a real dataset, with preprocessing (encoding + scaling) applied. The deployment is built using Flask (backend) and HTML (frontend), providing a simple web interface with three pages:

Welcome Page → Introduction & navigation

Input Page → User enters transaction details

Output Page → Displays fraud prediction

All ML artifacts (model, scaler, encoders) are stored in a single pickle file (model.pkl) for easy loading during inference.
