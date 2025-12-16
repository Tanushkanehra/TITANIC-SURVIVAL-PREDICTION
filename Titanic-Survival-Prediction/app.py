import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚢 Titanic Survival Prediction")

# ================= User Inputs =================

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 30)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# ================= Encoding =================

sex = 0 if sex == "Male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# ================= Feature Vector (MATCH TRAINING) =================

X = np.array([[
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked,
    family_size,
    is_alone
]])

# Scale Age and Fare ONLY
X[:, [2, 5]] = scaler.transform(X[:, [2, 5]])

# ================= Prediction =================

if st.button("Predict Survival"):
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    if prediction == 1:
        st.success(f"🎉 Passenger Survived (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Passenger Did Not Survive (Probability: {probability:.2f})")
